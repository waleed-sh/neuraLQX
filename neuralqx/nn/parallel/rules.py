# Copyright (c) 2026 The neuraLQX Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Architecture and layer rules for tensor partitioning.

Rules describe storage and contraction axes, not distributed numerical kernels.
JAX retains global shapes and supplies the required collectives and derivatives.
"""

from collections.abc import Callable

from flax import nnx

from jax.sharding import PartitionSpec as P

_RULES: dict[type, Callable] = {}


def register_tensor_parallel_rule(module_type: type, *, replace: bool = False):
    """Register partition rules for an NNX module class.

    The decorated function accepts ``(module, axis_name, partitions)`` and
    returns a mapping from relative parameter paths to ``PartitionSpec``.
    Paths are tuples of attribute names and integer container indices.
    A rule can describe a complete architecture or one reusable layer.
    Architecture rules take precedence over descendant layer defaults.
    Explicit adapter rules take precedence over both.

    Register before constructing a variational state. Rule functions run on
    the host during planning and must not inspect parameter values.

    Args:
        module_type: NNX module class to recognize, including its subclasses.
        replace: Permit replacement of a previously registered rule.

    Returns:
        A decorator that registers and returns the supplied function.
    """
    if not isinstance(module_type, type) or not issubclass(module_type, nnx.Module):
        raise TypeError("Tensor parallel rules require an NNX Module class.")

    def register(fn):
        if module_type in _RULES and not replace:
            raise ValueError(
                f"A tensor-parallel rule already exists for {module_type.__name__}."
            )
        _RULES[module_type] = fn
        return fn

    return register


def affine_rule(module, axis, partitions, *, dimension=None):
    kernel = module.kernel
    shape = kernel.shape
    if dimension is None:
        candidates = (-1, -2) if shape[-1] > 2 else (-2, -1)
        dimension = next(
            (
                d
                for d in candidates
                if shape[d] >= partitions and shape[d] % partitions == 0
            ),
            None,
        )
    if dimension is not None and shape[dimension] % partitions:
        # Retain exact shapes when the preferred pairing cannot divide a
        # width. A divisible contraction dimension can still be partitioned.
        dimension = next(
            (d for d in (-2, -1) if shape[d] > 2 and shape[d] % partitions == 0), None
        )
    if dimension is None:
        return {("kernel",): P(), ("bias",): P()}
    dim = dimension % len(shape)
    spec = [None] * len(shape)
    spec[dim] = axis
    return {("kernel",): P(*spec), ("bias",): P(axis) if dim == len(shape) - 1 else P()}


def model_rules(model, axis, partitions):
    """Resolve registered rules on an NNX graph without mutating the model."""
    _register_defaults()
    rules = {}
    # Children first, then architecture overrides. Shared variables are
    # validated by the resulting unique parameter paths in the plan.
    modules = [
        (path, node)
        for path, node in nnx.iter_graph(model)
        if isinstance(node, nnx.Module)
    ]
    for path, node in sorted(modules, key=lambda item: len(item[0]), reverse=True):
        handler = next(
            (_RULES[cls] for cls in type(node).__mro__ if cls in _RULES), None
        )
        if handler is not None:
            rules.update(
                {
                    tuple(path) + tuple(key): spec
                    for key, spec in handler(node, axis, partitions).items()
                }
            )
    for path, node in nnx.iter_graph(model):
        if (
            isinstance(node, nnx.Param)
            and node.get_metadata().get("out_sharding") is not None
        ):
            rules[tuple(path)] = nnx.get_partition_spec(node).get_value()
    for path, node in sorted(modules, key=lambda item: len(item[0]), reverse=True):
        for pattern, spec in getattr(node, "_neuralqx_tensor_parallel_rules", ()):
            rules[(*path, *pattern.split("/"))] = spec
    return rules


def _register_defaults():
    if getattr(_register_defaults, "done", False):
        return
    from neuralqx.nn.models import CNN
    from neuralqx.nn.models import GNN
    from neuralqx.nn.models import MLP
    from neuralqx.nn.models import RBM
    from neuralqx.nn.models import AutoregressiveRNN
    from neuralqx.nn.models import Jastrow

    _RULES.setdefault(nnx.Linear, affine_rule)
    _RULES.setdefault(nnx.Conv, affine_rule)

    def sequential(m, a, n):
        out = {}
        affine_index = 0
        for i, layer in enumerate(m.layers):
            if isinstance(layer, (nnx.Linear, nnx.Conv)):
                dim = -1 if affine_index % 2 == 0 else -2
                if layer.kernel.shape[-1] <= 2:
                    dim = -2
                out.update(
                    {
                        ("layers", i, *k): v
                        for k, v in affine_rule(layer, a, n, dimension=dim).items()
                    }
                )
                affine_index += 1
        return out

    _RULES.setdefault(nnx.Sequential, sequential)
    _RULES.setdefault(
        nnx.Embed,
        lambda m, a, n: {("embedding",): P(None, a) if m.features % n == 0 else P()},
    )
    for cls in (nnx.LayerNorm, nnx.RMSNorm):
        _RULES.setdefault(
            cls,
            lambda m, a, n: {
                (key,): (
                    P(a)
                    if getattr(m, key, None) is not None
                    and getattr(m, key).shape[-1] % n == 0
                    else P()
                )
                for key in ("scale", "bias")
            },
        )

    def mlp(m, a, n):
        out = {}
        layers = [*m.hidden, m.head]
        for i, layer in enumerate(layers):
            # Paired column/row contractions avoid communicating the large
            # intermediate hidden tensor. A small trailing head is replicated.
            dim = 1 if i % 2 == 0 else 0
            entries = affine_rule(layer, a, n, dimension=dim)
            if i == len(layers) - 1 and i % 2 == 0:
                entries = {("kernel",): P(), ("bias",): P()}
            path = ("head",) if i == len(layers) - 1 else ("hidden", i)
            out.update({path + key: spec for key, spec in entries.items()})
        return out

    def gnn(m, a, n):
        out = {
            ("encoder", *k): v
            for k, v in affine_rule(m.encoder, a, n, dimension=1).items()
        }
        out[("node_embedding",)] = (
            P(None, a) if m.encoder.out_features % n == 0 else P()
        )
        for i, layer in enumerate(m.messages):
            for name in ("self_map", "neighbor_map"):
                out.update(
                    {
                        ("messages", i, name, *k): v
                        for k, v in affine_rule(
                            getattr(layer, name), a, n, dimension=0
                        ).items()
                    }
                )
        for name in ("readout", "head"):
            out.update(
                {
                    (name, *k): v
                    for k, v in affine_rule(getattr(m, name), a, n, dimension=0).items()
                }
            )
        return out

    def cnn(m, a, n):
        out = {}
        for i, layer in enumerate(m.convolutions):
            out.update(
                {
                    ("convolutions", i, *k): v
                    for k, v in affine_rule(
                        layer, a, n, dimension=-1 if i == 0 else -2
                    ).items()
                }
            )
        for name in ("readout", "head"):
            out.update(
                {
                    (name, *k): v
                    for k, v in affine_rule(getattr(m, name), a, n, dimension=0).items()
                }
            )
        return out

    _RULES.setdefault(MLP, mlp)
    _RULES.setdefault(GNN, gnn)
    _RULES.setdefault(CNN, cnn)
    _RULES.setdefault(
        RBM,
        lambda m, a, n: {
            **{
                ("hidden", *k): v
                for k, v in affine_rule(m.hidden, a, n, dimension=1).items()
            },
            ("visible", "kernel"): P(),
            ("visible", "bias"): P(),
        },
    )
    _RULES.setdefault(
        Jastrow,
        lambda m, a, n: {
            ("coupling",): P(a, None, None) if m.in_features % n == 0 else P(),
            ("linear",): P(a, None) if m.in_features % n == 0 else P(),
        },
    )
    _RULES.setdefault(
        AutoregressiveRNN,
        lambda m, a, n: {
            **{
                ("recurrent", *k): v
                for k, v in affine_rule(m.recurrent, a, n, dimension=0).items()
            },
            **{
                ("probability", *k): v
                for k, v in affine_rule(m.probability, a, n, dimension=0).items()
            },
            **(
                {
                    ("phase", *k): v
                    for k, v in affine_rule(m.phase, a, n, dimension=0).items()
                }
                if m.phase is not None
                else {}
            ),
        },
    )
    _register_defaults.done = True


__all__ = ["register_tensor_parallel_rule"]
