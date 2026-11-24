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


"""Shape-first initialization of NNX networks on the tensor mesh."""

from collections.abc import Callable
from collections.abc import Mapping

from flax import nnx

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from neuralqx.jax.tensor_parallel import network_sharding_enabled
from neuralqx.jax.tensor_parallel import tensor_axis_name
from neuralqx.jax.tensor_parallel import tensor_mesh
from neuralqx.jax.tensor_parallel import tensor_partition_count
from neuralqx.utils.errors import NeuralqxTypeError

from .plan import TensorParallelPlan
from .plan import plan_parameters
from .rules import model_rules


def tensor_parallel_plan(
    model: nnx.Module, *, parameter_rules: Mapping[str, P] | None = None
) -> TensorParallelPlan:
    """Inspect the tensor layouts chosen for an initialized or abstract NNX model.

    Architecture rules pair contraction axes where the model structure is
    known. Layer rules cover custom compositions of neuraLQX or native NNX
    layers. Explicit parameter metadata overrides those defaults. Finally,
    ``parameter_rules`` overrides matching slash-separated paths or glob
    patterns, using the same syntax as :class:`TensorParallel`.

    The result includes replicated parameters and validates exact divisibility.
    Inspect ``plan.parameters`` for individual decisions or call
    ``plan.place(parameters)`` to place a matching parameter tree. No model
    values are changed and no model evaluation is performed.

    Args:
        model: An NNX module, including a result of ``nnx.eval_shape``.
        parameter_rules: Optional explicit mapping to ``PartitionSpec`` objects.

    Returns:
        An immutable, validated plan for the configured tensor mesh.
    """
    if not isinstance(model, nnx.Module):
        raise NeuralqxTypeError("tensor_parallel_plan expects an NNX module.")
    rules = {
        "/".join(map(str, path)): spec
        for path, spec in model_rules(
            model, tensor_axis_name(), tensor_partition_count()
        ).items()
    }
    explicit = {} if parameter_rules is None else dict(parameter_rules)
    rules.update(explicit)
    return plan_parameters(
        nnx.state(model, nnx.Param),
        tuple(rules.items()),
        required_patterns=tuple(explicit),
    )


def init_tensor_parallel(
    factory: Callable[[], nnx.Module], *, parameter_rules: Mapping[str, P] | None = None
) -> nnx.Module:
    """Initialize an NNX network directly into its distributed parameter shards.

    Use this constructor for networks whose complete parameters would not fit
    on one device. Ordinary initialized models can still be passed directly to
    ``VariationalState`` and are partitioned there when network sharding is on.
    Linen and callable initializers passed to ``VariationalState`` already use
    shape-first initialization in tensor mode.

    ``factory`` must create a fresh deterministic model from a fixed seed or
    key. It is traced twice, so it must not mutate external RNG streams or rely
    on host-side random values. The array initializers must be JAX-traceable.
    Integer seeds passed to neuraLQX constructors meet these requirements.
    Global initialization shapes and PRNG semantics are preserved across mesh
    sizes. Floating-point reductions can still differ in their rounding order.

    Example::

        # Set NEURALQX_SHARDING=samples,operators,tensor before importing.
        model = init_tensor_parallel(lambda: MLP(128, (1024, 1024), rngs=7))
        state = VariationalState(sampler, model, n_samples=4096)

    Args:
        factory: Zero-argument callable returning a fresh NNX module.
        parameter_rules: Optional path-to-``PartitionSpec`` overrides, retained
            as a static execution policy on the returned model. Checkpoint
            restoration selects layouts from the target model and configuration.

    Returns:
        An initialized NNX model. When tensor parallelism is disabled this
        calls ``factory`` directly, without shape tracing or compilation.
    """
    if not network_sharding_enabled():
        return factory()
    mesh = tensor_mesh()
    with jax.set_mesh(mesh):
        abstract = nnx.eval_shape(factory)
        graphdef, _, abstract_state = nnx.split(abstract, nnx.Param, ...)
        plan = tensor_parallel_plan(abstract, parameter_rules=parameter_rules)
        layouts = (
            jax.tree.map(lambda spec: NamedSharding(mesh, spec), plan.specs),
            jax.tree.map(lambda _: NamedSharding(mesh, P()), abstract_state),
        )

        def initialize():
            _, params, state = nnx.split(factory(), nnx.Param, ...)
            return params, state

        parameters, state = jax.jit(initialize, out_shardings=layouts)()
        model = nnx.merge(graphdef, parameters, state)
        # Execution policy is separate from variable metadata. In particular,
        # automatic partitioning must not change portable checkpoint schemas.
        if parameter_rules:
            model._neuralqx_tensor_parallel_rules = tuple(parameter_rules.items())
        return model


__all__ = ["init_tensor_parallel", "tensor_parallel_plan"]
