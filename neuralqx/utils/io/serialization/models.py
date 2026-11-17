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


"""Constructor recipes for the built-in NNX wavefunctions and their Linen facade."""

import numpy as np

import jax

from .errors import ReconstructionError
from .registry import _register_codec as register_codec


def _model_config(model):
    from flax import nnx

    from neuralqx import nn

    leaves = jax.tree.leaves(nnx.state(model, nnx.Param))
    dtypes = {np.dtype(leaf.dtype) for leaf in leaves}
    if len(dtypes) != 1:
        raise ReconstructionError(
            "A model with mixed parameter dtypes needs an explicit loading template."
        )
    config = {
        "dtype": model.dtype,
        "param_dtype": dtypes.pop(),
        "connection_policy": model.connection_policy,
    }
    if isinstance(model, nn.MLP):
        config.update(
            in_features=model.in_features,
            hidden_features=tuple(layer.out_features for layer in model.hidden),
            activation=model.activation,
            precision=model.head.precision,
        )
    elif isinstance(model, nn.RBM):
        config.update(
            in_features=model.in_features,
            hidden_features=model.hidden_features,
            precision=model.hidden.precision,
        )
    elif isinstance(model, nn.Jastrow):
        config.update(in_features=model.in_features, precision=model.precision)
    elif isinstance(model, nn.CNN):
        config.update(
            lattice_shape=model.lattice_shape,
            features=tuple(layer.out_features for layer in model.convolutions),
            kernel_size=model.convolutions[0].kernel_size[0],
            activation=model.activation,
            precision=model.head.precision,
        )
    elif isinstance(model, nn.GNN):

        def host(value):
            value = value[...]
            if not value.is_fully_addressable:
                value = value.addressable_shards[0].data
            return np.asarray(value)

        config.update(
            n_nodes=model.n_nodes,
            edges=np.column_stack((host(model.senders), host(model.receivers))),
            node_features=model.node_features,
            width=model.encoder.out_features,
            depth=len(model.messages),
            directed=True,
            node_embeddings=model.node_embedding is not None,
            activation=model.activation,
            precision=model.head.precision,
        )
    elif isinstance(model, nn.AutoregressiveRNN):
        config.update(
            in_features=model.in_features,
            hidden_features=model.hidden_features,
            local_states=model.local_states[...],
            complex_output=model.phase is not None,
            precision=model.recurrent.precision,
        )
    if hasattr(model, "complex_output"):
        config["complex_output"] = model.complex_output
    if hasattr(model, "input_scale"):
        config["input_scale"] = model.input_scale
    return config


def install_model_codecs():
    """Register only known model classes and the official neuraLQX Linen bridge."""
    from flax import nnx

    from neuralqx import nn
    from neuralqx.nn.utils.bridge import LinenModel

    models = {
        cls.__name__: cls
        for cls in (nn.MLP, nn.RBM, nn.Jastrow, nn.CNN, nn.GNN, nn.AutoregressiveRNN)
    }
    for name, cls in models.items():
        register_codec(
            cls,
            name=f"neuralqx.nn.{name}",
            encode=_model_config,
            decode=lambda config, cls=cls: nnx.eval_shape(
                lambda: cls(**config, rngs=0)
            ),
        )

    def encode_linen(model):
        if models.get(model.nnx_class.__name__) is not model.nnx_class:
            raise ReconstructionError(
                "A custom bridged model needs its own reconstruction codec."
            )
        return {
            "model": model.nnx_class.__name__,
            "args": model.args,
            "kwargs": dict(model.kwargs),
            "skip_rng": model.skip_rng,
            "name": model.name,
        }

    def decode_linen(config):
        return nn.to_linen(
            models[config["model"]],
            *config["args"],
            **config["kwargs"],
            skip_rng=config["skip_rng"],
            name=config["name"],
        )

    register_codec(
        LinenModel,
        name="neuralqx.nn.LinenModel",
        encode=encode_linen,
        decode=decode_linen,
    )
