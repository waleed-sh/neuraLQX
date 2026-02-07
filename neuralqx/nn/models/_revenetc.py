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

"""
RevNet implementation with complex-aware layers when needed

This revision of RevNet addresses two dtype consistency issues arising during initialising RevNet
with jax.numpy.complex datatype. Namely:

    - ModReLU bias parameter is now created with the real part dtype of the input tensor, avoiding
      inadvertent promotion of the whole activation path to complex128
    - LayerNorm, instead of BatchNorm, keeps real parameters (param_dtype = jnp.float32) even when
      the forward pass runs in complex precision, preserving the amplitude-only rescaling intent
"""

from __future__ import annotations

import math

from typing import Callable
from typing import Optional
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from ...lqx import LqxWCL

#
#
#   helpers and non-linearities


class ModReLU(nn.Module):
    """
    Holomorphic ReLU-style activation
    """

    bias_init: Callable = nn.initializers.zeros
    """Initialiser for the bias"""

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        # for real tensors fall back to ReLU
        if not jnp.iscomplexobj(z):
            return nn.relu(z)

        # the bias lives in the real dtype corresponding to the complex tensor
        real_dtype = z.real.dtype

        b = self.param("b", self.bias_init, (1,), real_dtype)
        r = jnp.abs(z)
        scale = nn.relu(r + b) / (r + 1e-9)

        return scale * z


def complex_glorot_init(key: jax.random.PRNGKey, shape: Sequence[int], dtype):
    """
    Glorot var-preserving initialiser for complex weights
    """

    if len(shape) < 2:
        fan_in = fan_out = 1
    else:
        receptive = math.prod(shape[:-2]) if len(shape) > 2 else 1
        fan_in = shape[-2] * receptive
        fan_out = shape[-1]
    std = math.sqrt(2.0 / (fan_in + fan_out)) / math.sqrt(2.0)

    # use the matching real dtype for the chosen complex precision
    real_dtype = jnp.float32 if dtype == jnp.complex64 else jnp.float64

    key_r, key_i = jax.random.split(key)
    w_r = jax.random.normal(key_r, shape, dtype=real_dtype) * std
    w_i = jax.random.normal(key_i, shape, dtype=real_dtype) * std

    return (w_r + 1j * w_i).astype(dtype)


class RevNetComplex(nn.Module):
    """
    RevNet implementation supporting real and complex dtypes
    """

    __name__ = "RevNet"
    """The name of the network, required for logging purposes"""

    number_of_cnn_features: int
    """The number of features in the Conv layer"""

    number_of_cnn_blocks: int
    """The number of CNN blocks in the network"""

    lqx: LqxWCL
    """An lqx object to obtain relevant graph and cutoff information from"""

    rng_key: jax.random.PRNGKey
    """A PRNGKey for reproducibility of results, used to init layers"""

    dropout_rate: float = 0.0
    """Dropout rate used when dropout layers are utilised"""

    free_idxs: Optional[jax.Array] = None
    """The indices of the free (non-gauge fixed) edges"""

    dtype: jax.numpy.dtype = jax.numpy.float64
    """The dtype of the network (e.g. complex or real weights and biases)"""

    # safer than BatchNorm for complex values
    use_layer_norm: bool = False
    """A flag to determine whether normalisation layers should be used or not"""

    _default_R_activation: Callable = nn.hard_silu
    """The default activation function for real valued network"""

    reduction_ratio: Optional[int] = 2
    """The reduction ratio in the attention mechanism"""

    def __hash__(self):
        return id(self)

    def _is_complex(self):
        """
        Helper function that returns true if the network is of complex dtype
        :return:
        """
        return jnp.issubdtype(self.dtype, jnp.complexfloating)

    def _act(self):
        """
        A helper function that acts with the appropriate activation function dependent on dtype
        :return:
        """
        return ModReLU() if self._is_complex() else self._default_R_activation

    def setup(self):
        """
        Graph dependent setup for layer sizes and number of blocks
        """

        # TODO: put back the number_of_cnn_blocks setup

        # graph‑dependent sizes
        self.cutoff = self.lqx.H.core.cutoff
        self.number_of_vertices = self.lqx.graph.dual_nk_graph.n_nodes

        # channel schedule
        self.cnn_block_features = [self.number_of_cnn_features] + [
            self.number_of_cnn_features // (2 * i)
            for i in range(1, self.number_of_cnn_blocks)
        ]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # gauge‑fixed truncation of the input tensor
        if self.free_idxs is not None:
            x = jnp.take(x, self.free_idxs, axis=1)

        # CNN‑attention stack
        for idx in range(self.number_of_cnn_blocks):
            x = self.cnn_block(
                x,
                features=self.cnn_block_features[idx],
                kernel_size=1,
            )
            x = self.attention_block(x, reduction_ratio=self.reduction_ratio)

        return self.eval_block(x)

    def cnn_block(self, x: jnp.ndarray, features: int, kernel_size: int):
        """
        The CNN sub-block, applies a classic CNN phase to the given input by using a Conv layer,
        a normalisation, an activation and (possibly...) pooling
        """

        x = nn.Conv(
            features=features,
            kernel_size=(kernel_size,),
            padding="CIRCULAR",
            dtype=self.dtype,
            param_dtype=self.dtype,
            kernel_init=(
                complex_glorot_init
                if self._is_complex()
                else nn.initializers.lecun_normal()
            ),
        )(x)

        # (Optional) LayerNorm, param_dtype kept real
        if self.use_layer_norm:
            x = nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(x)

        # activation, dtype dependent
        x = self._act()(x)
        return x

    def attention_block(
        self,
        x: jnp.ndarray,
        reduction_ratio: int,
    ):
        """
        An attention sub-block. This essentially is a form of self-attention in the form of a
        squeeze and excite (SE) block. However, before the SE block we essentially aggregate the
        entire output of the CNN phase into one input
        """

        # real magnitude
        pooled = jnp.mean(jnp.abs(x), axis=1, keepdims=True)

        # squeeze
        gate = nn.Dense(features=x.shape[-1] // reduction_ratio, dtype=jnp.float32)(
            pooled
        )

        # activate
        gate = nn.relu(gate)

        # excite
        gate = nn.Dense(features=x.shape[-1], dtype=jnp.float32)(gate)
        gate = nn.sigmoid(gate)

        # broadcast
        return x * gate

    def eval_block(self, x: jnp.ndarray):
        """
        The evaluation block, essentially a simple FFN
        """

        x = nn.Dense(1, dtype=self.dtype, param_dtype=self.dtype)(x)
        x = self._act()(x)
        return jnp.sum(x, axis=1).squeeze()
