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
This file contains the implementation of the RevNet architecture
"""

import math
from dataclasses import field

from typing import Optional
from typing import Callable
from typing import Union
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

import netket
from netket.nn.activation import reim


from ...lqx import LqxWCL
from ...lqx.wcl.core import LqxWCL4D
from ...lqx.wcl.core import LqxWCL3D

InitLike = Optional[Union[float, int, Callable[..., Any]]]


#
#
#   Helpers


def _real_dtype_for_complex(dtype):
    """map complex dtype -> matching real dtype for initialiser draws."""
    if dtype == jnp.complex64:
        return jnp.float32
    if dtype == jnp.complex128:
        return jnp.float64
    # default fallback
    return jnp.float32


def _as_initializer(spec: InitLike, default_init: Callable, dtype):
    """
    Turn a user spec into an initialiser callable.

    - If spec is callable: return as-is
    - If spec is a float/int: normal(stddev=spec)
    - If spec is None: default_init

    Wrap to complex if dtype is complex
    """
    if callable(spec):
        init = spec
    elif spec is None:
        init = default_init
    else:
        # treat numbers as stddevs
        std = float(spec)
        init = nn.initializers.normal(stddev=std)

    # complex wrapping (build complex from two independent real draws)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        real_init = init

        def complex_init(key, shape, _dtype=dtype):
            rkey, ikey = jax.random.split(key)
            rdtype = _real_dtype_for_complex(_dtype)
            real = real_init(rkey, shape, rdtype)
            imag = real_init(ikey, shape, rdtype)
            # 1/sqrt(2) keeps variance of complex entry similar to real init
            return (real + 1j * imag) / jnp.sqrt(2.0)

        return complex_init

    return init


#
#
#   Networks


class RevNetAuto(nn.Module):
    """
    The RevNet network implementation in flax
    """

    __name__ = "RevNetOld"
    """The name of the network, required for logging purposes"""

    number_of_cnn_features: int
    """The number of features in the Conv layer"""

    rng_key: "jax.random.PRNGKey"
    """A PRNGKey for reproducibility of results, used to init layers"""

    lqx: LqxWCL = field(repr=False, metadata={"pytree_node": False})
    """An lqx object to obtain relevant graph and cutoff information from"""

    dropout_rate: float = 0.2
    """Dropout rate used when dropout layers are utilised"""

    free_idxs: jax.Array = field(default=None, repr=False)
    """The indices of the free (non-gauge fixed) edges"""

    dtype: jax.numpy.dtype = jax.numpy.float64
    """The dtype of the network (e.g. complex or real weights and biases)"""

    def __hash__(self):
        return id(self)

    def setup(self):
        """
        This sets up necessary information used in several layers of the network
        """

        self.cutoff = self.lqx.hilbert.cutoff

        self.number_of_vertices = self.lqx.graph.n_vertices

        if isinstance(self.lqx.model, LqxWCL4D):
            if self.cutoff >= 2:
                self.number_of_cnn_blocks = 5 + self.cutoff
            else:
                self.number_of_cnn_blocks = 5
        elif isinstance(self.lqx.model, LqxWCL3D):
            if self.cutoff >= 2:
                self.number_of_cnn_blocks = 3 + self.cutoff
            else:
                self.number_of_cnn_blocks = 5
        else:
            if self.cutoff == 1:
                self.number_of_cnn_blocks = 1
            elif self.cutoff >= 4:
                self.number_of_cnn_blocks = 5
            else:
                self.number_of_cnn_blocks = 1

        self.cnn_block_features = [self.number_of_cnn_features] + [
            self.number_of_cnn_features // i
            for i in [2 * i for i in range(1, self.number_of_cnn_blocks)]
        ]

        self.layer_sizes = list(range(self.number_of_cnn_blocks, 0, -1))

        self.sizes = [
            (
                self.number_of_vertices * el
                if isinstance(self.number_of_vertices * el, int)
                else math.ceil(self.number_of_vertices * el) + 1
            )
            for el in self.layer_sizes
        ]

    @nn.compact
    def __call__(self, x):

        # store the net_input
        net_input = x
        reduction_ratio = 2

        # check if the free_idxs are specified
        if self.free_idxs is not None:
            # if so, then truncate the input to be only the free indices
            net_input = jax.numpy.take(net_input, self.free_idxs, axis=1)

        # the learning phase
        # this phase contains successive application of CNN blocks and attention blocks
        for _ in range(self.number_of_cnn_blocks):
            cnn_layer = self.cnn_block(net_input, self.cnn_block_features[_], 1)
            net_input = self.attention_block(cnn_layer, reduction_ratio)

        # the evaluation phase
        return self.eval_block(net_input)

    def eval_block(
        self,
        net_input,
    ):
        """
        The evaluation block, essentially a simple FFN
        """

        output = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(net_input)
        output = reim(nn.hard_silu)(output)
        output = jax.numpy.sum(output, axis=-1)
        return output

    def cnn_block(
        self,
        net_input,
        features,
        kernel_size,
    ):
        """
        The CNN sub-block, applies a classic CNN phase to the given input by using a Conv layer,
        a normalisation, an activation and (possibly...) pooling
        """

        # we do not use dropout for now, uncomment the following to do so
        # rng, subkey = jax.random.split(self.rng_key)
        # dropout_layer = nn.Dropout(rate=self.dropout_rate)
        # x = net_input

        # apply the convolution
        net_input = nn.Conv(
            features=features,
            kernel_size=(kernel_size,),
            padding="CIRCULAR",
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(net_input)

        # then batch normalisation
        net_input = nn.BatchNorm(
            use_running_average=True,
            epsilon=1e-5,
            momentum=0.85,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(net_input)

        # activation
        net_input = reim(nn.hard_silu)(net_input)

        # uncomment this for pooling, this is incompatible with complex128 param_dtype
        # net_input = nn.max_pool(input, window_shape = (1,))

        return net_input

    def attention_block(
        self,
        net_input,
        reduction_ratio: int,
    ):
        """
        An attention sub-block. This essentially is a form of self-attention in the form of a
        squeeze and excite (SE) block. However, before the SE block we essentially aggregate the
        entire output of the CNN phase into one input
        """

        # pool into one input
        avg_pooled = jax.numpy.mean(net_input, axis=1, keepdims=True)

        # squeeze
        fc1 = nn.Dense(
            features=net_input.shape[-1] // reduction_ratio,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(avg_pooled)

        # activate
        fc1 = reim(nn.relu)(fc1)

        # excite
        fc2 = nn.Dense(
            features=net_input.shape[-1],
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(fc1)

        # apply the attention weights
        attention_weights = reim(nn.sigmoid)(fc2)
        attended_input = net_input * attention_weights

        return attended_input


class RevNet(nn.Module):
    """
    Below is the same RevNet architecture with fully tunable normalisation and initialisers.

    What differs here from the above implementaion is:
    - `bn_use_running_average` is configurable (True/False) (CURRENTLY NO BN IMPLEMENTED)
    - per-layer initialiser knobs, each accepting:
        * a callable initialiser (Flax-style),
        * OR a float/stddev (wraps to nn.initializers.normal(stddev)),
        * OR None to use a sensible default.
    - complex-aware initialisation: if `dtype` is complex, weights are built from
      two real draws to keep magnitude/phase variance at init
    - activations are now log_cosh, instead of hard_silu

    RNG usage:
        - we do NOT pass rng inside the module, use Flax collections:
            params = model.init({'params': key}, x)
        The old `rng_key` field is deprecated and ignored

    Defaults chosen to introduce gentle variance without saturation:
        - Conv/Dense kernels: normal(stddev=0.05)
        - Biases: zeros
        - BatchNorm scale: ones; bias: zeros



    Initializer knobs (per layer)
    -----
    For each spec below, you can pass:
      * callable initializer (Flax style),
      * or a float/int (interpreted as stddev for normal init),
      * or None for default described here.

    conv_kernel_init : InitLike, default normal(0.05)
    conv_bias_init   : InitLike, default zeros

    bn_scale_init    : InitLike, default ones
    bn_bias_init     : InitLike, default zeros

    se_fc1_kernel_init : InitLike, default normal(0.05)
    se_fc1_bias_init   : InitLike, default zeros
    se_fc2_kernel_init : InitLike, default normal(0.05)
    se_fc2_bias_init   : InitLike, default zeros

    eval_dense_kernel_init : InitLike, default normal(0.05)
    eval_dense_bias_init   : InitLike, default zeros

    Notes
    -----
    - RNG: initialize with model.init({'params': key}, x). Different keys give
      different parameter samples. The deprecated `rng_key` field is ignored.
    - If you previously depended on BatchNorm with `use_running_average=True`
      during training, that can suppress variance. Toggle this via the knob.
    """

    __name__ = "RevNet"

    number_of_cnn_features: int
    """Base number of features for the first Conv layer of each block."""

    number_of_cnn_blocks: int
    """Number of CNN+SE attention blocks to apply."""

    lqx: LqxWCL = field(repr=False, metadata={"pytree_node": False})
    """An LQX object providing graph/cutoff information."""

    #
    #
    #   behavior knobs

    dropout_rate: float = 0.2
    """Kept for future use; not active in this implementation."""

    free_idxs: Optional[jax.Array] = field(default=None, repr=False)
    """Optional indices of "free" edges. If provided, the input is sliced to these columns."""

    dtype: jnp.dtype = jnp.float64
    """Compute and parameter dtype (supports real or complex)."""

    bn_use_running_average: bool = False
    """If True: BatchNorm uses running averages (inference-style)."""

    use_visible_bias: bool = True
    """If True adds a bias to the input not passed through the nonlinear layer."""

    #
    #
    #   Initialiser knobs (see the docstring)

    conv_kernel_init: InitLike = None
    conv_bias_init: InitLike = 0.1

    bn_scale_init: InitLike = None
    bn_bias_init: InitLike = 0.1

    se_fc1_kernel_init: InitLike = None
    se_fc1_bias_init: InitLike = 0.1
    se_fc2_kernel_init: InitLike = None
    se_fc2_bias_init: InitLike = 0.1

    eval_dense_kernel_init: InitLike = None
    eval_dense_bias_init: InitLike = 0.1

    visible_bias_init: InitLike = 0.8
    """Initialiser for the visible bias."""

    # deprecated: kept for API compatibility but not used
    rng_key: Optional[jax.random.PRNGKey] = field(default=None, repr=False)

    def __hash__(self):
        return id(self)

    def setup(self):

        # store the cutoff for the physical system
        self.cutoff = self.lqx.H.core.cutoff

        # store the number of vertices in the graph
        self.number_of_vertices = self.lqx.graph.dual_nk_graph.n_nodes

        # create the list of cnn features based on the base number of cnn features provided
        self.cnn_block_features = [self.number_of_cnn_features] + [
            self.number_of_cnn_features // i
            for i in [2 * i for i in range(1, self.number_of_cnn_blocks)]
        ]

        # create the layer sizes
        self.layer_sizes = list(range(self.number_of_cnn_blocks, 0, -1))

        # compute the size for each layer
        self.sizes = [
            (
                self.number_of_vertices * el
                if isinstance(self.number_of_vertices * el, int)
                else math.ceil(self.number_of_vertices * el) + 1
            )
            for el in self.layer_sizes
        ]

        # resolve default initialisers now (and wrap for complex dtype)
        self._conv_kernel_init = _as_initializer(
            self.conv_kernel_init,
            nn.initializers.normal(stddev=0.1),
            self.dtype,
        )
        self._conv_bias_init = _as_initializer(
            self.conv_bias_init,
            nn.initializers.normal(stddev=0.1),
            self.dtype,
        )

        self._bn_scale_init = _as_initializer(
            self.bn_scale_init,
            nn.initializers.normal(stddev=0.1),
            self.dtype,
        )
        self._bn_bias_init = _as_initializer(
            self.bn_bias_init,
            nn.initializers.zeros,
            self.dtype,
        )

        self._se_fc1_kernel_init = _as_initializer(
            self.se_fc1_kernel_init,
            nn.initializers.normal(stddev=0.1),
            self.dtype,
        )
        self._se_fc1_bias_init = _as_initializer(
            self.se_fc1_bias_init,
            nn.initializers.normal(stddev=0.1),
            self.dtype,
        )
        self._se_fc2_kernel_init = _as_initializer(
            self.se_fc2_kernel_init,
            nn.initializers.normal(stddev=0.1),
            self.dtype,
        )
        self._se_fc2_bias_init = _as_initializer(
            self.se_fc2_bias_init,
            nn.initializers.normal(stddev=0.1),
            self.dtype,
        )

        self._eval_dense_kernel_init = _as_initializer(
            self.eval_dense_kernel_init,
            nn.initializers.normal(stddev=0.1),
            self.dtype,
        )
        self._eval_dense_bias_init = _as_initializer(
            self.eval_dense_bias_init,
            nn.initializers.normal(stddev=0.1),
            self.dtype,
        )

        self._visible_bias_init = _as_initializer(
            self.visible_bias_init,
            nn.initializers.normal(stddev=0.8),
            self.dtype,
        )

    @nn.compact
    def __call__(self, x):
        net_input = x
        reduction_ratio = 2

        if self.free_idxs is not None:
            net_input = jax.numpy.take(net_input, self.free_idxs, axis=1)

        for b in range(self.number_of_cnn_blocks):
            cnn_layer = self.cnn_block(net_input, self.cnn_block_features[b], 1)
            net_input = self.attention_block(cnn_layer, reduction_ratio)

        evals = self.eval_block(net_input)

        # optional visible bias term: <input, v_bias>
        if self.use_visible_bias:
            v_bias = self.param(
                "visible_bias",
                self._visible_bias_init,
                (net_input.shape[-1],),
                self.dtype,
            )

            evals = evals + jnp.sum(jnp.conj(net_input) * v_bias, axis=-1)

        return evals

    def eval_block(self, net_input):
        """
        Evaluation block: Dense(1) -> log_cosh -> sum over features
        """

        # dense layer
        output = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=self.dtype,
            kernel_init=self._eval_dense_kernel_init,
            bias_init=self._eval_dense_bias_init,
        )(net_input)

        # activation
        output = netket.nn.log_cosh(output)

        # aggregate
        output = jax.numpy.sum(output, axis=-1)

        return output

    def cnn_block(self, net_input, features, kernel_size):
        """
        CNN sub-block:
          Conv1D (CIRCULAR) -> BatchNorm (NOT IMPLEMENTED) -> log_cosh
        """

        # convolution
        net_input = nn.Conv(
            features=features,
            kernel_size=(kernel_size,),
            padding="CIRCULAR",
            dtype=self.dtype,
            param_dtype=self.dtype,
            kernel_init=self._conv_kernel_init,
            bias_init=self._conv_bias_init,
        )(net_input)

        # uncomment this chunk of code if you want to use batch normalisation
        net_input = nn.BatchNorm(
            use_running_average=self.bn_use_running_average,
            epsilon=1e-5,
            momentum=0.85,
            dtype=self.dtype,
            param_dtype=self.dtype,
            scale_init=self._bn_scale_init,
            bias_init=self._bn_bias_init,
        )(net_input)

        # activation
        net_input = netket.nn.log_cosh(net_input)

        return net_input

    def attention_block(self, net_input, reduction_ratio: int):
        """
        Squeeze-and-Excite attention:
          avg-pool -> Dense(reduced) -> ReLU -> Dense(channels) -> sigmoid -> scale
        """

        # pool into one input
        avg_pooled = jax.numpy.mean(net_input, axis=1, keepdims=True)

        # squeeze
        fc1 = nn.Dense(
            features=max(1, net_input.shape[-1] // reduction_ratio),
            dtype=self.dtype,
            param_dtype=self.dtype,
            kernel_init=self._se_fc1_kernel_init,
            bias_init=self._se_fc1_bias_init,
        )(avg_pooled)

        # activation
        fc1 = reim(nn.relu)(fc1)

        # excite
        fc2 = nn.Dense(
            features=net_input.shape[-1],
            dtype=self.dtype,
            param_dtype=self.dtype,
            kernel_init=self._se_fc2_kernel_init,
            bias_init=self._se_fc2_bias_init,
        )(fc1)

        # activation
        attention_weights = reim(nn.sigmoid)(fc2)

        # aggregate
        attended_input = net_input * attention_weights

        return attended_input
