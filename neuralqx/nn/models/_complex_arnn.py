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

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jax import numpy as jnp
from jax.nn.initializers import zeros

from netket.nn import MaskedDense1D
from netket.nn import activation as nkactivation
from netket.nn.masked_linear import default_kernel_init
from netket.models.autoreg import ARNNSequential
from netket.models.autoreg import _get_feature_list
from netket.models.autoreg import _normalize
from netket.utils.types import Array
from netket.utils.types import DType
from netket.utils.types import NNInitFunc


class ComplexARNNDense(ARNNSequential):
    """
    Complex-valued dense ARNN following the NetKet ARNNSequential recommendation.

    The model uses real-valued parameters and computes two outputs from the final
    autoregressive hidden representation:
    - log-modulus terms (normalised with `machine_pow`)
    - phase terms

    These are combined as
        log psi = log|psi| + i * phase
    in `conditionals_log_psi`.

    To avoid useless phase work during autoregressive sampling, `conditional`
    computes only the modulus branch.
    """

    layers: int
    features: tuple[int, ...] | int
    activation: Callable[[Array], Array] = nkactivation.reim_selu
    use_bias: bool = True
    param_dtype: DType = jnp.float64
    precision: Any = None
    kernel_init: NNInitFunc = default_kernel_init
    bias_init: NNInitFunc = zeros
    machine_pow: int = 2

    def setup(self):
        features = list(_get_feature_list(self))

        # Keep the same interface as NetKet ARNNDense:
        # the final "local_size" output layer is replaced by two heads
        # (modulus and phase), both fed by the previous hidden representation.
        hidden_features = features[:-1]

        self._hidden_layers = [
            MaskedDense1D(
                features=hidden_features[i],
                exclusive=(i == 0),
                use_bias=self.use_bias,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(len(hidden_features))
        ]

        head_exclusive = len(hidden_features) == 0
        local_size = self.hilbert.local_size

        self._modulus_head = MaskedDense1D(
            features=local_size,
            exclusive=head_exclusive,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self._phase_head = MaskedDense1D(
            features=local_size,
            exclusive=head_exclusive,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    def _forward_hidden(self, inputs: Array) -> Array:
        x = self.reshape_inputs(inputs)
        x = jnp.expand_dims(x, axis=-1)

        for i, layer in enumerate(self._hidden_layers):
            if i > 0:
                x = self.activation(x)
            x = layer(x)

        if len(self._hidden_layers) > 0:
            x = self.activation(x)

        return x

    def _modulus_logits(self, inputs: Array) -> Array:
        x = self._forward_hidden(inputs)
        logits = self._modulus_head(x)
        logits = logits.reshape((logits.shape[0], -1, logits.shape[-1]))
        return logits

    def conditionals_log_psi(self, inputs: Array) -> Array:
        x = self._forward_hidden(inputs)

        log_modulus = self._modulus_head(x)
        phase = self._phase_head(x)

        log_modulus = log_modulus.reshape(
            (log_modulus.shape[0], -1, log_modulus.shape[-1])
        )
        phase = phase.reshape((phase.shape[0], -1, phase.shape[-1]))

        log_modulus = _normalize(log_modulus, self.machine_pow)
        return log_modulus + 1j * phase

    def conditional(self, inputs: Array, index: int) -> Array:
        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)

        log_modulus = self._modulus_logits(inputs)
        log_modulus = _normalize(log_modulus, self.machine_pow)
        return jnp.exp(self.machine_pow * log_modulus[:, index, :].real)
