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

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nk = pytest.importorskip("netket")

from neuralqx.nn.models import ComplexARNNDense


def test_complex_arnn_dense_outputs_complex_logpsi_and_valid_conditionals():
    hi = nk.hilbert.Spin(s=0.5, N=6)

    model = ComplexARNNDense(
        hi,
        layers=2,
        features=16,
        use_bias=True,
        param_dtype=jnp.float64,
        machine_pow=2,
    )

    sigma = jnp.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, +1, -1, +1, -1, +1],
        ],
        dtype=jnp.float64,
    )

    variables = model.init(jax.random.PRNGKey(0), sigma)

    logpsi = model.apply(variables, sigma)
    assert jnp.iscomplexobj(logpsi)
    assert logpsi.shape == (2,)

    cond_logpsi = model.apply(variables, sigma, method=model.conditionals_log_psi)
    assert jnp.iscomplexobj(cond_logpsi)
    assert cond_logpsi.shape == (2, hi.size, hi.local_size)

    cond_probs = model.apply(variables, sigma, 3, method=model.conditional)
    assert cond_probs.shape == (2, hi.local_size)
    assert not jnp.iscomplexobj(cond_probs)
    assert jnp.allclose(cond_probs.sum(axis=-1), 1.0, atol=1e-10)
