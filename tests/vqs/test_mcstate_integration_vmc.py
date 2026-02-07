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


import pytest
import functools
import flax.linen as nn


@functools.lru_cache(maxsize=None)
def _jax():
    return pytest.importorskip("jax")


@functools.lru_cache(maxsize=None)
def _jnp():
    jax = _jax()
    import jax.numpy as jnp

    return jnp


class TinyModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.astype(_jnp().float32)
        w = self.param("w", nn.initializers.normal(stddev=0.1), (x.shape[-1],))
        return (x @ w).astype(_jnp().complex64)


@pytest.mark.integration
def test_nqx_mcstate_runs_inside_netket_vmc(tmp_path, nk, nqx, jnp):

    hi = nk.hilbert.Spin(s=0.5, N=2)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=2)
    model = TinyModel()

    st = nqx.vqs.MCState(
        sampler,
        model,
        n_samples=32,
        n_discard_per_chain=0,
        seed=1,
        sampler_seed=2,
    )

    op_a = nk.operator.spin.sigmax(hi, 0)
    op_b = nk.operator.spin.sigmaz(hi, 1)
    H = op_a + op_b

    opt = nk.optimizer.Sgd(learning_rate=0.01)

    VMC = getattr(nk.driver, "VMC", None) or getattr(nk, "VMC", None)
    assert VMC is not None, "Could not locate NetKet VMC driver API."

    driver = VMC(hamiltonian=H, optimizer=opt, variational_state=st)
    driver.run(n_iter=1, out=str(tmp_path))

    stats = st.expect([op_a, op_b])
    assert jnp.isfinite(jnp.asarray(stats.Mean))
