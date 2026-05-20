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

import flax.linen as nn
import numpy as np
import pytest
from flax.core import unfreeze

from neuralqx.vqs.mc.mc_state.state import deserialize_MCState, serialize_MCState

pytestmark = [pytest.mark.regression, pytest.mark.integration]


class IdentityWrapper(nn.Module):
    base: nn.Module

    @nn.compact
    def __call__(self, x):
        return self.base(x)


def _wrap_identity(model, **_):
    return IdentityWrapper(base=model)


def _unwrap_projected_params_tree(tree):
    if hasattr(tree, "keys") and "base" in tree:
        return tree["base"]
    return tree


def test_mcstate_serialise_deserialise_preserves_wavefunction(mcstate, jnp):
    state_dict = serialize_MCState(mcstate)
    restored = deserialize_MCState(mcstate, state_dict)

    psi_before = mcstate.to_array(normalize=True)
    psi_after = restored.to_array(normalize=True)

    assert restored is not mcstate
    assert restored.n_samples == mcstate.n_samples
    assert restored.n_discard_per_chain == mcstate.n_discard_per_chain
    assert restored.chunk_size == mcstate.chunk_size
    assert jnp.allclose(psi_before, psi_after, rtol=1e-7, atol=1e-8)


def test_mcstate_project_identity_preserves_expectation_and_grad(
    mcstate, ops_spin_2, helpers, jnp
):
    mean, tree_allclose = helpers
    sx0, sz0, _ = ops_spin_2
    op = sx0 + sz0

    projected = mcstate.project(_wrap_identity, reuse_cached_samples=True)

    stats_ref, grad_ref = mcstate.expect_and_grad(op, mutable=False)
    stats_proj, grad_proj = projected.expect_and_grad(op, mutable=False)

    assert projected is not mcstate
    assert projected.samples is mcstate.samples
    assert jnp.allclose(
        mean(stats_ref),
        mean(stats_proj),
        rtol=1e-6,
        atol=1e-7,
    )
    tree_allclose(
        unfreeze(grad_ref),
        unfreeze(_unwrap_projected_params_tree(grad_proj)),
        rtol=1e-6,
        atol=1e-7,
    )


def test_mcstate_project_without_cached_samples_recomputes_samples(
    mcstate, ops_spin_2, jnp
):
    sx0, _, _ = ops_spin_2
    projected = mcstate.project(_wrap_identity, reuse_cached_samples=False)

    assert projected._samples is None

    projected.sample(n_samples=64, n_discard_per_chain=0)
    assert projected._samples is not None
    stats = projected.expect(sx0)
    mean = np.asarray(stats.Mean)
    assert np.isfinite(np.real(mean))
    assert np.isfinite(np.imag(mean))
    assert jnp.asarray(projected.samples).shape[0] > 0
