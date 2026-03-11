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
import jax.numpy as jnp


@pytest.mark.integration
def test_exact_diagonalization_computes_ground_energy_and_state(
    toy_objects_gauge_invariant,
):
    H, GG, model_constraint, iface = toy_objects_gauge_invariant

    _ = iface.constraint

    iface.exact_diagonalization(recompute=True)

    assert iface.has_ground_energy
    assert iface.has_ground_state

    ge = iface.ground_energy
    ge0 = ge[0] if hasattr(ge, "__len__") else ge
    assert jnp.allclose(jnp.asarray(ge0), jnp.asarray(-2.0), atol=1e-6, rtol=1e-6)

    assert iface.ground_state is not None


def test_user_supplied_ground_energy_disables_ground_state(toy_objects_gauge_invariant):
    *_rest, iface = toy_objects_gauge_invariant

    iface.exact_diagonalization(ground_energy=-123.0, recompute=True)

    assert iface.has_ground_energy
    assert not iface.has_ground_state
    assert iface.ground_energy == -123.0

    with pytest.raises(ValueError):
        _ = iface.ground_state


def test_non_indexable_hilbert_skips_diagonalization(nk_spin_2, nk):
    import types
    from tests.lqx.conftest import DummyHilbertInterface, DummyGaugeGroup, ToyInterface

    g, hi = nk_spin_2

    dummy_core = types.SimpleNamespace(
        hilbert=types.SimpleNamespace(is_indexable=False)
    )
    H = DummyHilbertInterface(
        graph=g, nk_hilbert=hi, is_gauge_invariant=True, gauge_dimensions=1
    )
    H.hilbert = dummy_core

    z0 = nk.operator.spin.sigmaz(hi, 0)
    z1 = nk.operator.spin.sigmaz(hi, 1)
    model_constraint = z0 + z1

    GG = DummyGaugeGroup(constraint=z0 * 0.0, dimensions=1)
    iface = ToyInterface(H, GG, model_constraint=model_constraint, lazy_load=True)
    iface.printer.messages.clear()

    ok = iface._maybe_diagonalised()
    assert ok is False

    assert not iface.has_ground_energy
    assert not iface.has_ground_state

    assert any("too large to be indexed" in msg for msg in iface.printer.messages)
