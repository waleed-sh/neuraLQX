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


import jax.numpy as jnp


def _conn_signature(op, sigma):
    xp, mels = op.get_conn_padded(sigma)
    return jnp.asarray(xp), jnp.asarray(mels)


def test_constraint_lazy_init_on_access_gauge_invariant(toy_objects_gauge_invariant):
    H, GG, model_constraint, iface = toy_objects_gauge_invariant

    assert iface._constraint is None

    c = iface.constraint
    assert c is not None

    sigma = jnp.asarray([1, 1])
    xp1, m1 = _conn_signature(c, sigma)
    xp2, m2 = _conn_signature(model_constraint, sigma)

    assert jnp.array_equal(xp1, xp2)
    assert jnp.allclose(m1, m2)


def test_constraint_lazy_init_on_access_not_gauge_invariant(
    toy_objects_not_gauge_invariant,
):
    H, GG, model_constraint, iface = toy_objects_not_gauge_invariant

    assert iface._constraint is None
    c = iface.constraint
    assert c is not None

    expected = model_constraint + GG.constraint

    sigma = jnp.asarray([1, -1])
    xp1, m1 = _conn_signature(c, sigma)
    xp2, m2 = _conn_signature(expected, sigma)

    assert jnp.array_equal(xp1, xp2)
    assert jnp.allclose(m1, m2)


def test_constraint_eager_init_when_lazy_load_false(toy_objects_gauge_invariant):
    H, GG, model_constraint, _iface = toy_objects_gauge_invariant

    from tests.lqx.conftest import ToyInterface

    iface2 = ToyInterface(H, GG, model_constraint=model_constraint, lazy_load=False)

    assert iface2._constraint is not None


def test_constraint_setter_overrides(toy_objects_gauge_invariant):
    H, GG, model_constraint, iface = toy_objects_gauge_invariant

    _ = iface.constraint
    old = iface.constraint

    new = model_constraint * 0.0
    iface.constraint = new
    assert iface.constraint is new
    assert iface.constraint is not old
