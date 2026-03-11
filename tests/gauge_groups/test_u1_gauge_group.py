#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import numpy as np
import pytest


def _import_u1_group():
    try:
        from neuralqx.gauge_groups.u1 import U1GaugeGroup

        return U1GaugeGroup
    except Exception:
        from neuralqx.gauge_groups.u1.u1_gauge_group import U1GaugeGroup

        return U1GaugeGroup


def _import_errors():
    try:
        from neuralqx.utils.errors import (
            IncompatibleHilbertSpaceError,
            IncompatibleModdedOperatorWarning,
        )

        return IncompatibleHilbertSpaceError, IncompatibleModdedOperatorWarning
    except Exception as e:
        raise RuntimeError(f"Could not import error types: {e}")


def _to_2d(states):
    states = np.asarray(states)
    if states.ndim == 1:
        states = states[None, :]
    return states


def _num_dual_sites(G) -> int:
    return len(list(G.dual_nx_graph.nodes))


def _manual_gauss_value(H, states, gauge_dimensions: int):
    G = H.graph
    states = _to_2d(states)
    n_dual = _num_dual_sites(G)

    vals = np.zeros((states.shape[0],), dtype=np.int64)

    connectivity = G.handler.list_of_node_connectivity

    for b in range(states.shape[0]):
        total = 0
        x = states[b]
        for copy_idx in range(int(gauge_dimensions)):
            offset = copy_idx * n_dual
            for _node, attrs in connectivity.items():
                div = 0
                for e in attrs["incoming"]:
                    div += int(x[offset + G.edge_to_index(e)])
                for e in attrs["outgoing"]:
                    div -= int(x[offset + G.edge_to_index(e)])
                total += div * div
        vals[b] = total

    return vals


def _diag_element_from_get_conn(op, state):
    if not hasattr(op, "get_conn"):
        return None

    xp, mels = op.get_conn(state)
    xp = np.asarray(xp)
    mels = np.asarray(mels)

    if xp.ndim == 1:
        xp = xp[None, :]

    # NetKet may drop explicit zero-valued diagonal entries from get_conn.
    # For a diagonal constraint this means "no connections" can represent a
    # zero diagonal matrix element.
    if len(mels) == 0:
        return 0.0

    if len(mels) != 1:
        raise AssertionError(
            f"Expected diagonal operator with 1 connected state, got {len(mels)}"
        )
    if not np.array_equal(np.asarray(xp[0]), np.asarray(state)):
        raise AssertionError("Expected only self-connection for diagonal constraint.")
    return float(mels[0])


def _diag_elements(op, states):
    out = []
    for x in _to_2d(states):
        v = _diag_element_from_get_conn(op, x)
        if v is None:
            return None
        out.append(v)
    return np.asarray(out, dtype=np.float64)


def _u1_modulus_from_hilbert(H) -> int:

    dofs = getattr(H, "dofs", None)
    if dofs is not None and hasattr(dofs, "length"):
        return int(dofs.length)

    cutoff = getattr(H, "cutoff", None)
    if cutoff is not None:
        return 2 * int(cutoff) + 1
    try:
        vals = np.asarray(list(dofs), dtype=int)
        return int(np.unique(vals).size)
    except Exception:
        raise RuntimeError("Could not infer U(1) modulus from Hilbert.")


def _manual_gauge_invariant_flags(
    H, states, gauge_dimensions: int, *, modded: bool
) -> np.ndarray:
    G = H.graph
    states = _to_2d(states)
    n_dual = _num_dual_sites(G)
    conn = G.handler.list_of_node_connectivity

    modulus = _u1_modulus_from_hilbert(H) if modded else None

    out = np.ones((states.shape[0],), dtype=bool)
    for b in range(states.shape[0]):
        x = states[b]
        ok = True
        for copy_idx in range(int(gauge_dimensions)):
            offset = copy_idx * n_dual
            for _node, attrs in conn.items():
                div = 0
                for e in attrs["incoming"]:
                    div += int(x[offset + G.edge_to_index(e)])
                for e in attrs["outgoing"]:
                    div -= int(x[offset + G.edge_to_index(e)])

                if modded:
                    if (div % modulus) != 0:
                        ok = False
                        break
                else:
                    if div != 0:
                        ok = False
                        break

            if not ok:
                break
        out[b] = ok

    return out


def test_u1_group_rejects_incompatible_hilbert(neuralqx):
    U1GaugeGroup = _import_u1_group()
    IncompatibleHilbertSpaceError, _ = _import_errors()

    class NotHilbertU1:
        __name__ = "NotHilbertU1"
        pass

    with pytest.raises(IncompatibleHilbertSpaceError):
        U1GaugeGroup(NotHilbertU1())


def test_u1_group_basic_properties(hilbert_u1_full_1copy):
    U1GaugeGroup = _import_u1_group()

    g = U1GaugeGroup(hilbert_u1_full_1copy, computational=True, jax=True, lazy=True)
    assert g.is_abelian is True
    assert g.dimensions == hilbert_u1_full_1copy.gauge_dimensions
    assert "U(1)" in g.name
    assert str(g.dimensions) in g.name

    assert g.graph is hilbert_u1_full_1copy.graph
    assert g.hilbert is hilbert_u1_full_1copy


@pytest.mark.parametrize(
    "computational,jax_flag",
    [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ],
)
def test_u1_group_constraint_type_matches_flags(
    hilbert_u1_full_1copy, netket, computational, jax_flag
):
    from netket.operator import LocalOperator

    U1GaugeGroup = _import_u1_group()

    g = U1GaugeGroup(
        hilbert_u1_full_1copy,
        computational=computational,
        jax=jax_flag,
        lazy=True,
    )

    c = g.constraint
    if computational:
        assert not isinstance(c, LocalOperator)
        assert g.is_computational is True
        assert g.is_jax is bool(jax_flag)
    else:
        assert isinstance(c, LocalOperator)
        assert g.is_computational is False
        assert g.is_jax is bool(jax_flag)


def test_u1_group_constraint_setter_caches_and_reinits(hilbert_u1_full_1copy, netket):
    from netket.operator import LocalOperator

    U1GaugeGroup = _import_u1_group()

    g = U1GaugeGroup(hilbert_u1_full_1copy, computational=False, jax=False, lazy=True)
    assert isinstance(g.constraint, LocalOperator)

    c0 = g.constraint
    g.constraint = {"lazy": True, "computational": False, "jax": False, "reinit": False}
    assert g.constraint is c0

    g.constraint = {"lazy": True, "computational": False, "jax": False, "reinit": True}
    assert g.constraint is not c0


def test_u1_group_modded_warning_only_for_localoperator(hilbert_u1_full_1copy):
    U1GaugeGroup = _import_u1_group()
    _, IncompatibleModdedOperatorWarning = _import_errors()

    g = U1GaugeGroup(
        hilbert_u1_full_1copy,
        computational=False,
        jax=False,
        lazy=True,
    )

    with pytest.warns(IncompatibleModdedOperatorWarning):
        _ = g.init_constraint(
            computational=False,
            jax=False,
            lazy=True,
            modded=True,
        )


def test_u1_localoperator_constraint_is_diagonal_and_nonnegative(
    hilbert_u1_full_1copy, jax
):
    U1GaugeGroup = _import_u1_group()

    g = U1GaugeGroup(hilbert_u1_full_1copy, computational=False, jax=False, lazy=True)
    op = g.constraint

    states = hilbert_u1_full_1copy.random_state(jax.random.PRNGKey(0), size=10)
    states = _to_2d(states)

    diag = _diag_elements(op, states)
    assert diag is not None, "Expected LocalOperator to implement get_conn."
    assert np.all(diag >= -1e-12)


def test_u1_localoperator_constraint_matches_manual_gauss_and_is_gauge_invariant(
    hilbert_u1_full_1copy, jax
):
    U1GaugeGroup = _import_u1_group()

    g = U1GaugeGroup(hilbert_u1_full_1copy, computational=False, jax=False, lazy=True)
    op = g.constraint

    states = hilbert_u1_full_1copy.random_state(jax.random.PRNGKey(1), size=50)
    states = _to_2d(states)

    gi = np.asarray(
        hilbert_u1_full_1copy.hilbert.is_gauge_invariant(states), dtype=bool
    )
    manual = _manual_gauss_value(hilbert_u1_full_1copy, states, gauge_dimensions=1)

    diag = _diag_elements(op, states)
    assert diag is not None, "Expected LocalOperator to implement get_conn."

    assert np.allclose(diag, manual.astype(np.float64), rtol=0, atol=1e-12)

    strict_gi = _manual_gauge_invariant_flags(
        hilbert_u1_full_1copy, states, gauge_dimensions=1, modded=False
    )
    modded_gi = _manual_gauge_invariant_flags(
        hilbert_u1_full_1copy, states, gauge_dimensions=1, modded=True
    )

    if np.any(gi & (manual != 0)):
        expected = modded_gi
    else:
        expected = strict_gi

    assert np.array_equal(gi, expected), (
        "H.hilbert.is_gauge_invariant disagrees with both strict and modded Gauss-law checks.\n"
        f"gi={gi}\n"
        f"strict_gi={strict_gi}\n"
        f"modded_gi={modded_gi}\n"
        f"manual(int-div^2)={manual}"
    )

    assert np.all(
        gi[manual == 0]
    ), "States with zero integer Gauss constraint should always be GI."


def test_u1_localoperator_two_copy_constraint_matches_sum_of_copies(
    hilbert_u1_full_2copy, jax
):

    U1GaugeGroup = _import_u1_group()

    g = U1GaugeGroup(hilbert_u1_full_2copy, computational=False, jax=False, lazy=True)

    base = g._one_copy_constraint()
    shifted = g._shift_copy(base, 1)
    total = g.constraint

    states = hilbert_u1_full_2copy.random_state(jax.random.PRNGKey(2), size=25)
    states = _to_2d(states)

    d_base = _diag_elements(base, states)
    d_shift = _diag_elements(shifted, states)
    d_total = _diag_elements(total, states)

    assert d_base is not None and d_shift is not None and d_total is not None
    assert np.allclose(d_total, d_base + d_shift, rtol=0, atol=1e-12)

    manual = _manual_gauss_value(hilbert_u1_full_2copy, states, gauge_dimensions=2)
    assert np.allclose(d_total, manual.astype(np.float64), rtol=0, atol=1e-12)


def test_u1_shift_copy_offsets_acting_on(hilbert_u1_full_2copy):
    U1GaugeGroup = _import_u1_group()
    g = U1GaugeGroup(hilbert_u1_full_2copy, computational=False, jax=False, lazy=True)

    base = g._one_copy_constraint()
    shifted = g._shift_copy(base, 1)

    n_dual = _num_dual_sites(g.graph)

    base_sites = sorted({s for tpl in base.acting_on for s in tpl})
    shifted_sites = sorted({s for tpl in shifted.acting_on for s in tpl})

    assert len(base_sites) > 0
    assert len(base_sites) == len(shifted_sites)
    assert shifted_sites == [s + n_dual for s in base_sites]


def test_u1_computational_and_localoperator_agree_on_diagonal_values(
    hilbert_u1_full_1copy, jax
):

    U1GaugeGroup = _import_u1_group()

    g_local = U1GaugeGroup(
        hilbert_u1_full_1copy, computational=False, jax=False, lazy=True
    )
    g_comp = U1GaugeGroup(
        hilbert_u1_full_1copy, computational=True, jax=False, lazy=True
    )

    states = hilbert_u1_full_1copy.random_state(jax.random.PRNGKey(3), size=30)
    states = _to_2d(states)

    d_local = _diag_elements(g_local.constraint, states)
    d_comp = _diag_elements(g_comp.constraint, states)

    if d_comp is None:
        pytest.skip(
            "Computational operator does not expose get_conn, cannot compare diagonals."
        )

    assert d_local is not None
    assert np.allclose(d_comp, d_local, rtol=0, atol=1e-12)


def test_u1_group_composition_mul_pow_imul(hilbert_u1_full_4copy):

    U1GaugeGroup = _import_u1_group()

    g = U1GaugeGroup(hilbert_u1_full_4copy, computational=False, jax=False, lazy=True)

    g1 = g._clone_with_dimensions(1)
    g3 = g._clone_with_dimensions(3)

    g4 = g1 * g3
    assert g4.dimensions == 4

    g2 = g._clone_with_dimensions(2)
    g4b = g2**2
    assert g4b.dimensions == 4

    with pytest.raises(ValueError):
        _ = g2**0
    with pytest.raises(ValueError):
        _ = g2 ** (-1)
    with pytest.raises(ValueError):
        _ = g2**1.5

    g_tmp = g._clone_with_dimensions(1)
    g_tmp *= g._clone_with_dimensions(3)
    assert g_tmp.dimensions == 4


def test_u1_mul_different_hilbert_raises(neuralqx, theta_graph):
    U1GaugeGroup = _import_u1_group()
    Hilb = neuralqx.hilbert.u1.HilbertU1

    H1 = Hilb(
        theta_graph,
        cutoff=1,
        is_gauge_invariant=False,
        auto_constraint=1,
        gauge_dimensions=2,
    )
    H2 = Hilb(
        theta_graph,
        cutoff=1,
        is_gauge_invariant=False,
        auto_constraint=1,
        gauge_dimensions=2,
    )

    g1 = U1GaugeGroup(H1, computational=False, jax=False, lazy=True)
    g2 = U1GaugeGroup(H2, computational=False, jax=False, lazy=True)

    with pytest.raises(ValueError):
        _ = g1 * g2


def test_u1_repr_contains_key_fields(hilbert_u1_full_1copy):
    U1GaugeGroup = _import_u1_group()
    g = U1GaugeGroup(hilbert_u1_full_1copy, computational=True, jax=True, lazy=True)

    s = repr(g)
    assert "U1GaugeGroup(" in s
    assert "dimensions=" in s
    assert "is_abelian=" in s
    assert "is_computational=" in s
