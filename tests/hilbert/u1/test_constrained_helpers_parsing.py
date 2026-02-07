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


import pytest


def test_parse_sign_conventions_N_and_explicit_minus(graph_small):

    from neuralqx.hilbert.u1.constrained_core import _parse_gauge_fixing_to_constraints

    gf = [[[1], [0, "N", 2, "-3"]]]

    constraints = _parse_gauge_fixing_to_constraints(graph_small, gf)
    assert isinstance(constraints, tuple)
    assert len(constraints) == 1

    c = constraints[0]
    assert c.lhs == 1
    assert c.rhs == ((0, +1), (2, -1), (3, +1))


def test_parse_rejects_duplicate_lhs(graph_small):
    from neuralqx.hilbert.u1.constrained_core import _parse_gauge_fixing_to_constraints
    from neuralqx.utils.errors import IncorrectGaugeFixingArrayError

    gf = [
        [[1], [0, 2]],
        [[1], [3, 4]],
    ]

    with pytest.raises(IncorrectGaugeFixingArrayError, match=r"Duplicate LHS edges"):
        _parse_gauge_fixing_to_constraints(graph_small, gf)


def test_parse_rejects_empty_rhs(graph_small):
    from neuralqx.hilbert.u1.constrained_core import _parse_gauge_fixing_to_constraints
    from neuralqx.utils.errors import IncorrectGaugeFixingArrayError

    gf = [[[1], []]]

    with pytest.raises(IncorrectGaugeFixingArrayError, match=r"empty RHS"):
        _parse_gauge_fixing_to_constraints(graph_small, gf)


def test_topological_ordering_detects_cycles():
    from neuralqx.hilbert.u1.constrained_core import _Constraint, _topological_ordering
    from neuralqx.utils.errors import CyclicGaugeFixingError

    cons = (
        _Constraint(lhs=0, rhs=((1, +1),)),
        _Constraint(lhs=1, rhs=((0, +1),)),
    )

    with pytest.raises(CyclicGaugeFixingError, match=r"cycle"):
        _topological_ordering(cons)


def test_topological_ordering_returns_expected_free_and_slaves():
    from neuralqx.hilbert.u1.constrained_core import _Constraint, _topological_ordering

    cons = (
        _Constraint(lhs=2, rhs=((0, +1), (1, +1))),
        _Constraint(lhs=3, rhs=((2, +1),)),
    )

    topo = _topological_ordering(cons)

    assert topo.free == (0, 1)
    assert topo.slave_topo == (2, 3)
    assert topo.slave_fixers == (
        ((0, +1), (1, +1)),
        ((2, +1),),
    )
    assert topo.n_free == 2
