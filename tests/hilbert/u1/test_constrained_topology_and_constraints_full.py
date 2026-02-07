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


def test_generated_gauge_fixing_partitions_dependency_graph(constrained_core_small):

    core = constrained_core_small
    gf = core.gauge_fixing

    free = set(int(x) for x in gf.free)
    slaves = set(int(x) for x in gf.slave_topo)

    assert free.isdisjoint(slaves)
    assert gf.n_free == len(free)

    nodes = set()
    for c in core.constraints_base:
        nodes.add(int(c.lhs))
        for ridx, _sgn in c.rhs:
            nodes.add(int(ridx))

    assert nodes == (free | slaves)


def test_constraints_full_is_base_constraints_replicated_per_gauge_copy(
    constrained_core_small,
):

    core = constrained_core_small
    base = core.constraints_base
    full = getattr(core, "_constraints_full")

    G = int(core.gauge_dimensions)
    E = int(core.tiny_size)
    n_base = len(base)

    assert len(full) == G * n_base

    for gd in range(G):
        off = gd * E
        for i, c in enumerate(base):
            inc, out = full[gd * n_base + i]
            assert inc == ((int(c.lhs) + off, +1),)
            assert out == tuple((int(ridx) + off, int(sgn)) for (ridx, sgn) in c.rhs)


def test_netket_constraint_object_is_attached(constrained_core_small):

    from neuralqx.hilbert.constraints.u1_constraint import U1Constraint

    core = constrained_core_small
    h = core.hilbert

    assert hasattr(h, "constraint")
    assert isinstance(h.constraint, U1Constraint)
