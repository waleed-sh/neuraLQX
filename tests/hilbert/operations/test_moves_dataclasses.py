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


def test_move_classes_are_frozen_slots_and_hashable():
    from dataclasses import FrozenInstanceError
    from neuralqx.hilbert.u1.operations.moves import (
        Move,
        FreeEdgeFlipSingleGauge,
        FreeEdgeFlipAllGauge,
        PlaquetteFlipSingleGauge,
        PlaquetteFlipAllGauge,
    )

    m1 = FreeEdgeFlipSingleGauge()
    m2 = FreeEdgeFlipSingleGauge(n_edges=2, adjacency=True)
    m3 = FreeEdgeFlipAllGauge()
    p1 = PlaquetteFlipSingleGauge()
    p2 = PlaquetteFlipAllGauge()

    assert isinstance(m1, Move)
    assert isinstance(p1, Move)

    with pytest.raises(FrozenInstanceError):
        m1.n_edges = 99

    d = {m1: "a", m2: "b", m3: "c", p1: "d", p2: "e"}
    assert d[FreeEdgeFlipSingleGauge()] == "a"
    assert d[FreeEdgeFlipSingleGauge(n_edges=2, adjacency=True)] == "b"
