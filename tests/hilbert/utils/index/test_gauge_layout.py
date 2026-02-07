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


def test_gauge_layout_bijection():
    from neuralqx.hilbert.utils.layout.gauge_strided import GaugeLayout

    E = 7
    G = 4
    layout = GaugeLayout(edges_per_copy=E, gauge_dimensions=G)

    for g in range(G):
        for e in range(E):
            s = layout.site(e, g)
            g2, e2 = layout.decode(s)
            assert (g2, e2) == (g, e)

    sites = {layout.site(e, g) for g in range(G) for e in range(E)}
    assert sites == set(range(E * G))


def test_gauge_layout_range_checks():
    from neuralqx.hilbert.utils.layout.gauge_strided import GaugeLayout

    layout = GaugeLayout(edges_per_copy=3, gauge_dimensions=2)

    with pytest.raises(ValueError, match="gauge_copy"):
        layout.site(0, -1)

    with pytest.raises(ValueError, match="gauge_copy"):
        layout.site(0, 2)

    with pytest.raises(ValueError, match="edge_index"):
        layout.site(-1, 0)

    with pytest.raises(ValueError, match="edge_index"):
        layout.site(3, 0)

    with pytest.raises(ValueError, match="site"):
        layout.decode(-1)

    with pytest.raises(ValueError, match="site"):
        layout.decode(6)


def test_gauge_layout_is_frozen_and_sloted():
    from dataclasses import FrozenInstanceError
    from neuralqx.hilbert.utils.layout.gauge_strided import GaugeLayout

    layout = GaugeLayout(edges_per_copy=2, gauge_dimensions=2)
    with pytest.raises(FrozenInstanceError):
        layout.edges_per_copy = 99
