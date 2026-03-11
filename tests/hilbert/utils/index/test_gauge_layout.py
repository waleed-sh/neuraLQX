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


from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from neuralqx.hilbert.utils.layout._abstract_layout import AbstractBasisLayout
from neuralqx.hilbert.u1.layout import GaugeCoord
from neuralqx.hilbert.u1.layout import StridedGaugeCopyLayout
from neuralqx.hilbert.u1.layout import GaugeLayout


class _DummyLayout(AbstractBasisLayout[int]):

    def __init__(self, n: int):
        self._validate_positive_int("n", n)
        self._n = n

    @property
    def size(self) -> int:
        return self._n

    def site_of(self, coord: int) -> int:
        self._validate_int("coord", coord)
        self.validate_site(coord)
        return coord

    def coord_of(self, site: int) -> int:
        self.validate_site(site)
        return site


def test_abstract_basis_layout_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractBasisLayout()


def test_abstract_helpers_encode_decode_aliases():
    layout = _DummyLayout(5)
    assert layout.encode_coord(3) == 3
    assert layout.decode_coord(4) == 4


def test_abstract_helpers_validate_site_valid():
    layout = _DummyLayout(3)
    layout.validate_site(0)
    layout.validate_site(2)


@pytest.mark.parametrize("site", [-1, 3, 100])
def test_abstract_helpers_validate_site_out_of_range(site):
    layout = _DummyLayout(3)
    with pytest.raises(IndexError):
        layout.validate_site(site)


@pytest.mark.parametrize("site", [1.5, "1", None, object()])
def test_abstract_helpers_validate_site_type(site):
    layout = _DummyLayout(3)
    with pytest.raises(TypeError):
        layout.validate_site(site)


def test_abstract_helpers_contains_site():
    layout = _DummyLayout(4)
    assert layout.contains_site(0) is True
    assert layout.contains_site(3) is True
    assert layout.contains_site(-1) is False
    assert layout.contains_site(4) is False
    assert layout.contains_site(1.0) is False
    assert layout.contains_site("1") is False


def test_abstract_helpers_len():
    layout = _DummyLayout(7)
    assert len(layout) == 7


def test_abstract_static_validate_int():
    AbstractBasisLayout._validate_int("x", 1)  # no raise
    with pytest.raises(TypeError):
        AbstractBasisLayout._validate_int("x", 1.0)


def test_abstract_static_validate_positive_int():
    AbstractBasisLayout._validate_positive_int("x", 1)  # no raise
    with pytest.raises(ValueError):
        AbstractBasisLayout._validate_positive_int("x", 0)
    with pytest.raises(ValueError):
        AbstractBasisLayout._validate_positive_int("x", -1)
    with pytest.raises(TypeError):
        AbstractBasisLayout._validate_positive_int("x", 1.0)


def test_gauge_coord_fields():
    c = GaugeCoord(gauge_copy=2, edge_index=5)
    assert c.gauge_copy == 2
    assert c.edge_index == 5


def test_gauge_coord_is_frozen():
    c = GaugeCoord(gauge_copy=0, edge_index=1)
    with pytest.raises(FrozenInstanceError):
        c.gauge_copy = 3  # type: ignore[misc]


def test_strided_layout_basic_properties():
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    assert layout.edges_per_copy == 4
    assert layout.gauge_dimensions == 3
    assert layout.size == 12
    assert len(layout) == 12
    assert layout.shape == (3, 4)


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        (dict(edges_per_copy=0, gauge_dimensions=2), ValueError),
        (dict(edges_per_copy=-1, gauge_dimensions=2), ValueError),
        (dict(edges_per_copy=3, gauge_dimensions=0), ValueError),
        (dict(edges_per_copy=3, gauge_dimensions=-2), ValueError),
        (dict(edges_per_copy=3.0, gauge_dimensions=2), TypeError),
        (dict(edges_per_copy=3, gauge_dimensions=2.0), TypeError),
    ],
)
def test_strided_layout_constructor_validation(kwargs, exc):
    with pytest.raises(exc):
        StridedGaugeCopyLayout(**kwargs)


def test_validate_indices_valid():
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    layout.validate_indices(0, 0)
    layout.validate_indices(2, 3)


@pytest.mark.parametrize("g,e", [(-1, 0), (3, 0), (100, 0)])
def test_validate_indices_bad_gauge_copy(g, e):
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    with pytest.raises(IndexError):
        layout.validate_indices(g, e)


@pytest.mark.parametrize("g,e", [(0, -1), (0, 4), (0, 100)])
def test_validate_indices_bad_edge_index(g, e):
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    with pytest.raises(IndexError):
        layout.validate_indices(g, e)


@pytest.mark.parametrize("g,e", [(1.0, 0), (0, 2.0), ("0", 1), (1, None)])
def test_validate_indices_type_errors(g, e):
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    with pytest.raises(TypeError):
        layout.validate_indices(g, e)


def test_validate_coord_valid():
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    layout.validate_coord(GaugeCoord(gauge_copy=1, edge_index=2))


def test_validate_coord_wrong_type():
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    with pytest.raises(TypeError):
        layout.validate_coord((1, 2))  # type: ignore[arg-type]


def test_validate_coord_out_of_range():
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    with pytest.raises(IndexError):
        layout.validate_coord(GaugeCoord(gauge_copy=3, edge_index=0))


def test_encode_matches_formula():
    layout = StridedGaugeCopyLayout(edges_per_copy=5, gauge_dimensions=4)
    assert layout.encode(0, 0) == 0
    assert layout.encode(0, 3) == 3
    assert layout.encode(2, 1) == 11  # 2*5 + 1
    assert layout.encode(3, 4) == 19  # max valid


def test_coord_of_matches_formula():
    layout = StridedGaugeCopyLayout(edges_per_copy=5, gauge_dimensions=4)
    assert layout.coord_of(0) == GaugeCoord(0, 0)
    assert layout.coord_of(3) == GaugeCoord(0, 3)
    assert layout.coord_of(11) == GaugeCoord(2, 1)
    assert layout.coord_of(19) == GaugeCoord(3, 4)


def test_site_of_matches_formula():
    layout = StridedGaugeCopyLayout(edges_per_copy=5, gauge_dimensions=4)
    assert layout.site_of(GaugeCoord(0, 0)) == 0
    assert layout.site_of(GaugeCoord(2, 1)) == 11
    assert layout.site_of(GaugeCoord(3, 4)) == 19


def test_roundtrip_all_sites():
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    for s in range(layout.size):
        c = layout.coord_of(s)
        assert layout.site_of(c) == s


def test_roundtrip_all_coords():
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    for g in range(layout.gauge_dimensions):
        for e in range(layout.edges_per_copy):
            c = GaugeCoord(gauge_copy=g, edge_index=e)
            s = layout.site_of(c)
            assert layout.coord_of(s) == c


def test_encode_equals_site_of():
    layout = StridedGaugeCopyLayout(edges_per_copy=7, gauge_dimensions=2)
    c = GaugeCoord(gauge_copy=1, edge_index=6)
    assert layout.encode(1, 6) == layout.site_of(c)


def test_coord_of_invalid_site_raises():
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    with pytest.raises(IndexError):
        layout.coord_of(-1)
    with pytest.raises(IndexError):
        layout.coord_of(layout.size)


def test_coord_of_invalid_site_type_raises():
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    with pytest.raises(TypeError):
        layout.coord_of(1.0)  # type: ignore[arg-type]


def test_site_of_invalid_coord_type_raises():
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    with pytest.raises(TypeError):
        layout.site_of((1, 2))  # type: ignore[arg-type]


def test_decode_returns_tuple_and_warns():
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    with pytest.warns(FutureWarning, match=r"StridedGaugeCopyLayout\.decode"):
        out = layout.decode(6)
    assert out == (1, 2)


def test_site_legacy_wrapper_warns_and_matches_encode():
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    with pytest.warns(FutureWarning, match=r"StridedGaugeCopyLayout\.site"):
        s = layout.site(edge_index=2, gauge_copy=1)
    assert s == layout.encode(gauge_copy=1, edge_index=2)
    assert s == 6


def test_copy_slice():
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    assert layout.copy_slice(0) == slice(0, 4)
    assert layout.copy_slice(1) == slice(4, 8)
    assert layout.copy_slice(2) == slice(8, 12)


@pytest.mark.parametrize("g", [-1, 3])
def test_copy_slice_out_of_range(g):
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    with pytest.raises(IndexError):
        layout.copy_slice(g)


def test_copy_slice_type_error():
    layout = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)
    with pytest.raises(TypeError):
        layout.copy_slice(1.0)  # type: ignore[arg-type]


def test_iter_coords_order_matches_flattening():
    layout = StridedGaugeCopyLayout(edges_per_copy=3, gauge_dimensions=2)
    coords = list(layout.iter_coords())
    assert coords == [
        GaugeCoord(0, 0),
        GaugeCoord(0, 1),
        GaugeCoord(0, 2),
        GaugeCoord(1, 0),
        GaugeCoord(1, 1),
        GaugeCoord(1, 2),
    ]


def test_iter_coords_has_size_elements_and_roundtrips():
    layout = StridedGaugeCopyLayout(edges_per_copy=5, gauge_dimensions=2)
    coords = list(layout.iter_coords())
    assert len(coords) == layout.size
    assert [layout.site_of(c) for c in coords] == list(range(layout.size))


def test_gauge_layout_alias_instantiation_warns():
    with pytest.warns(FutureWarning, match=r"deprecated class 'GaugeLayout'"):
        layout = GaugeLayout(edges_per_copy=4, gauge_dimensions=3)

    assert isinstance(layout, GaugeLayout)
    assert isinstance(layout, StridedGaugeCopyLayout)
    assert layout.size == 12


def test_gauge_layout_alias_behavior_matches_strided():
    with pytest.warns(FutureWarning):
        old = GaugeLayout(edges_per_copy=4, gauge_dimensions=3)

    new = StridedGaugeCopyLayout(edges_per_copy=4, gauge_dimensions=3)

    # compare a few core operations without using deprecated wrappers
    for s in range(new.size):
        assert old.coord_of(s) == new.coord_of(s)

    assert old.encode(2, 1) == new.encode(2, 1)
    assert old.copy_slice(1) == new.copy_slice(1)


def test_gauge_layout_bijection():
    from neuralqx.hilbert.u1.layout import StridedGaugeCopyLayout

    E = 7
    G = 4
    layout = StridedGaugeCopyLayout(edges_per_copy=E, gauge_dimensions=G)

    for g in range(G):
        for e in range(E):
            s = layout.encode(g, e)
            _c = layout.coord_of(s)
            g2, e2 = _c.gauge_copy, _c.edge_index
            assert (g2, e2) == (g, e)

    sites = {layout.encode(g, e) for g in range(G) for e in range(E)}
    assert sites == set(range(E * G))


def test_gauge_layout_range_checks():
    from neuralqx.hilbert.u1.layout import StridedGaugeCopyLayout

    layout = StridedGaugeCopyLayout(edges_per_copy=3, gauge_dimensions=2)

    with pytest.raises(IndexError, match="gauge_copy"):
        layout.encode(edge_index=0, gauge_copy=-1)

    with pytest.raises(IndexError, match="gauge_copy"):
        layout.encode(edge_index=0, gauge_copy=2)

    with pytest.raises(IndexError, match="edge_index"):
        layout.encode(edge_index=-1, gauge_copy=0)

    with pytest.raises(IndexError, match="edge_index"):
        layout.encode(edge_index=3, gauge_copy=0)

    with pytest.raises(IndexError, match="site"):
        layout.coord_of(-1)

    with pytest.raises(IndexError, match="site"):
        layout.coord_of(6)


def test_gauge_layout_is_frozen_and_sloted():
    from dataclasses import FrozenInstanceError
    from neuralqx.hilbert.u1.layout import StridedGaugeCopyLayout

    layout = StridedGaugeCopyLayout(edges_per_copy=2, gauge_dimensions=2)
    with pytest.raises(FrozenInstanceError):
        layout.edges_per_copy = 99
