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


import importlib
import pytest
from dataclasses import FrozenInstanceError, dataclass


@pytest.fixture
def smod():
    return importlib.import_module("neuralqx.utils.symmetries.types.symmetry")


EDGE1 = (0, 1)
EDGE2 = (1, 2)


def test_map_edge_delegates_to_getitem(smod):
    Symmetry = smod.Symmetry

    class Swap01(Symmetry):
        def __getitem__(self, e):
            a, b = e
            return (b, a)

    s = Swap01()
    assert s.map_edge(EDGE1) == (1, 0)
    assert s.map_edge(EDGE2) == (2, 1)


def test_call_delegates_to_map_edge(smod):
    Symmetry = smod.Symmetry

    class Identity(Symmetry):
        def __getitem__(self, e):
            return e

    s = Identity()
    assert s(EDGE1) == EDGE1
    assert s(EDGE2) == EDGE2


def test_map_edge_raises_typeerror_if_getitem_missing(smod):
    Symmetry = smod.Symmetry

    class MissingGetItem(Symmetry):
        pass

    s = MissingGetItem()

    with pytest.raises(
        TypeError, match=r"MissingGetItem must implement __getitem__\(edge\)"
    ):
        s.map_edge(EDGE1)


def test_map_edge_error_chains_original_exception(smod):
    Symmetry = smod.Symmetry

    class BrokenGetItem(Symmetry):
        def __getitem__(self, e):
            raise RuntimeError("boom")

    s = BrokenGetItem()

    with pytest.raises(TypeError) as excinfo:
        s.map_edge(EDGE1)

    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "boom" in str(excinfo.value.__cause__)


def test_override_map_edge_without_getitem_is_allowed(smod):
    Symmetry = smod.Symmetry

    class CustomMapEdge(Symmetry):
        def map_edge(self, e):
            return ("mapped", e)

    s = CustomMapEdge()
    assert s.map_edge(EDGE1) == ("mapped", EDGE1)
    assert s(EDGE2) == ("mapped", EDGE2)


def test_symmetry_base_is_hashable(smod):
    Symmetry = smod.Symmetry

    class Identity(Symmetry):
        def __getitem__(self, e):
            return e

    s = Identity()
    d = {s: "ok"}
    assert d[s] == "ok"


def test_symmetry_subclass_is_not_automatically_frozen_unless_dataclass(smod):
    """
    Regression/contract:
    Symmetry itself is a frozen dataclass, but subclasses are NOT automatically frozen
    unless they are also dataclasses (or implement their own immutability).
    """
    Symmetry = smod.Symmetry

    class PlainSubclass(Symmetry):
        def __getitem__(self, e):
            return e

    s = PlainSubclass()

    s.new_attr = 123
    assert s.new_attr == 123


def test_symmetry_subclass_can_be_frozen_if_declared_as_dataclass(smod):
    Symmetry = smod.Symmetry

    @dataclass(frozen=True)
    class FrozenSubclass(Symmetry):
        tag: str = "X"

        def __getitem__(self, e):
            return e

    s = FrozenSubclass()

    with pytest.raises(FrozenInstanceError):
        s.tag = "Y"

    with pytest.raises(FrozenInstanceError):
        s.new_attr = 1

    d = {s: "ok"}
    assert d[s] == "ok"


@pytest.fixture
def utils():
    return importlib.import_module("neuralqx.utils.symmetries.types.utils")


@pytest.fixture
def perm_mod():
    return importlib.import_module("neuralqx.utils.symmetries.types.permutation")


@pytest.fixture
def cyc_mod():
    return importlib.import_module("neuralqx.utils.symmetries.types.cycle")


@pytest.fixture
def auto_mod():
    return importlib.import_module("neuralqx.utils.symmetries.types.automorphism")


def _rev_edge(e):
    if len(e) == 2:
        a, b = e
        return (b, a)
    elif len(e) == 3:
        a, b, k = e
        return (b, a, k)
    raise AssertionError(f"Unexpected edge arity: {e}")


def test_permutation_getitem_respects_orientation(utils, perm_mod):
    Permutation = perm_mod.Permutation

    e = (0, 1, 7)
    e_rev = _rev_edge(e)
    img = (2, 3, 9)

    mapping_canon = {
        utils._canon_edge(e): utils._canon_edge(img),
    }
    domain = [e, e_rev]

    p = Permutation(mapping_canon=mapping_canon, domain=domain)

    out = p[e]
    out_rev = p[e_rev]

    assert out_rev == _rev_edge(out)

    assert p.map_edge(e) == out
    assert p(e) == out


def test_permutation_iter_yields_in_domain_order_and_orientation(utils, perm_mod):
    Permutation = perm_mod.Permutation

    e = (0, 1, 0)
    e_rev = _rev_edge(e)
    img = (5, 6, 1)

    mapping_canon = {utils._canon_edge(e): utils._canon_edge(img)}

    domain = [e_rev, e]

    p = Permutation(mapping_canon=mapping_canon, domain=domain)

    items = list(p)
    assert len(items) == len(domain)

    assert items[0][0] == e_rev
    assert items[0][1] == p[e_rev]

    assert items[1][0] == e
    assert items[1][1] == p[e]


def test_permutation_len_is_mapping_size_not_domain_size(utils, perm_mod):
    Permutation = perm_mod.Permutation

    e1 = (0, 1, 0)
    e2 = (1, 2, 0)
    img1 = (10, 11, 0)
    img2 = (11, 12, 0)

    mapping_canon = {
        utils._canon_edge(e1): utils._canon_edge(img1),
        utils._canon_edge(e2): utils._canon_edge(img2),
    }

    domain = [e1]

    p = Permutation(mapping_canon=mapping_canon, domain=domain)
    assert len(p) == 2

    assert len(list(p)) == 1


def test_permutation_missing_edge_raises_keyerror(utils, perm_mod):
    Permutation = perm_mod.Permutation

    mapping_canon = {utils._canon_edge((0, 1, 0)): utils._canon_edge((2, 3, 0))}
    p = Permutation(mapping_canon=mapping_canon, domain=[(0, 1, 0)])

    with pytest.raises(KeyError):
        _ = p[(999, 1000, 0)]


def test_permutation_is_frozen(utils, perm_mod):
    Permutation = perm_mod.Permutation
    mapping_canon = {utils._canon_edge((0, 1, 0)): utils._canon_edge((2, 3, 0))}
    p = Permutation(mapping_canon=mapping_canon, domain=[(0, 1, 0)])
    with pytest.raises(FrozenInstanceError):
        p.name = "x"


def test_cycle_getitem_respects_orientation_and_shift_field(utils, cyc_mod):
    Cycle = cyc_mod.Cycle

    e = (0, 1, 0)
    e_rev = _rev_edge(e)
    img = (2, 3, 0)

    mapping_canon = {utils._canon_edge(e): utils._canon_edge(img)}
    c = Cycle(mapping_canon=mapping_canon, domain=[e, e_rev], shift=3)

    assert c.shift == 3
    assert c.name == "cyc"

    out = c[e]
    out_rev = c[e_rev]
    assert out_rev == _rev_edge(out)

    assert c.map_edge(e) == out
    assert c(e) == out


def test_cycle_iter_and_len(utils, cyc_mod):
    Cycle = cyc_mod.Cycle

    e1 = (0, 1, 0)
    e2 = (1, 2, 0)
    img1 = (10, 11, 0)
    img2 = (11, 12, 0)

    mapping_canon = {
        utils._canon_edge(e1): utils._canon_edge(img1),
        utils._canon_edge(e2): utils._canon_edge(img2),
    }
    domain = [_rev_edge(e2), e1]

    c = Cycle(mapping_canon=mapping_canon, domain=domain, shift=-1)

    assert len(c) == 2
    items = list(c)
    assert [pair[0] for pair in items] == domain
    assert [pair[1] for pair in items] == [c[d] for d in domain]


def test_cycle_is_frozen(utils, cyc_mod):
    Cycle = cyc_mod.Cycle
    mapping_canon = {utils._canon_edge((0, 1, 0)): utils._canon_edge((2, 3, 0))}
    c = Cycle(mapping_canon=mapping_canon, domain=[(0, 1, 0)], shift=0)
    with pytest.raises(FrozenInstanceError):
        c.shift = 2


def test_automorphism_getitem_respects_orientation_and_map_vertex(utils, auto_mod):
    Automorphism = auto_mod.Automorphism

    vertex_map = {0: 2, 1: 3, 2: 0, 3: 1}

    e = (0, 1, 7)
    e_rev = _rev_edge(e)
    img = (2, 3, 9)

    edge_map_canon = {utils._canon_edge(e): utils._canon_edge(img)}
    a = Automorphism(
        vertex_map=vertex_map,
        edge_map_canon=edge_map_canon,
        domain=[e, e_rev],
    )

    assert a.name == "auto"
    assert a.map_vertex(0) == 2
    assert a.map_vertex(3) == 1

    out = a[e]
    out_rev = a[e_rev]
    assert out_rev == _rev_edge(out)

    assert a.map_edge(e) == out
    assert a(e) == out


def test_automorphism_iter_and_len_are_edge_map_based(utils, auto_mod):
    Automorphism = auto_mod.Automorphism

    vertex_map = {0: 1, 1: 0}
    e1 = (0, 1, 0)
    e2 = (2, 3, 0)
    img1 = (10, 11, 0)
    img2 = (12, 13, 0)

    edge_map_canon = {
        utils._canon_edge(e1): utils._canon_edge(img1),
        utils._canon_edge(e2): utils._canon_edge(img2),
    }

    domain = [e1]

    a = Automorphism(
        vertex_map=vertex_map, edge_map_canon=edge_map_canon, domain=domain
    )

    assert len(a) == 2

    assert len(list(a)) == 1


def test_automorphism_missing_vertex_or_edge_raises_keyerror(utils, auto_mod):
    Automorphism = auto_mod.Automorphism

    a = Automorphism(
        vertex_map={0: 1},
        edge_map_canon={utils._canon_edge((0, 1, 0)): utils._canon_edge((2, 3, 0))},
        domain=[(0, 1, 0)],
    )

    with pytest.raises(KeyError):
        _ = a.map_vertex(999)

    with pytest.raises(KeyError):
        _ = a[(9, 9, 0)]


def test_automorphism_is_frozen(utils, auto_mod):
    Automorphism = auto_mod.Automorphism

    a = Automorphism(
        vertex_map={0: 1},
        edge_map_canon={utils._canon_edge((0, 1, 0)): utils._canon_edge((2, 3, 0))},
        domain=[(0, 1, 0)],
    )

    with pytest.raises(FrozenInstanceError):
        a.vertex_map = {}


@pytest.mark.parametrize(
    "e, expected",
    [
        ((0, 1, 0), (0, 1, 0)),
        ((1, 0, 0), (0, 1, 0)),
        ((2, 2, 7), (2, 2, 7)),
        ((-3, 5, 9), (-3, 5, 9)),
        ((5, -3, 9), (-3, 5, 9)),
    ],
)
def test_canon_edge_orders_endpoints_only(utils, e, expected):
    assert utils._canon_edge(e) == expected


def test_canon_edge_preserves_key(utils):
    e = (9, 1, 123)
    c = utils._canon_edge(e)
    assert c[-1] == 123


def test_canon_edge_idempotent(utils):
    e = (10, 3, 0)
    assert utils._canon_edge(utils._canon_edge(e)) == utils._canon_edge(e)


@pytest.mark.parametrize(
    "e, expected",
    [
        ((0, 1, 0), False),
        ((1, 0, 0), True),
        ((2, 2, 7), False),
    ],
)
def test_is_reversed_vs_canon(utils, e, expected):
    assert utils._is_reversed_vs_canon(e) is expected


def test_is_reversed_vs_canon_consistent_with_canon_edge(utils):
    e = (7, 2, 5)
    assert utils._is_reversed_vs_canon(e) == (e != utils._canon_edge(e))


def test_orient_like_keeps_key_and_uses_like_orientation(utils):
    edge_canon = (0, 9, 123)

    like_forward = (0, 1, 999)
    like_reversed = (1, 0, 999)

    out_fwd = utils._orient_like(edge_canon, like_forward)
    out_rev = utils._orient_like(edge_canon, like_reversed)

    assert out_fwd == (0, 9, 123)
    assert out_rev == (9, 0, 123)
    assert out_fwd[-1] == 123 and out_rev[-1] == 123


def test_orient_like_like_equal_endpoints_no_flip(utils):
    edge_canon = (2, 5, 0)
    like = (7, 7, 99)
    assert utils._orient_like(edge_canon, like) == edge_canon


def test_orient_like_is_involution_wrt_like(utils):

    edge_canon = (0, 9, 0)
    like_reversed = (1, 0, 0)

    once = utils._orient_like(edge_canon, like_reversed)
    twice = utils._orient_like(once, like_reversed)

    assert twice == edge_canon


def test_orient_like_matches_like_reversed_flag(utils):

    canon_edge = (0, 9, 0)

    like_forward = (0, 2, 7)
    like_reverse = (2, 0, 7)

    assert (
        utils._is_reversed_vs_canon(utils._orient_like(canon_edge, like_forward))
        is False
    )
    assert (
        utils._is_reversed_vs_canon(utils._orient_like(canon_edge, like_reverse))
        is True
    )


def test_orient_like_requires_canonical_input_for_strong_invariants(utils):

    noncanon = (9, 0, 0)
    like_forward = (0, 1, 0)

    out = utils._orient_like(noncanon, like_forward)
    assert out == noncanon
