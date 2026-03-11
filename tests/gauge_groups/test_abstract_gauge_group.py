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


def test_abstract_gauge_group_constraint_caching(neuralqx):
    try:
        from neuralqx.gauge_groups._abstract_gauge_group import AbstractGaugeGroup
    except Exception:
        from neuralqx.gauge_groups import AbstractGaugeGroup

    class DummyHilbert:
        def __init__(self):
            self.graph = object()
            self.gauge_dimensions = 7
            self.hilbert = "netket_hilbert_placeholder"

    class DummyGroup(AbstractGaugeGroup):
        def __init__(self, H, **kwargs):
            self._init_calls = 0
            super().__init__(H, **kwargs)

        @property
        def name(self) -> str:
            return "dummy"

        @property
        def is_abelian(self) -> bool:
            return True

        def init_constraint(self, *, computational=True, jax=False, lazy=True):
            self._init_calls += 1
            return object()

    H = DummyHilbert()
    g = DummyGroup(H, lazy=True, computational=True, jax=False)

    assert g._init_calls == 1
    c0 = g.constraint

    g.constraint = {"lazy": True, "computational": True, "jax": False, "reinit": False}
    assert g._init_calls == 1
    assert g.constraint is c0

    g.constraint = {"lazy": True, "computational": True, "jax": False, "reinit": True}
    assert g._init_calls == 2
    assert g.constraint is not c0


def test_abstract_gauge_group_flags(neuralqx):
    try:
        from neuralqx.gauge_groups._abstract_gauge_group import AbstractGaugeGroup
    except Exception:
        from neuralqx.gauge_groups import AbstractGaugeGroup

    class DummyHilbert:
        def __init__(self):
            self.graph = object()
            self.gauge_dimensions = 1
            self.hilbert = "hk"
            self.hilbert_netket = "nkh"

    class DummyGroup(AbstractGaugeGroup):
        @property
        def name(self) -> str:
            return "dummy"

        @property
        def is_abelian(self) -> bool:
            return True

        def init_constraint(self, *, computational=True, jax=False, lazy=True):
            return object()

    H = DummyHilbert()

    g1 = DummyGroup(H, computational=True, jax=True)
    assert g1.is_computational is True
    assert g1.is_jax is True

    g2 = DummyGroup(H, computational=False, jax=False)
    assert g2.is_computational is False
    assert g2.is_jax is False


def test_abstract_gauge_group_repr(neuralqx):
    try:
        from neuralqx.gauge_groups._abstract_gauge_group import AbstractGaugeGroup
    except Exception:
        from neuralqx.gauge_groups import AbstractGaugeGroup

    class DummyHilbert:
        def __init__(self):
            self.graph = object()
            self.gauge_dimensions = 1
            self.hilbert_netket = "HK_REPR"

    class DummyGroup(AbstractGaugeGroup):
        @property
        def name(self) -> str:
            return "dummy"

        @property
        def is_abelian(self) -> bool:
            return True

        def init_constraint(self, *, computational=True, jax=False, lazy=True):
            return object()

    g = DummyGroup(DummyHilbert())
    s = repr(g)
    assert "AbstractGaugeGroup" in s
    assert "HK_REPR" in s


def test_abstract_gauge_group_math_ops_return_notimplemented(neuralqx):
    try:
        from neuralqx.gauge_groups._abstract_gauge_group import AbstractGaugeGroup
    except Exception:
        from neuralqx.gauge_groups import AbstractGaugeGroup

    class DummyHilbert:
        def __init__(self):
            self.graph = object()
            self.gauge_dimensions = 1
            self.hilbert = "HK"

    class DummyGroup(AbstractGaugeGroup):
        @property
        def name(self) -> str:
            return "dummy"

        @property
        def is_abelian(self) -> bool:
            return True

        def init_constraint(self, *, computational=True, jax=False, lazy=True):
            return object()

    g = DummyGroup(DummyHilbert())
    assert g.__mul__(object()) is NotImplemented
    assert g.__rmul__(object()) is NotImplemented
    assert g.__imul__(object()) is NotImplemented
    assert g.__add__(object()) is NotImplemented
    assert g.__radd__(object()) is NotImplemented
    assert g.__iadd__(object()) is NotImplemented
    assert g.__pow__(2) is NotImplemented
