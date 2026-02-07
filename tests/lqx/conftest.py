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


import types
import pytest

nk = pytest.importorskip("netket")

import neuralqx as nqx

from neuralqx.lqx.abstract_lqx_model import AbstractLqxModel
from neuralqx.lqx.abstract_lqx_interface import AbstractLqxInterface


class DummyPrinter:
    def __init__(self):
        self.messages = []

    def print(self, *args, **kwargs):
        self.messages.append(" ".join(str(a) for a in args))


class DummyHilbertInterface:

    def __init__(
        self, *, graph, nk_hilbert, is_gauge_invariant: bool, gauge_dimensions: int = 1
    ):
        self.graph = graph
        self.is_gauge_invariant = is_gauge_invariant
        self.gauge_dimensions = gauge_dimensions
        self.core = types.SimpleNamespace(hilbert=nk_hilbert)

        self.dimensions = getattr(nk_hilbert, "n_states", None)
        if self.dimensions is None:
            self.dimensions = int(getattr(nk_hilbert, "size"))


class DummyGaugeGroup:
    def __init__(self, constraint, dimensions):
        self.constraint = constraint
        self.dimensions = dimensions


class ToyModel(AbstractLqxModel):

    def __init__(
        self,
        hilbert,
        graph,
        gauge_group,
        *,
        constraint,
        computational=True,
        spacetime_dimensions=4,
        model_name="ToyModel",
    ):
        super().__init__(
            hilbert=hilbert,
            graph=graph,
            gauge_group=gauge_group,
            computational=computational,
            spacetime_dimensions=spacetime_dimensions,
            model_name=model_name,
        )
        self._constraint = constraint

    @property
    def constraint(self):
        return self._constraint

    @constraint.setter
    def constraint(self, val):
        self._constraint = val


class ToyInterface(AbstractLqxInterface):

    def __init__(
        self,
        H,
        gauge_group,
        *,
        model_constraint,
        lazy_load=True,
        model_name="ToyInterface",
    ):
        self._model = ToyModel(
            H, H.graph, gauge_group, constraint=model_constraint, model_name="ToyModel"
        )
        super().__init__(H, gauge_group, lazy_load=lazy_load, model_name=model_name)

        self.printer = DummyPrinter()


@pytest.fixture
def nk_spin_2():
    hi = nk.hilbert.Spin(s=0.5, N=2)
    g = nk.graph.Hypercube(length=2, n_dim=1, pbc=False)
    return g, hi


@pytest.fixture
def toy_objects_gauge_invariant(nk_spin_2):
    g, hi = nk_spin_2
    H = DummyHilbertInterface(
        graph=g, nk_hilbert=hi, is_gauge_invariant=True, gauge_dimensions=1
    )

    z0 = nk.operator.spin.sigmaz(hi, 0)
    z1 = nk.operator.spin.sigmaz(hi, 1)
    model_constraint = z0 + z1

    gx = nk.operator.spin.sigmax(hi, 0)
    GG = DummyGaugeGroup(constraint=gx, dimensions=1)

    iface = ToyInterface(H, GG, model_constraint=model_constraint, lazy_load=True)
    return H, GG, model_constraint, iface


@pytest.fixture
def toy_objects_not_gauge_invariant(nk_spin_2):
    g, hi = nk_spin_2
    H = DummyHilbertInterface(
        graph=g, nk_hilbert=hi, is_gauge_invariant=False, gauge_dimensions=1
    )

    z0 = nk.operator.spin.sigmaz(hi, 0)
    z1 = nk.operator.spin.sigmaz(hi, 1)
    model_constraint = z0 + z1

    gx = nk.operator.spin.sigmax(hi, 0)
    GG = DummyGaugeGroup(constraint=gx, dimensions=1)

    iface = ToyInterface(H, GG, model_constraint=model_constraint, lazy_load=True)
    return H, GG, model_constraint, iface
