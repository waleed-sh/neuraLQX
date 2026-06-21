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


"""Hilbert-space APIs for graph-based quantum models in neuraLQX.

The package defines local quantum-number domains, flat discrete Hilbert spaces,
graph-aware state layouts, constraints, random-state generation, proposal
moves, and specialized U(1) and SU(2) gauge-invariant spaces. Public imports
from this package are intended to let callers describe both generic discrete
spaces and graph-structured lattice-gauge spaces without depending on internal
layout modules.
"""

from .constraint import AbstractConstraint
from .constraint import AbstractDiscreteConstraint
from .constraint import AndConstraint
from .constraint import CallableDiscreteConstraint
from .constraint import IdentityConstraint
from .constraint import LinearConstraint
from .constraint import NoConstraint
from .constraint import NotConstraint
from .constraint import OrConstraint

from .layout import DofKind
from .layout import GraphDofLayout
from .layout import GraphHilbertLayout
from .layout import HilbertStateView
from .layout import SiteDof

from .local import AbstractLocalSpace
from .local import ExplicitLocalSpace
from .local import HeterogeneousLocalSpace
from .local import LocalRange
from .local import VectorRange
from .local import as_local_space

from .proposal import AbstractProposalMove
from .proposal import AdjacentSiteUpdate
from .proposal import UniformSiteUpdate
from .proposal import propose

from .random import random_state

from .space import AbstractDiscreteHilbert
from .space import AbstractHilbertSpace
from .space import DiscreteHilbertSpace
from .space import HeterogeneousDiscreteHilbert
from .space import HeterogeneousHilbert
from .space import HomogeneousDiscreteHilbert
from .space import HomogeneousHilbert

from .u1 import ConstrainedU1Hilbert
from .u1 import FreeEdgeFlipAllGauge
from .u1 import FreeEdgeFlipSingleGauge
from .u1 import PlaquetteFlipAllGauge
from .u1 import PlaquetteFlipSingleGauge
from .u1 import U1GaugeInvariantHilbert
from .u1 import U1Hilbert
from .u1 import UnconstrainedU1Hilbert

__all__ = [
    "AbstractConstraint",
    "AbstractDiscreteConstraint",
    "AbstractDiscreteHilbert",
    "AbstractHilbertSpace",
    "AbstractLocalSpace",
    "AbstractProposalMove",
    "AdjacentSiteUpdate",
    "AndConstraint",
    "CallableDiscreteConstraint",
    "DiscreteHilbertSpace",
    "DofKind",
    "ExplicitLocalSpace",
    "FreeEdgeFlipAllGauge",
    "FreeEdgeFlipSingleGauge",
    "GraphDofLayout",
    "GraphHilbertLayout",
    "HeterogeneousLocalSpace",
    "HeterogeneousDiscreteHilbert",
    "HeterogeneousHilbert",
    "HilbertStateView",
    "HomogeneousDiscreteHilbert",
    "HomogeneousHilbert",
    "IdentityConstraint",
    "LinearConstraint",
    "LocalRange",
    "NoConstraint",
    "NotConstraint",
    "OrConstraint",
    "PlaquetteFlipAllGauge",
    "PlaquetteFlipSingleGauge",
    "SiteDof",
    "U1GaugeInvariantHilbert",
    "U1Hilbert",
    "UniformSiteUpdate",
    "UnconstrainedU1Hilbert",
    "VectorRange",
    "as_local_space",
    "ConstrainedU1Hilbert",
    "propose",
    "random_state",
]
