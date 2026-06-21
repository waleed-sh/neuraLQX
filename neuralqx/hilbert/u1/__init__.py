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


"""Public U(1)^N Hilbert-space API.

The package exports unconstrained and constructive gauge-fixed U(1) edge
Hilbert spaces, gauge-fixing data structures, gauge-fixing utilities, proposal
move descriptors, and modular arithmetic helpers from their implementation
subpackages.
"""

from .constraints import GaugeFixingSpec
from .constraints import GaugeFixingTopology
from .constraints import GaugeRelation
from .constraints import U1CyclicGaugeFixingError
from .constraints import U1GaugeConstraint
from .constraints import U1GaugeFixingError
from .constraints import auto_gauge_fixing
from .constraints import build_topology
from .constraints import edge_to_index
from .constraints import parse_gauge_fixing
from .constraints import pretty_gauge_fixing
from .proposals import AbstractU1Move
from .proposals import FreeEdgeFlipAllGauge
from .proposals import FreeEdgeFlipSingleGauge
from .proposals import PlaquetteFlipAllGauge
from .proposals import PlaquetteFlipSingleGauge
from .spaces import ConstrainedU1Hilbert
from .spaces import U1GaugeInvariantHilbert
from .spaces import U1Hilbert
from .spaces import UnconstrainedU1Hilbert
from .utils import modular_add
from .utils import signed_modular_sum
from .utils import wrap_values

__all__ = [
    "AbstractU1Move",
    "ConstrainedU1Hilbert",
    "FreeEdgeFlipAllGauge",
    "FreeEdgeFlipSingleGauge",
    "GaugeFixingSpec",
    "GaugeFixingTopology",
    "GaugeRelation",
    "PlaquetteFlipAllGauge",
    "PlaquetteFlipSingleGauge",
    "U1CyclicGaugeFixingError",
    "U1GaugeConstraint",
    "U1GaugeFixingError",
    "U1GaugeInvariantHilbert",
    "U1Hilbert",
    "UnconstrainedU1Hilbert",
    "auto_gauge_fixing",
    "build_topology",
    "edge_to_index",
    "modular_add",
    "parse_gauge_fixing",
    "pretty_gauge_fixing",
    "signed_modular_sum",
    "wrap_values",
]
