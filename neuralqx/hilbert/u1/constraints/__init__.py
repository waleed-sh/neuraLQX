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


"""Gauge-fixing constraints for U(1)^N spaces."""

from .auto import auto_gauge_fixing
from .constraint import U1GaugeConstraint
from .errors import U1CyclicGaugeFixingError
from .errors import U1GaugeFixingError
from .parser import edge_to_index
from .parser import parse_gauge_fixing
from .pretty import pretty_gauge_fixing
from .project import reimpose_gauge_fixing_batch
from .relation import GaugeRelation
from .topology import GaugeFixingTopology
from .topology import build_topology
from .types import GaugeFixingSpec

__all__ = [
    "GaugeFixingSpec",
    "GaugeFixingTopology",
    "GaugeRelation",
    "U1CyclicGaugeFixingError",
    "U1GaugeConstraint",
    "U1GaugeFixingError",
    "auto_gauge_fixing",
    "build_topology",
    "edge_to_index",
    "parse_gauge_fixing",
    "pretty_gauge_fixing",
    "reimpose_gauge_fixing_batch",
]
