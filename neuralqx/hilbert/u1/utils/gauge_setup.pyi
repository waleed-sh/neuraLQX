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

from __future__ import annotations

from typing import Any

from neuralqx.hilbert.u1.constraints import GaugeFixingTopology
from neuralqx.hilbert.u1.constraints import GaugeRelation
from neuralqx.hilbert.u1.constraints import U1GaugeConstraint

def validate_integer_constraint_range(
    cutoff: int | float, step: int | float
) -> None: ...
def relations_from_inputs(space: Any) -> tuple[GaugeRelation, ...]: ...
def constraint_from_topology(
    space: Any,
    relations: tuple[GaugeRelation, ...],
    topology: GaugeFixingTopology,
) -> U1GaugeConstraint: ...
def constrained_dimension(space: Any, topology: GaugeFixingTopology) -> int: ...
