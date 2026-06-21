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

import jax

from neuralqx.graph import AbstractGraph
from neuralqx.hilbert.u1.constraints import GaugeFixingSpec
from neuralqx.hilbert.u1.constraints import GaugeFixingTopology
from neuralqx.hilbert.u1.constraints import GaugeRelation

from .unconstrained import U1Hilbert

class U1GaugeInvariantHilbert(U1Hilbert):
    gauge_fixing: GaugeFixingSpec | None
    auto_constraint: bool
    relations: tuple[GaugeRelation, ...]
    topology: GaugeFixingTopology
    def __init__(
        self,
        graph: AbstractGraph,
        cutoff: int,
        step: int = 1,
        gauge_dimensions: int = 1,
        positive_qn: bool = False,
        qn_start: int | None = None,
        gauge_fixing: GaugeFixingSpec | None = None,
        auto_constraint: bool = False,
    ) -> None: ...
    @property
    def constraints_base(self) -> tuple[GaugeRelation, ...]: ...
    @property
    def gauge_fixing_topology(self) -> GaugeFixingTopology: ...
    def reimpose_gauge_fixing(self, states: Any) -> jax.Array: ...
    def check_states(self, states: Any) -> jax.Array: ...
    def is_gauge_invariant(self, states: Any) -> jax.Array: ...
    def random_state(
        self,
        key: jax.Array,
        size: int | tuple[int, ...] | None = None,
        *,
        dtype: Any | None = None,
        max_trials: int = 1024,
    ) -> jax.Array: ...
    def gauge_fixing_pretty(self) -> str: ...
    def __repr__(self) -> str: ...

ConstrainedU1Hilbert = U1GaugeInvariantHilbert
