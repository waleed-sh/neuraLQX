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


"""Compatibility facade for operator dispatch planning.

The implementation lives in :mod:`neuralqx.jax.distributed`; this module keeps
the previous operator-sharding import path stable for internal and downstream
users while routing policy decisions through the distributed coordinator.
"""

from __future__ import annotations

import jax
from neuralqx.jax.distributed.coordinator import plan_distributed_execution
from neuralqx.jax.distributed.scheduling import schedule_balanced_slots
from neuralqx.utils.typing import DistributedPhase
from neuralqx.utils.typing import DistributedPlan
from neuralqx.utils.typing import DistributedWorkload

OperatorDispatchPlan = DistributedPlan
OperatorDispatchMode = str


def plan_operator_dispatch(
    term_costs: tuple[int, ...],
    *,
    sample_count: int | None,
    enabled: bool | None,
) -> OperatorDispatchPlan | None:
    """Return the distributed coordinator plan for operator local values."""
    return plan_distributed_execution(
        DistributedPhase.OPERATOR_LOCAL_VALUES,
        DistributedWorkload.operator_local_values(
            operator_costs=term_costs,
            sample_count=sample_count,
            enabled=enabled,
        ),
    )


def scheduled_operator_term_indices(
    term_costs: tuple[int, ...],
    partitions: int,
) -> jax.Array:
    """Return cost-balanced operator term slots."""
    return schedule_balanced_slots(term_costs, partitions)


__all__ = [
    "OperatorDispatchMode",
    "OperatorDispatchPlan",
    "plan_operator_dispatch",
    "scheduled_operator_term_indices",
]
