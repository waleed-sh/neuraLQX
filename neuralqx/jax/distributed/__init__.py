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


"""Distributed execution coordinator for neuraLQX JAX backends."""

# Import default planners for their decorator side effects.
from neuralqx.utils.typing import DistributedAxis
from neuralqx.utils.typing import DistributedLayoutKind
from neuralqx.utils.typing import DistributedPhase
from neuralqx.utils.typing import DistributedPlan
from neuralqx.utils.typing import DistributedWorkload
from neuralqx.utils.typing import DistributionRequest
from neuralqx.utils.typing import Planner
from neuralqx.utils.typing import WorkloadBuilder

from . import defaults as _defaults  # noqa: F401
from .coordinator import DEFAULT_COORDINATOR
from .coordinator import DistributedCoordinator
from .coordinator import distributed_planner
from .coordinator import plan_distributed_execution
from .coordinator import with_distributed_plan
from .scheduling import schedule_balanced_slots

__all__ = [
    "DEFAULT_COORDINATOR",
    "DistributedAxis",
    "DistributedCoordinator",
    "DistributedLayoutKind",
    "DistributedPhase",
    "DistributedPlan",
    "DistributedWorkload",
    "DistributionRequest",
    "Planner",
    "WorkloadBuilder",
    "distributed_planner",
    "plan_distributed_execution",
    "schedule_balanced_slots",
    "with_distributed_plan",
]
