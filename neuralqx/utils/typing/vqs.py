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


"""Variational-state type aliases."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeAlias

if TYPE_CHECKING:
    from neuralqx.utils.stats import Stats
    from neuralqx.vqs.estimators.options import LocalEstimatorOptions
    from neuralqx.vqs.estimators.statistics import LocalEstimators

EstimatorRule: TypeAlias = (
    "Callable[[Any, Any, Any, LocalEstimatorOptions], LocalEstimators]"
)
"""Dispatch rule signature for local-estimator evaluation."""

PenaltyExpectationRule: TypeAlias = "Callable[[Any, Any, dict[str, Any]], Stats]"
"""Dispatch rule signature for penalty-cost expectation evaluation."""

PenaltyForcesRule: TypeAlias = "Callable[[Any, Any, dict[str, Any]], tuple[Stats, Any]]"
"""Dispatch rule signature for penalty-cost force evaluation."""

PenaltyGradientRule: TypeAlias = (
    "Callable[[Any, Any, dict[str, Any], Any], tuple[Stats, Any]]"
)
"""Dispatch rule signature for penalty-cost gradient evaluation."""

__all__ = [
    "EstimatorRule",
    "PenaltyExpectationRule",
    "PenaltyForcesRule",
    "PenaltyGradientRule",
]
