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


"""Expectation-value dispatch."""

from __future__ import annotations

from typing import Any

from plum import dispatch

from neuralqx.operator.wrappers import PenaltyCost
from neuralqx.vqs.estimators import Stats
from neuralqx.vqs.estimators import local_estimators

from .penalties import penalty_expect


def expect(vstate: Any, operator: Any, **kwargs: Any) -> Stats:
    """Estimate ``<operator>`` from local estimators."""
    return _expect(vstate, operator, kwargs)


@dispatch
def _expect(vstate: Any, operator: Any, kwargs: dict[str, Any]) -> Stats:
    return local_estimators(vstate, operator, **kwargs).to_stats()


@_expect.dispatch
def _expect_penalty(
    vstate: Any, operator: PenaltyCost, kwargs: dict[str, Any]
) -> Stats:
    return penalty_expect(vstate, operator, kwargs)


__all__ = ["expect"]
