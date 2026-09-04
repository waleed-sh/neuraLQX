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


"""Solver fallback combinators."""

from __future__ import annotations

from typing import Any

from ._api import has_non_finite


def nan_fallback(primary: Any, fallback: Any):
    """Return a solver that falls back if the primary result is non-finite."""

    def solve(operator: Any, rhs: Any, *, x0: Any | None = None, **kwargs: Any):
        primary_solution, primary_info = primary(operator, rhs, x0=x0, **kwargs)
        if bool(has_non_finite(primary_solution)):
            fallback_solution, fallback_info = fallback(operator, rhs, x0=x0, **kwargs)
            return fallback_solution, {
                "solver_fallback": True,
                "primary": primary_info,
                "fallback": fallback_info,
            }
        return primary_solution, {
            "solver_fallback": False,
            "primary": primary_info,
        }

    return solve


__all__ = ["nan_fallback"]
