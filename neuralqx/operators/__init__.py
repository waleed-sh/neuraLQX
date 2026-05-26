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

"""
This module contains the implementation of different operators used in different gravitational
models such as holonomies, n-point functions, etc.
"""

from . import types
from .holonomies import holonomy
from .number import get_quantum_number, shifted_get_quantum_number
from .coloring import charge_coloring, coloring
from .coloring import n_point_function
from ._lazy import (
    InverseExpectationCost,
    PenaltyCost,
    penalty_expectation_gradient,
    penalty_expectation_value,
    penalty_is_linear,
    penalty_linear_scale,
    penalty_local_value_coefficients,
)
from . import computational

__all__ = [
    "types",
    "holonomy",
    "get_quantum_number",
    "shifted_get_quantum_number",
    "charge_coloring",
    "coloring",
    "n_point_function",
    "PenaltyCost",
    "InverseExpectationCost",
    "penalty_expectation_gradient",
    "penalty_expectation_value",
    "penalty_is_linear",
    "penalty_linear_scale",
    "penalty_local_value_coefficients",
    "computational",
]
