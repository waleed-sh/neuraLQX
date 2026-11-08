#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


"""Operator interfaces, algebra wrappers, and prebuilt model operators.

Computational operators expose matrix rows for local Monte Carlo estimators.
The wrappers compose these operators or mark expectations, such as the lazy
positive product represented by `Squared`.
"""

from .abstract import AbstractOperator
from .computational import ComputationalOperator
from .connectivity import coalesce_connections
from .discrete import DiscreteOperator
from .wrappers import ComputationalWrappedOperator
from .wrappers import InverseExpectationCost
from .wrappers import PenaltyCost
from .wrappers import ProductOperator
from .wrappers import ScaledOperator
from .wrappers import Squared
from .wrappers import SumOfSquares
from .wrappers import SumOperator
from .wrappers import VertexSum
from .wrappers import WrappedOperator
from .wrappers import penalty_expectation_gradient
from .wrappers import penalty_expectation_value
from .wrappers import penalty_is_linear
from .wrappers import penalty_linear_scale
from .wrappers import penalty_local_value_coefficients

# Prebuilt operators import the public base classes above.
from . import prebuilt  # isort: skip

__all__ = [
    "AbstractOperator",
    "ComputationalOperator",
    "ComputationalWrappedOperator",
    "DiscreteOperator",
    "InverseExpectationCost",
    "PenaltyCost",
    "ProductOperator",
    "ScaledOperator",
    "Squared",
    "SumOfSquares",
    "SumOperator",
    "VertexSum",
    "WrappedOperator",
    "coalesce_connections",
    "penalty_expectation_gradient",
    "penalty_expectation_value",
    "penalty_is_linear",
    "penalty_linear_scale",
    "penalty_local_value_coefficients",
    "prebuilt",
]
