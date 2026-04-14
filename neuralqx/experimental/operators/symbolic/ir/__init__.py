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


"""Symbolic operator intermediate representation (IR)."""

from neuralqx.experimental.operators.symbolic._deprecation import (
    deprecate_symbolic_public_api,
)

from .expressions import AmplitudeExpr
from .expressions import coerce_amplitude_expr

from .predicates import PredicateExpr
from .predicates import coerce_predicate_expr

from .update import UpdateOp
from .update import UpdateProgram

from .term import EmissionSpec
from .term import KBodyIteratorSpec
from .term import SymbolicIRTerm

from .program import SymbolicOperatorIR
from .validate import validate_symbolic_ir

__all__ = [
    "AmplitudeExpr",
    "coerce_amplitude_expr",
    "EmissionSpec",
    "KBodyIteratorSpec",
    "PredicateExpr",
    "coerce_predicate_expr",
    "SymbolicIRTerm",
    "SymbolicOperatorIR",
    "UpdateOp",
    "UpdateProgram",
    "validate_symbolic_ir",
]

deprecate_symbolic_public_api(
    globals(),
)
