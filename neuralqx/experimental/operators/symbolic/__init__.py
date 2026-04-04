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
Symbolic operator subsystem for neuraLQX.

Typical workflow::

    from neuralqx.experimental.operators.symbolic import DOperator
    from neuralqx.experimental.operators.symbolic.dsl import site, shift, swap

    hop = (
        DOperator(hi, "hopping")
        .for_each_pair("i", "j")
        .where(site("i") > 0)
        .emit(shift("i", -1).shift("j", +1), amplitude=1.0)
        .build()
    )
    compiled = hop.compile()
    xp, mels = compiled.get_conn_padded(x_batch)
"""

#
#
#   Primary user-facing entry points

from neuralqx.experimental.operators.symbolic.dsl import DOperator
from neuralqx.experimental.operators.symbolic.dsl import Update
from neuralqx.experimental.operators.symbolic.dsl import ExpressionContext
from neuralqx.experimental.operators.symbolic.dsl import SiteSelector
from neuralqx.experimental.operators.symbolic.dsl import affine
from neuralqx.experimental.operators.symbolic.dsl import identity
from neuralqx.experimental.operators.symbolic.dsl import permute
from neuralqx.experimental.operators.symbolic.dsl import scatter
from neuralqx.experimental.operators.symbolic.dsl import shift
from neuralqx.experimental.operators.symbolic.dsl import shift_mod
from neuralqx.experimental.operators.symbolic.dsl import site
from neuralqx.experimental.operators.symbolic.dsl import emitted
from neuralqx.experimental.operators.symbolic.dsl import swap
from neuralqx.experimental.operators.symbolic.dsl import symbol
from neuralqx.experimental.operators.symbolic.dsl import write

#
#
#   Operator types

from neuralqx.experimental.operators.symbolic.core import AbstractSymbolicOperator
from neuralqx.experimental.operators.symbolic.core import CompiledOperator
from neuralqx.experimental.operators.symbolic.core import SymbolicOperator
from neuralqx.experimental.operators.symbolic.core import SymbolicOperatorSum

#
#
#   Compiler

from neuralqx.experimental.operators.symbolic.compiler import SymbolicCacheKey
from neuralqx.experimental.operators.symbolic.compiler import SymbolicCompiledArtifact
from neuralqx.experimental.operators.symbolic.compiler import SymbolicCompilationContext
from neuralqx.experimental.operators.symbolic.compiler import (
    SymbolicCompilationSignature,
)
from neuralqx.experimental.operators.symbolic.compiler import SymbolicCompilerOptions

#
#
#   IR

from neuralqx.experimental.operators.symbolic.ir import AmplitudeExpr
from neuralqx.experimental.operators.symbolic.ir import EmissionSpec
from neuralqx.experimental.operators.symbolic.ir import KBodyIteratorSpec
from neuralqx.experimental.operators.symbolic.ir import PredicateExpr
from neuralqx.experimental.operators.symbolic.ir import SymbolicIRTerm
from neuralqx.experimental.operators.symbolic.ir import SymbolicOperatorIR
from neuralqx.experimental.operators.symbolic.ir import UpdateOp
from neuralqx.experimental.operators.symbolic.ir import UpdateProgram
from neuralqx.experimental.operators.symbolic.ir import coerce_amplitude_expr
from neuralqx.experimental.operators.symbolic.ir import coerce_predicate_expr

__all__ = [
    # Primary API
    "DOperator",
    "Update",
    "shift",
    "shift_mod",
    "write",
    "swap",
    "permute",
    "affine",
    "scatter",
    "identity",
    "site",
    "emitted",
    "symbol",
    "SiteSelector",
    # Operator types
    "SymbolicOperator",
    "CompiledOperator",
    "AbstractSymbolicOperator",
    "SymbolicOperatorSum",
    "ExpressionContext",
    # Compiler
    "SymbolicCompilerOptions",
    "SymbolicCompiledArtifact",
    "SymbolicCompilationContext",
    "SymbolicCacheKey",
    "SymbolicCompilationSignature",
    # IR
    "SymbolicOperatorIR",
    "SymbolicIRTerm",
    "EmissionSpec",
    "KBodyIteratorSpec",
    "AmplitudeExpr",
    "PredicateExpr",
    "UpdateProgram",
    "UpdateOp",
    "coerce_amplitude_expr",
    "coerce_predicate_expr",
]
