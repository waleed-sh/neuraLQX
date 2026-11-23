# Copyright (c) 2026 The neuraLQX and nkDSL Authors - All rights reserved.
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
Symbolic operators compiled to native neuraLQX v2 kernels.

Typical workflow::

    from neuralqx.operator.symbolic import SymbolicDiscreteJaxOperator
    from neuralqx.operator.symbolic.dsl import site, hop

    hopping_op = (
        SymbolicDiscreteJaxOperator(hi, "hopping")
        .for_each_pair("i", "j")
        .where(site("i") > 0)
        .emit(hop("i", "j"), matrix_element=1.0)
        .build()
    )
    compiled = hopping_op.compile()
    xp, mels = compiled.get_conn_padded(x_batch)
"""

#
#
#   Primary user-facing entry points

from neuralqx.operator.symbolic.compiler import DEFAULT_SYMBOLIC_OPERATOR_LOWERING

#
#
#   Compiler
from neuralqx.operator.symbolic.compiler import SymbolicCacheKey
from neuralqx.operator.symbolic.compiler import SymbolicCompilationContext
from neuralqx.operator.symbolic.compiler import SymbolicCompilationSignature
from neuralqx.operator.symbolic.compiler import SymbolicCompiledArtifact
from neuralqx.operator.symbolic.compiler import SymbolicCompiler
from neuralqx.operator.symbolic.compiler import SymbolicCompilerOptions
from neuralqx.operator.symbolic.compiler import SymbolicOperatorLoweringRegistry
from neuralqx.operator.symbolic.compiler import SymbolicOperatorLoweringTarget
from neuralqx.operator.symbolic.compiler import compile_symbolic_operator
from neuralqx.operator.symbolic.compiler import default_symbolic_artifact_store
from neuralqx.operator.symbolic.compiler import default_symbolic_lowerer_registry
from neuralqx.operator.symbolic.compiler import (
    default_symbolic_operator_lowering_registry,
)
from neuralqx.operator.symbolic.compiler import default_symbolic_pass_pipeline
from neuralqx.operator.symbolic.compiler.defaults import LOWERING_REGISTRY

#
#
#   Config
from neuralqx.operator.symbolic.configs import cfg

#
#
#   Operator types
from neuralqx.operator.symbolic.core import AbstractSymbolicOperator
from neuralqx.operator.symbolic.core import CompiledOperator
from neuralqx.operator.symbolic.core import SymbolicOperator
from neuralqx.operator.symbolic.core import SymbolicOperatorSum
from neuralqx.operator.symbolic.dsl import AbstractEmissionClause
from neuralqx.operator.symbolic.dsl import AbstractIteratorClause
from neuralqx.operator.symbolic.dsl import AbstractPredicateClause
from neuralqx.operator.symbolic.dsl import EmissionClauseSpec
from neuralqx.operator.symbolic.dsl import ExpressionContext
from neuralqx.operator.symbolic.dsl import SiteSelector
from neuralqx.operator.symbolic.dsl import SymbolicDiscreteJaxOperator
from neuralqx.operator.symbolic.dsl import SymbolicOperatorBuilder
from neuralqx.operator.symbolic.dsl import Update
from neuralqx.operator.symbolic.dsl import affine
from neuralqx.operator.symbolic.dsl import available_emission_clause_names
from neuralqx.operator.symbolic.dsl import available_iterator_clause_names
from neuralqx.operator.symbolic.dsl import available_predicate_clause_names
from neuralqx.operator.symbolic.dsl import emitted
from neuralqx.operator.symbolic.dsl import hop
from neuralqx.operator.symbolic.dsl import identity
from neuralqx.operator.symbolic.dsl import permute
from neuralqx.operator.symbolic.dsl import register
from neuralqx.operator.symbolic.dsl import register_emission_clause
from neuralqx.operator.symbolic.dsl import register_iterator_clause
from neuralqx.operator.symbolic.dsl import register_predicate_clause
from neuralqx.operator.symbolic.dsl import scatter
from neuralqx.operator.symbolic.dsl import shift
from neuralqx.operator.symbolic.dsl import shift_mod
from neuralqx.operator.symbolic.dsl import site
from neuralqx.operator.symbolic.dsl import source_index
from neuralqx.operator.symbolic.dsl import swap
from neuralqx.operator.symbolic.dsl import symbol
from neuralqx.operator.symbolic.dsl import target_index
from neuralqx.operator.symbolic.dsl import write

#
#
#   Errors
from neuralqx.operator.symbolic.errors import ConfigError
from neuralqx.operator.symbolic.errors import ConfigValidationError
from neuralqx.operator.symbolic.errors import SymbolicCompilationError
from neuralqx.operator.symbolic.errors import SymbolicCompilerError
from neuralqx.operator.symbolic.errors import SymbolicDiagnosticsError
from neuralqx.operator.symbolic.errors import SymbolicError
from neuralqx.operator.symbolic.errors import SymbolicOperatorError
from neuralqx.operator.symbolic.errors import SymbolicOperatorExecutionError
from neuralqx.operator.symbolic.errors import UnknownOptionError

#
#
#   IR
from neuralqx.operator.symbolic.ir import AmplitudeExpr
from neuralqx.operator.symbolic.ir import EmissionSpec
from neuralqx.operator.symbolic.ir import KBodyIteratorSpec
from neuralqx.operator.symbolic.ir import PredicateExpr
from neuralqx.operator.symbolic.ir import SymbolicIRTerm
from neuralqx.operator.symbolic.ir import SymbolicOperatorIR
from neuralqx.operator.symbolic.ir import UpdateOp
from neuralqx.operator.symbolic.ir import UpdateProgram
from neuralqx.operator.symbolic.ir import coerce_amplitude_expr
from neuralqx.operator.symbolic.ir import coerce_predicate_expr
from neuralqx.operator.symbolic.warnings import LoweringWarning
from neuralqx.operator.symbolic.warnings import RegistrationWarning
from neuralqx.operator.symbolic.warnings import SymbolicWarning

config = cfg
"""Runtime configuration. Alias of :data:`cfg`, including ``LOWERING``."""

__all__ = [  # noqa: RUF022 - grouped by public API role
    # Primary API
    "SymbolicOperatorBuilder",
    "SymbolicDiscreteJaxOperator",
    "Update",
    "shift",
    "shift_mod",
    "hop",
    "write",
    "swap",
    "permute",
    "affine",
    "scatter",
    "identity",
    "site",
    "emitted",
    "symbol",
    "source_index",
    "target_index",
    "SiteSelector",
    "AbstractIteratorClause",
    "AbstractPredicateClause",
    "AbstractEmissionClause",
    "EmissionClauseSpec",
    "register",
    "register_iterator_clause",
    "register_predicate_clause",
    "register_emission_clause",
    "available_iterator_clause_names",
    "available_predicate_clause_names",
    "available_emission_clause_names",
    # Operator types
    "SymbolicOperator",
    "CompiledOperator",
    "AbstractSymbolicOperator",
    "SymbolicOperatorSum",
    "ExpressionContext",
    # Compiler
    "SymbolicCompiler",
    "compile_symbolic_operator",
    "SymbolicCompilerOptions",
    "SymbolicCompiledArtifact",
    "SymbolicCompilationContext",
    "SymbolicCacheKey",
    "SymbolicCompilationSignature",
    "default_symbolic_pass_pipeline",
    "default_symbolic_lowerer_registry",
    "default_symbolic_operator_lowering_registry",
    "default_symbolic_artifact_store",
    "DEFAULT_SYMBOLIC_OPERATOR_LOWERING",
    "SymbolicOperatorLoweringRegistry",
    "SymbolicOperatorLoweringTarget",
    "LOWERING_REGISTRY",
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
    # Errors
    "ConfigError",
    "ConfigValidationError",
    "UnknownOptionError",
    "SymbolicError",
    "SymbolicOperatorError",
    "SymbolicOperatorExecutionError",
    "SymbolicCompilationError",
    "SymbolicCompilerError",
    "SymbolicDiagnosticsError",
    "SymbolicWarning",
    "LoweringWarning",
    "RegistrationWarning",
    # Configs
    "cfg",
    "config",
]
