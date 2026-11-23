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


"""Symbolic operator compiler package."""

from neuralqx.operator.symbolic.compiler.compiler import SymbolicCompiler
from neuralqx.operator.symbolic.compiler.compiler import compile_symbolic_operator
from neuralqx.operator.symbolic.compiler.compiler import reset_default_symbolic_compiler
from neuralqx.operator.symbolic.compiler.core.artifact import SymbolicCompiledArtifact
from neuralqx.operator.symbolic.compiler.core.context import SymbolicCompilationContext
from neuralqx.operator.symbolic.compiler.core.options import SymbolicCompilerOptions
from neuralqx.operator.symbolic.compiler.core.signature import SymbolicCacheKey
from neuralqx.operator.symbolic.compiler.core.signature import (
    SymbolicCompilationSignature,
)
from neuralqx.operator.symbolic.compiler.defaults import default_symbolic_artifact_store
from neuralqx.operator.symbolic.compiler.defaults import (
    default_symbolic_lowerer_registry,
)
from neuralqx.operator.symbolic.compiler.defaults import (
    default_symbolic_operator_lowering_registry,
)
from neuralqx.operator.symbolic.compiler.defaults import default_symbolic_pass_pipeline
from neuralqx.operator.symbolic.compiler.defaults import (
    reset_default_symbolic_singletons,
)
from neuralqx.operator.symbolic.compiler.lowering.operator_registry import (
    DEFAULT_SYMBOLIC_OPERATOR_LOWERING,
)
from neuralqx.operator.symbolic.compiler.lowering.operator_registry import (
    SymbolicOperatorLoweringRegistry,
)
from neuralqx.operator.symbolic.compiler.lowering.operator_registry import (
    SymbolicOperatorLoweringTarget,
)
from neuralqx.operator.symbolic.errors import SymbolicCompilerError

__all__ = [  # noqa: RUF022 - grouped by public API role
    # Primary API
    "SymbolicCompiler",
    "compile_symbolic_operator",
    "reset_default_symbolic_compiler",
    # Options
    "SymbolicCompilerOptions",
    # Artifact and context
    "SymbolicCompiledArtifact",
    "SymbolicCompilationContext",
    "SymbolicCacheKey",
    "SymbolicCompilationSignature",
    # Default factories
    "default_symbolic_pass_pipeline",
    "default_symbolic_lowerer_registry",
    "default_symbolic_operator_lowering_registry",
    "default_symbolic_artifact_store",
    "reset_default_symbolic_singletons",
    # Operator-lowering targets
    "DEFAULT_SYMBOLIC_OPERATOR_LOWERING",
    "SymbolicOperatorLoweringRegistry",
    "SymbolicOperatorLoweringTarget",
    # Errors
    "SymbolicCompilerError",
]
