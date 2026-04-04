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


"""Symbolic operator compiler package."""

from neuralqx.experimental.operators.symbolic.compiler.compiler import SymbolicCompiler
from neuralqx.experimental.operators.symbolic.compiler.compiler import (
    compile_symbolic_operator,
)

from neuralqx.experimental.operators.symbolic.compiler.core.options import (
    SymbolicCompilerOptions,
)

from neuralqx.experimental.operators.symbolic.compiler.core.artifact import (
    SymbolicCompiledArtifact,
)

from neuralqx.experimental.operators.symbolic.compiler.core.context import (
    SymbolicCompilationContext,
)

from neuralqx.experimental.operators.symbolic.compiler.core.signature import (
    SymbolicCacheKey,
)
from neuralqx.experimental.operators.symbolic.compiler.core.signature import (
    SymbolicCompilationSignature,
)

from neuralqx.experimental.operators.symbolic.compiler.defaults import (
    default_symbolic_artifact_store,
)
from neuralqx.experimental.operators.symbolic.compiler.defaults import (
    default_symbolic_lowerer_registry,
)
from neuralqx.experimental.operators.symbolic.compiler.defaults import (
    default_symbolic_pass_pipeline,
)

__all__ = [
    # Primary API
    "SymbolicCompiler",
    "compile_symbolic_operator",
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
    "default_symbolic_artifact_store",
]
