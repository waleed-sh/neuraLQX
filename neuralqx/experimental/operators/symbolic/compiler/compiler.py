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


"""Symbolic operator compiler orchestrator."""

from __future__ import annotations

from typing import Any

from neuralqx.experimental.operators.symbolic.compiler.cache.store import (
    AbstractSymbolicArtifactStore,
)
from neuralqx.experimental.operators.symbolic.compiler.core.artifact import (
    SymbolicCompiledArtifact,
)
from neuralqx.experimental.operators.symbolic.compiler.core.context import (
    SymbolicCompilationContext,
)
from neuralqx.experimental.operators.symbolic.compiler.core.options import (
    SymbolicCompilerOptions,
)
from neuralqx.experimental.operators.symbolic.compiler.core.pipeline import (
    SymbolicPassPipeline,
)
from neuralqx.experimental.operators.symbolic.compiler.core.signature import (
    SymbolicCompilationSignature,
)
from neuralqx.experimental.operators.symbolic.compiler.lowering.registry import (
    SymbolicLowererRegistry,
)
from neuralqx.operators.types.computational_operator.jax import ComputationalJaxOperator


class SymbolicCompiler:
    """
    Orchestrates the symbolic operator compilation pipeline.

    The compiler accepts a symbolic operator (an
    :class:`~neuralqx.experimental.operators.symbolic.core.base.AbstractSymbolicOperator`),
    runs it through the registered pass pipeline, optionally resolves a cache
    hit, and, on a miss, invokes the appropriate lowerer to produce a
    concrete :class:`~neuralqx.operators.types.computational_operator.jax.ComputationalJaxOperator`.

    Typical usage::

        from neuralqx.experimental.operators.symbolic import SymbolicCompiler

        compiler = SymbolicCompiler()
        compiled_op = compiler.compile_operator(my_symbolic_op)
        xp, mels = compiled_op.get_conn_padded(x_batch)

    Args:
        pipeline: Pass pipeline to use.  Defaults to
            :func:`~neuralqx.experimental.operators.symbolic.compiler.defaults.default_symbolic_pass_pipeline`.
        lowerer_registry: Lowerer registry.  Defaults to
            :func:`~neuralqx.experimental.operators.symbolic.compiler.defaults.default_symbolic_lowerer_registry`.
        artifact_store: Artifact cache store.  Defaults to the module-level
            shared :func:`~neuralqx.experimental.operators.symbolic.compiler.defaults.default_symbolic_artifact_store`.
        options: Compiler options.  Defaults to :class:`SymbolicCompilerOptions` with all defaults.
    """

    def __init__(
        self,
        *,
        pipeline: SymbolicPassPipeline | None = None,
        lowerer_registry: SymbolicLowererRegistry | None = None,
        artifact_store: AbstractSymbolicArtifactStore | None = None,
        options: SymbolicCompilerOptions | None = None,
    ) -> None:
        from neuralqx.experimental.operators.symbolic.compiler.defaults import (
            default_symbolic_artifact_store,
        )
        from neuralqx.experimental.operators.symbolic.compiler.defaults import (
            default_symbolic_lowerer_registry,
        )
        from neuralqx.experimental.operators.symbolic.compiler.defaults import (
            default_symbolic_pass_pipeline,
        )

        self._pipeline = pipeline or default_symbolic_pass_pipeline()
        self._registry = lowerer_registry or default_symbolic_lowerer_registry()
        self._store = artifact_store or default_symbolic_artifact_store()
        self._options = options or SymbolicCompilerOptions()

    def compile(
        self,
        operator: Any,
        *,
        options: SymbolicCompilerOptions | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SymbolicCompiledArtifact:
        """
        Compiles a symbolic operator to a :class:`SymbolicCompiledArtifact`.

        Steps:

        1. Extracts the :class:`~neuralqx.experimental.operators.symbolic.ir.program.SymbolicOperatorIR`
           from the operator via ``to_ir()``.
        2. Creates a :class:`~neuralqx.experimental.operators.symbolic.compiler.core.context.SymbolicCompilationContext`.
        3. Runs the pre-cache pass stage.
        4. Computes the cache key (when caching is enabled) and checks the
           artifact store.
        5. On a cache hit: returns the cached artifact.
        6. On a cache miss: runs post-cache passes, resolves a lowerer,
           lowers the operator, stores the artifact, and returns it.

        Args:
            operator: Symbolic operator with a ``to_ir()`` method.
            options: Override compiler options for this single invocation.
            metadata: Extra metadata forwarded to the compilation context.

        Returns:
            Compiled artifact.

        Raises:
            :class:`~neuralqx.utils.errors.SymbolicCompilerError`: On
                unrecoverable compilation failure.
        """
        effective_options = options or self._options

        # Extract IR
        try:
            ir = operator.to_ir()
        except Exception as exc:
            from neuralqx.utils.errors import SymbolicCompilerError

            raise SymbolicCompilerError(
                f"Failed to extract IR from operator {operator!r}: {exc}"
            ) from exc

        # Build context
        context = SymbolicCompilationContext(
            operator=operator,
            ir=ir,
            options=effective_options,
            metadata=metadata,
        )

        # Pre-cache passes
        try:
            self._pipeline.run_pre_cache(context)
        except Exception as exc:
            from neuralqx.utils.errors import SymbolicCompilerError

            raise SymbolicCompilerError(
                f"Pre-cache pass failed for operator {ir.operator_name!r}: {exc}"
            ) from exc

        # Cache lookup
        if effective_options.cache_enabled:
            sig = SymbolicCompilationSignature.from_context(context)
            cache_key = sig.build_cache_key(
                namespace=effective_options.cache_namespace,
            )
            cached = self._store.get(cache_key)
            if cached is not None:
                return cached
        else:
            cache_key = None

        # Post-cache passes
        try:
            self._pipeline.run_post_cache(context)
        except Exception as exc:
            from neuralqx.utils.errors import SymbolicCompilerError

            raise SymbolicCompilerError(
                f"Post-cache pass failed for operator {ir.operator_name!r}: {exc}"
            ) from exc

        # Resolve lowerer and lower
        try:
            lowerer = self._registry.resolve(context)
            artifact = lowerer.lower(context)
        except Exception as exc:
            from neuralqx.utils.errors import SymbolicCompilerError

            raise SymbolicCompilerError(
                f"Lowering failed for operator {ir.operator_name!r}: {exc}"
            ) from exc

        # Attach cache key to artifact when caching is enabled
        if effective_options.cache_enabled and cache_key is not None:
            artifact = SymbolicCompiledArtifact.create(
                operator_name=artifact.operator_name,
                backend=artifact.backend,
                lowerer_name=artifact.lowerer_name,
                compiled_operator=artifact.compiled_operator,
                cache_key=cache_key,
                pass_reports=artifact.pass_reports,
                metadata=artifact.metadata_map(),
            )
            self._store.put(cache_key, artifact)

        return artifact

    def compile_operator(
        self,
        operator: Any,
        *,
        options: SymbolicCompilerOptions | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ComputationalJaxOperator:
        """
        Compiles a symbolic operator and returns the executable operator directly.

        Convenience wrapper around :meth:`compile` that unwraps the artifact.

        Args:
            operator: Symbolic operator with a ``to_ir()`` method.
            options: Override compiler options.
            metadata: Extra metadata for the context.

        Returns:
            Executable :class:`~neuralqx.operators.types.computational_operator.jax.ComputationalJaxOperator`.
        """
        artifact = self.compile(operator, options=options, metadata=metadata)
        return artifact.compiled_operator

    def clear_cache(self) -> None:
        """Clears all entries from the artifact store."""
        self._store.clear()

    @property
    def cache_size(self) -> int:
        """Returns the number of cached artifacts."""
        return len(self._store)

    @property
    def pass_names(self) -> tuple[str, ...]:
        """Returns the full ordered pass name sequence."""
        return self._pipeline.pass_names()

    @property
    def lowerer_names(self) -> tuple[str, ...]:
        """Returns registered lowerer names."""
        return self._registry.lowerer_names

    def __repr__(self) -> str:
        return (
            f"SymbolicCompiler("
            f"passes={self.pass_names!r}, "
            f"lowerers={self.lowerer_names!r}, "
            f"cache_size={self.cache_size})"
        )


_DEFAULT_COMPILER: SymbolicCompiler | None = None


def compile_symbolic_operator(
    operator: Any,
    *,
    options: SymbolicCompilerOptions | None = None,
    metadata: dict[str, Any] | None = None,
) -> ComputationalJaxOperator:
    """
    Module-level convenience function for one-shot symbolic compilation.

    Uses the module-level shared :class:`SymbolicCompiler` instance (lazily
    created). The shared compiler reuses the global in-process artifact cache.

    Args:
        operator: Symbolic operator with a ``to_ir()`` method.
        options: Override compiler options.
        metadata: Extra metadata for the context.

    Returns:
        Executable :class:`~neuralqx.operators.types.computational_operator.jax.ComputationalJaxOperator`.

    Example::

        from neuralqx.experimental.operators.symbolic import compile_symbolic_operator

        compiled_op = compile_symbolic_operator(my_symbolic_op)
        xp, mels = compiled_op.get_conn_padded(x_batch)
    """
    global _DEFAULT_COMPILER  # noqa: PLW0603
    if _DEFAULT_COMPILER is None:
        _DEFAULT_COMPILER = SymbolicCompiler()
    return _DEFAULT_COMPILER.compile_operator(
        operator, options=options, metadata=metadata
    )


__all__ = [
    "SymbolicCompiler",
    "compile_symbolic_operator",
]
