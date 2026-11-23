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


"""Symbolic operator compiler orchestrator."""

from __future__ import annotations

import dataclasses
from typing import Any

from neuralqx.operator.symbolic.compiler.cache.store import (
    AbstractSymbolicArtifactStore,
)
from neuralqx.operator.symbolic.compiler.core.artifact import SymbolicCompiledArtifact
from neuralqx.operator.symbolic.compiler.core.context import SymbolicCompilationContext
from neuralqx.operator.symbolic.compiler.core.options import SymbolicCompilerOptions
from neuralqx.operator.symbolic.compiler.core.pipeline import SymbolicPassPipeline
from neuralqx.operator.symbolic.compiler.core.signature import (
    SymbolicCompilationSignature,
)
from neuralqx.operator.symbolic.compiler.lowering.operator_registry import (
    SymbolicOperatorLoweringRegistry,
)
from neuralqx.operator.symbolic.compiler.lowering.registry import (
    SymbolicLowererRegistry,
)
from neuralqx.operator.symbolic.debug import event as debug_event
from neuralqx.operator.symbolic.errors import SymbolicCompilerError


class SymbolicCompiler:
    """
    Orchestrates the symbolic operator compilation pipeline.

    The compiler accepts a symbolic operator (an
    :class:`~neuralqx.operator.symbolic.core.base.AbstractSymbolicOperator`),
    runs it through the registered pass pipeline, optionally resolves a cache
    hit, and, on a miss, invokes the appropriate lowerer to produce a
    concrete executable operator instance.

    Typical usage::

        from neuralqx.operator.symbolic import SymbolicCompiler

        compiler = SymbolicCompiler()
        compiled_op = compiler.compile_operator(my_symbolic_op)
        xp, mels = compiled_op.get_conn_padded(x_batch)

    Args:
        pipeline: Pass pipeline to use.  Defaults to
            :func:`~neuralqx.operator.symbolic.compiler.defaults.default_symbolic_pass_pipeline`.
        lowerer_registry: Lowerer registry.  Defaults to
            :func:`~neuralqx.operator.symbolic.compiler.defaults.default_symbolic_lowerer_registry`.
        operator_lowering_registry: Registry mapping operator-lowering names
            to target classes and connection methods. Defaults to
            :func:`~neuralqx.operator.symbolic.compiler.defaults.default_symbolic_operator_lowering_registry`.
        artifact_store: Artifact cache store.  Defaults to the module-level
            shared :func:`~neuralqx.operator.symbolic.compiler.defaults.default_symbolic_artifact_store`.
        options: Compiler options.  Defaults to :class:`SymbolicCompilerOptions` with all defaults.
        deduplicate_connected_components: Convenience override for
            ``options.deduplicate_connected_components``.
        operator_lowering: Convenience override for ``options.operator_lowering``.
        optimize_kernels: Convenience override for ``options.optimize_kernels``.
    """

    def __init__(
        self,
        *,
        pipeline: SymbolicPassPipeline | None = None,
        lowerer_registry: SymbolicLowererRegistry | None = None,
        operator_lowering_registry: SymbolicOperatorLoweringRegistry | None = None,
        artifact_store: AbstractSymbolicArtifactStore | None = None,
        options: SymbolicCompilerOptions | None = None,
        backend_preference: str | None = None,
        cache_enabled: bool | None = None,
        deduplicate_connected_components: bool | None = None,
        operator_lowering: str | None = None,
        optimize_kernels: bool | None = None,
    ) -> None:
        from neuralqx.operator.symbolic.compiler.cache.store import (
            InMemorySymbolicArtifactStore,
        )
        from neuralqx.operator.symbolic.compiler.defaults import (
            default_symbolic_lowerer_registry,
        )
        from neuralqx.operator.symbolic.compiler.defaults import (
            default_symbolic_operator_lowering_registry,
        )
        from neuralqx.operator.symbolic.compiler.defaults import (
            default_symbolic_pass_pipeline,
        )

        self._pipeline = pipeline or default_symbolic_pass_pipeline()
        self._operator_lowering_registry = (
            operator_lowering_registry
            if operator_lowering_registry is not None
            else default_symbolic_operator_lowering_registry()
        )
        self._registry = lowerer_registry or default_symbolic_lowerer_registry(
            operator_lowering_registry=self._operator_lowering_registry
        )
        self._store = (
            artifact_store
            if artifact_store is not None
            else InMemorySymbolicArtifactStore()
        )

        resolved_options = options or SymbolicCompilerOptions()
        if (
            backend_preference is not None
            or cache_enabled is not None
            or deduplicate_connected_components is not None
            or operator_lowering is not None
            or optimize_kernels is not None
        ):
            overrides: dict[str, Any] = {
                "backend_preference": backend_preference,
                "cache_enabled": cache_enabled,
                "deduplicate_connected_components": deduplicate_connected_components,
                "operator_lowering": operator_lowering,
                "optimize_kernels": optimize_kernels,
            }
            resolved_options = dataclasses.replace(
                resolved_options,
                **{key: value for key, value in overrides.items() if value is not None},
            )
        self._options = resolved_options

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

        1. Extracts the :class:`~neuralqx.operator.symbolic.ir.program.SymbolicOperatorIR`
           from the operator via ``to_ir()``.
        2. Creates a :class:`~neuralqx.operator.symbolic.compiler.core.context.SymbolicCompilationContext`.
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
            :class:`~neuralqx.operator.symbolic.errors.SymbolicCompilerError`: On
                unrecoverable compilation failure.
        """
        effective_options = options or self._options
        debug_event(
            "starting symbolic compilation",
            scope="compile",
            tag="COMPILER",
            backend_preference=effective_options.backend_preference,
            operator_lowering=effective_options.operator_lowering,
            cache_enabled=effective_options.cache_enabled,
            strict_validation=effective_options.strict_validation,
        )

        # Extract IR
        try:
            ir = operator.to_ir()
        except Exception as exc:
            raise SymbolicCompilerError(
                f"Failed to extract IR from operator {operator!r}: {exc}"
            ) from exc
        debug_event(
            "extracted symbolic ir from operator",
            scope="ir",
            tag="IR",
            operator_name=ir.operator_name,
            term_count=ir.term_count,
        )

        # Bind the target once so configuration changes cannot alter this compile.
        requested_lowering = effective_options.operator_lowering
        if requested_lowering is None:
            requested_lowering = self._options.operator_lowering
        try:
            lowering_target = self._operator_lowering_registry.resolve(
                requested_lowering
            )
        except Exception as exc:
            raise SymbolicCompilerError(
                f"Cannot resolve operator lowering for {ir.operator_name!r}: {exc}"
            ) from exc
        effective_options = dataclasses.replace(
            effective_options, operator_lowering=lowering_target.name
        )

        # Build context
        context = SymbolicCompilationContext(
            operator=operator,
            ir=ir,
            options=effective_options,
            metadata=metadata,
        )
        context.set_analysis("operator_lowering_target", lowering_target)
        debug_event(
            "created compilation context",
            scope="compile",
            tag="COMPILER",
            operator_name=ir.operator_name,
            metadata_keys=tuple(sorted((metadata or {}).keys())),
        )

        # Pre-cache passes
        try:
            self._pipeline.run_pre_cache(context)
        except Exception as exc:
            raise SymbolicCompilerError(
                f"Pre-cache pass failed for operator {ir.operator_name!r}: {exc}"
            ) from exc
        debug_event(
            "completed pre-cache pass stage",
            scope="compile",
            tag="COMPILER",
            operator_name=ir.operator_name,
            pass_count=len(context.pass_reports),
        )

        # Cache lookup
        if effective_options.cache_enabled:
            sig = SymbolicCompilationSignature.from_context(context)
            cache_key = sig.build_cache_key(
                namespace=effective_options.cache_namespace,
                extension_context={
                    "operator_lowering_target": lowering_target.cache_signature,
                    "hilbert_identity": id(operator.hilbert),
                },
            )
            cached = self._store.get(cache_key)
            if cached is not None:
                debug_event(
                    "cache hit",
                    scope="cache",
                    tag="CACHE",
                    operator_name=ir.operator_name,
                    cache_key=cache_key,
                )
                return cached
            debug_event(
                "cache miss",
                scope="cache",
                tag="CACHE",
                operator_name=ir.operator_name,
                cache_key=cache_key,
            )
        else:
            cache_key = None
            debug_event(
                "cache disabled for compile invocation",
                scope="cache",
                tag="CACHE",
                operator_name=ir.operator_name,
            )

        # Post-cache passes
        try:
            self._pipeline.run_post_cache(context)
        except Exception as exc:
            raise SymbolicCompilerError(
                f"Post-cache pass failed for operator {ir.operator_name!r}: {exc}"
            ) from exc
        debug_event(
            "completed post-cache pass stage",
            scope="compile",
            tag="COMPILER",
            operator_name=ir.operator_name,
            pass_count=len(context.pass_reports),
        )

        # Resolve lowerer and lower
        try:
            lowerer = self._registry.resolve(context)
            debug_event(
                "resolved lowerer",
                scope="lowering",
                tag="LOWERING",
                operator_name=ir.operator_name,
                lowerer_name=lowerer.name,
                backend=lowerer.backend,
            )
            artifact = lowerer.lower(context)
        except Exception as exc:
            raise SymbolicCompilerError(
                f"Lowering failed for operator {ir.operator_name!r}: {exc}"
            ) from exc
        debug_event(
            "lowered symbolic operator",
            scope="lowering",
            tag="LOWERING",
            operator_name=ir.operator_name,
            selected_lowerer=context.selected_lowerer,
            backend=artifact.backend,
        )

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
            debug_event(
                "stored compiled artifact in cache",
                scope="cache",
                tag="CACHE",
                operator_name=ir.operator_name,
                cache_key=cache_key,
                cache_size=len(self._store),
            )
        else:
            debug_event(
                "completed compilation without cache store",
                scope="compile",
                tag="COMPILER",
                operator_name=ir.operator_name,
            )

        return artifact

    def compile_operator(
        self,
        operator: Any,
        *,
        options: SymbolicCompilerOptions | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Compiles a symbolic operator and returns the executable operator directly.

        Convenience wrapper around :meth:`compile` that unwraps the artifact.

        Args:
            operator: Symbolic operator with a ``to_ir()`` method.
            options: Override compiler options.
            metadata: Extra metadata for the context.

        Returns:
            Executable compiled operator instance.
        """
        debug_event(
            "compile_operator wrapper invoked",
            scope="compile",
            tag="COMPILER",
        )
        artifact = self.compile(operator, options=options, metadata=metadata)
        return artifact.compiled_operator

    def clear_cache(self) -> None:
        """Clears all entries from the artifact store."""
        self._store.clear()

    @property
    def options(self) -> SymbolicCompilerOptions:
        """Compiler defaults, with omitted lowering resolved on each invocation."""
        return self._options

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

    @property
    def operator_lowering_names(self) -> tuple[str, ...]:
        """Returns registered operator-lowering target names."""
        return self._operator_lowering_registry.target_names

    def __repr__(self) -> str:
        return (
            f"SymbolicCompiler("
            f"passes={self.pass_names!r}, "
            f"lowerers={self.lowerer_names!r}, "
            f"operator_lowerings={self.operator_lowering_names!r}, "
            f"cache_size={self.cache_size})"
        )


_DEFAULT_COMPILER: SymbolicCompiler | None = None


def compile_symbolic_operator(
    operator: Any,
    *,
    options: SymbolicCompilerOptions | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """
    Module-level convenience function for one-shot symbolic compilation.

    Uses the module-level shared :class:`SymbolicCompiler` instance (lazily
    created). The shared compiler reuses the global in-process artifact cache.

    Args:
        operator: Symbolic operator with a ``to_ir()`` method.
        options: Override compiler options.
        metadata: Extra metadata for the context.

    Returns:
        Executable compiled operator instance.

    Example::

        from neuralqx.operator.symbolic import compile_symbolic_operator

        compiled_op = compile_symbolic_operator(my_symbolic_op)
        xp, mels = compiled_op.get_conn_padded(x_batch)
    """
    global _DEFAULT_COMPILER
    if _DEFAULT_COMPILER is None:
        _DEFAULT_COMPILER = SymbolicCompiler()
        debug_event(
            "initialized default symbolic compiler",
            scope="compile",
            tag="COMPILER",
        )
    return _DEFAULT_COMPILER.compile_operator(
        operator, options=options, metadata=metadata
    )


def reset_default_symbolic_compiler() -> None:
    """
    Resets the module-level shared compiler instance.

    This helper primarily exists for tests that need deterministic singleton
    lifecycle behavior across cases.
    """
    global _DEFAULT_COMPILER
    _DEFAULT_COMPILER = None


__all__ = [
    "SymbolicCompiler",
    "compile_symbolic_operator",
    "reset_default_symbolic_compiler",
]
