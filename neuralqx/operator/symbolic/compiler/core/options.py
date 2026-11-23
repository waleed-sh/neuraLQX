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


"""Symbolic compiler option definitions."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from neuralqx.utils.errors import NeuralqxValueError

_VALID_BACKENDS: frozenset[str] = frozenset({"jax", "auto"})
_DEFAULT_CACHE_NAMESPACE: str = "nqx_symbolic_v2"


@dataclasses.dataclass(frozen=True, repr=False)
class SymbolicCompilerOptions:
    """
    Static and runtime controls for symbolic compiler execution.

    Attributes:
        backend_preference: Preferred lowering backend (currently only ``jax``
            is supported, ``auto`` resolves to ``jax``).
        deduplicate_connected_components: Whether lowered connectivity kernels
            should merge duplicate connected states and drop zero matrix elements.
            Static assembly analysis derives a reduced unique-target output bound
            and limits runtime checks to unresolved collisions.
        optimize_kernels: Whether to plan structured scalar DAGs and direct target
            construction. Disable to use the reference mutation-based lowerer.
        enable_fusion: Whether compatible scalar expressions are batched during
            kernel planning, and legacy fusion groups are recorded.
        strict_validation: Whether validation passes fail hard on errors.
        cache_enabled: Whether compiled artifacts are cached in-process.
        cache_namespace: Namespace string used in cache-key generation.
        operator_lowering: Registry key selecting the compiled-operator target.
            ``None`` inherits the compiler or registry default on each compilation.
        diagnostics_enabled: Whether DSL diagnostics pass is enabled.
        diagnostics_min_severity: Minimum severity shown/enforced by diagnostics
            (``"info"``, ``"warning"``, ``"error"``).
        fail_on_warnings: Whether warnings should fail compilation.
        max_diagnostics: Maximum number of diagnostics retained per compile.
        lint_state_sample_size: Number of source states sampled by dynamic
            connectivity diagnostics.
        lint_branch_sample_cap: Maximum sampled branch evaluations for dynamic
            connectivity diagnostics.
        lint_max_exact_hilbert_states: Maximum Hilbert cardinality for exact
            support-membership checks during diagnostics.
        debug_flags: Optional debug / instrumentation flags.
    """

    backend_preference: str = "auto"
    deduplicate_connected_components: bool = True
    optimize_kernels: bool = dataclasses.field(default=True, kw_only=True)
    enable_fusion: bool = True
    strict_validation: bool = True
    cache_enabled: bool = True
    cache_namespace: str = _DEFAULT_CACHE_NAMESPACE
    operator_lowering: str | None = None
    diagnostics_enabled: bool = True
    diagnostics_min_severity: str = "warning"
    fail_on_warnings: bool = False
    max_diagnostics: int = 200
    lint_state_sample_size: int = 32
    lint_branch_sample_cap: int = 4096
    lint_max_exact_hilbert_states: int = 8192
    debug_flags: tuple = dataclasses.field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.backend_preference not in _VALID_BACKENDS:
            raise NeuralqxValueError(
                f"Unsupported backend_preference: {self.backend_preference!r}. "
                f"Allowed: {sorted(_VALID_BACKENDS)}."
            )
        if not self.cache_namespace.strip():
            raise NeuralqxValueError("cache_namespace must be a non-empty string.")
        if self.operator_lowering is not None and (
            not isinstance(self.operator_lowering, str)
            or not self.operator_lowering.strip()
        ):
            raise NeuralqxValueError("operator_lowering must be a non-empty string.")
        normalized_severity = str(self.diagnostics_min_severity).strip().lower()
        if normalized_severity not in {"info", "warning", "error"}:
            raise NeuralqxValueError(
                f"Unsupported diagnostics_min_severity: {self.diagnostics_min_severity!r}. "
                "Allowed: ['error', 'info', 'warning']."
            )
        object.__setattr__(self, "diagnostics_min_severity", normalized_severity)
        if int(self.max_diagnostics) <= 0:
            raise NeuralqxValueError("max_diagnostics must be a positive integer.")
        if int(self.lint_state_sample_size) <= 0:
            raise NeuralqxValueError(
                "lint_state_sample_size must be a positive integer."
            )
        if int(self.lint_branch_sample_cap) <= 0:
            raise NeuralqxValueError(
                "lint_branch_sample_cap must be a positive integer."
            )
        if int(self.lint_max_exact_hilbert_states) <= 0:
            raise NeuralqxValueError(
                "lint_max_exact_hilbert_states must be a positive integer."
            )

    @classmethod
    def from_mapping(
        cls,
        *,
        backend_preference: str = "auto",
        deduplicate_connected_components: bool = True,
        optimize_kernels: bool = True,
        enable_fusion: bool = True,
        strict_validation: bool = True,
        cache_enabled: bool = True,
        cache_namespace: str = _DEFAULT_CACHE_NAMESPACE,
        operator_lowering: str | None = None,
        diagnostics_enabled: bool = True,
        diagnostics_min_severity: str = "warning",
        fail_on_warnings: bool = False,
        max_diagnostics: int = 200,
        lint_state_sample_size: int = 32,
        lint_branch_sample_cap: int = 4096,
        lint_max_exact_hilbert_states: int = 8192,
        debug_flags: Mapping[str, Any] | None = None,
    ) -> SymbolicCompilerOptions:
        """Builds options from user-friendly keyword arguments."""
        flags: tuple
        if debug_flags is None:
            flags = ()
        else:
            flags = tuple(sorted(debug_flags.items()))
        return cls(
            backend_preference=backend_preference,
            deduplicate_connected_components=bool(deduplicate_connected_components),
            optimize_kernels=bool(optimize_kernels),
            enable_fusion=bool(enable_fusion),
            strict_validation=bool(strict_validation),
            cache_enabled=bool(cache_enabled),
            cache_namespace=cache_namespace,
            operator_lowering=operator_lowering,
            diagnostics_enabled=bool(diagnostics_enabled),
            diagnostics_min_severity=diagnostics_min_severity,
            fail_on_warnings=bool(fail_on_warnings),
            max_diagnostics=int(max_diagnostics),
            lint_state_sample_size=int(lint_state_sample_size),
            lint_branch_sample_cap=int(lint_branch_sample_cap),
            lint_max_exact_hilbert_states=int(lint_max_exact_hilbert_states),
            debug_flags=flags,
        )

    def debug_flag_map(self) -> dict[str, Any]:
        """Returns debug flags as a mutable dictionary."""
        return dict(self.debug_flags)

    def static_signature(self) -> tuple:
        """Returns a deterministic static signature for cache-key generation."""
        return (
            ("backend_preference", self.backend_preference),
            (
                "deduplicate_connected_components",
                int(self.deduplicate_connected_components),
            ),
            ("optimize_kernels", int(self.optimize_kernels)),
            ("enable_fusion", int(self.enable_fusion)),
            ("strict_validation", int(self.strict_validation)),
            ("cache_enabled", int(self.cache_enabled)),
            ("cache_namespace", self.cache_namespace),
            ("operator_lowering", self.operator_lowering),
            ("diagnostics_enabled", int(self.diagnostics_enabled)),
            ("diagnostics_min_severity", self.diagnostics_min_severity),
            ("fail_on_warnings", int(self.fail_on_warnings)),
            ("max_diagnostics", int(self.max_diagnostics)),
            ("lint_state_sample_size", int(self.lint_state_sample_size)),
            ("lint_branch_sample_cap", int(self.lint_branch_sample_cap)),
            ("lint_max_exact_hilbert_states", int(self.lint_max_exact_hilbert_states)),
            ("debug_flags", self.debug_flags),
        )

    def __repr__(self) -> str:
        return (
            f"SymbolicCompilerOptions("
            f"backend_preference={self.backend_preference!r}, "
            f"deduplicate_connected_components={self.deduplicate_connected_components}, "
            f"operator_lowering={self.operator_lowering!r}, "
            f"strict_validation={self.strict_validation}, "
            f"diagnostics_enabled={self.diagnostics_enabled}, "
            f"cache_enabled={self.cache_enabled})"
        )


__all__ = ["SymbolicCompilerOptions"]
