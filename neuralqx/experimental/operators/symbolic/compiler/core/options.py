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


"""Symbolic compiler option definitions."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

_VALID_BACKENDS: frozenset[str] = frozenset({"jax", "auto"})
_DEFAULT_CACHE_NAMESPACE: str = "nqx_symbolic_v1"


@dataclasses.dataclass(frozen=True, repr=False)
class SymbolicCompilerOptions:
    """
    Static and runtime controls for symbolic compiler execution.

    Attributes:
        backend_preference: Preferred lowering backend (currently only ``jax``
            is supported, ``auto`` resolves to ``jax``).
        enable_fusion: Whether fusion-planning passes are enabled.
        strict_validation: Whether validation passes fail hard on errors.
        cache_enabled: Whether compiled artifacts are cached in-process.
        cache_namespace: Namespace string used in cache-key generation.
        debug_flags: Optional debug / instrumentation flags.
    """

    backend_preference: str = "auto"
    enable_fusion: bool = True
    strict_validation: bool = True
    cache_enabled: bool = True
    cache_namespace: str = _DEFAULT_CACHE_NAMESPACE
    debug_flags: tuple = dataclasses.field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.backend_preference not in _VALID_BACKENDS:
            raise ValueError(
                f"Unsupported backend_preference: {self.backend_preference!r}. "
                f"Allowed: {sorted(_VALID_BACKENDS)}."
            )
        if not self.cache_namespace.strip():
            raise ValueError("cache_namespace must be a non-empty string.")

    @classmethod
    def from_mapping(
        cls,
        *,
        backend_preference: str = "auto",
        enable_fusion: bool = True,
        strict_validation: bool = True,
        cache_enabled: bool = True,
        cache_namespace: str = _DEFAULT_CACHE_NAMESPACE,
        debug_flags: Mapping[str, Any] | None = None,
    ) -> "SymbolicCompilerOptions":
        """Builds options from user-friendly keyword arguments."""
        flags: tuple
        if debug_flags is None:
            flags = ()
        else:
            flags = tuple(sorted(debug_flags.items()))
        return cls(
            backend_preference=backend_preference,
            enable_fusion=bool(enable_fusion),
            strict_validation=bool(strict_validation),
            cache_enabled=bool(cache_enabled),
            cache_namespace=cache_namespace,
            debug_flags=flags,
        )

    def debug_flag_map(self) -> dict[str, Any]:
        """Returns debug flags as a mutable dictionary."""
        return dict(self.debug_flags)

    def static_signature(self) -> tuple:
        """Returns a deterministic static signature for cache-key generation."""
        return (
            ("backend_preference", self.backend_preference),
            ("enable_fusion", int(self.enable_fusion)),
            ("strict_validation", int(self.strict_validation)),
            ("cache_enabled", int(self.cache_enabled)),
            ("cache_namespace", self.cache_namespace),
            ("debug_flags", self.debug_flags),
        )

    def __repr__(self) -> str:
        return (
            f"SymbolicCompilerOptions("
            f"backend_preference={self.backend_preference!r}, "
            f"strict_validation={self.strict_validation}, "
            f"cache_enabled={self.cache_enabled})"
        )


__all__ = ["SymbolicCompilerOptions"]
