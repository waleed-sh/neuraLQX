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


"""Compilation-signature helpers for deterministic artifact caching."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from neuralqx.experimental.operators.symbolic.compiler.core.context import (
    SymbolicCompilationContext,
)


def _stable_hash(payload: Any) -> str:
    """Returns a stable SHA-256 hex digest over a JSON-serializable payload."""
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True, repr=False)
class SymbolicCompilationSignature:
    """
    Deterministic compilation signature for cache-key generation.

    Attributes:
        operator_ir_fingerprint: Stable digest of the operator IR.
        backend_target: Resolved backend name.
        hilbert_size: Hilbert space size.
        dtype_str: Matrix-element dtype string.
        options_signature: Static compiler-options signature.
    """

    operator_ir_fingerprint: str
    backend_target: str
    hilbert_size: int
    dtype_str: str
    options_signature: tuple = dataclasses.field(default_factory=tuple)

    @classmethod
    def from_context(
        cls,
        context: SymbolicCompilationContext,
    ) -> "SymbolicCompilationSignature":
        """Builds a signature from a post-backend-selection context."""
        selected_backend = (
            context.selected_backend
            if context.selected_backend is not None
            else context.options.backend_preference
        )
        return cls(
            operator_ir_fingerprint=context.ir.static_fingerprint(),
            backend_target=str(selected_backend),
            hilbert_size=context.ir.hilbert_size,
            dtype_str=context.ir.dtype_str,
            options_signature=context.options.static_signature(),
        )

    def build_cache_key(
        self,
        *,
        namespace: str,
        extension_context: Mapping[str, Any] | None = None,
    ) -> "SymbolicCacheKey":
        """
        Builds a deterministic cache key.

        Args:
            namespace: Cache namespace string.
            extension_context: Optional extra context included in the key.

        Returns:
            Immutable cache key.
        """
        payload: dict[str, Any] = {
            "namespace": namespace,
            "operator_ir_fingerprint": self.operator_ir_fingerprint,
            "backend_target": self.backend_target,
            "hilbert_size": self.hilbert_size,
            "dtype_str": self.dtype_str,
            "options_signature": list(self.options_signature),
        }
        if extension_context is not None:
            payload["extension_context"] = extension_context
        token = _stable_hash(payload)
        return SymbolicCacheKey(token=token, namespace=namespace)

    def as_dict(self) -> dict[str, Any]:
        """Returns dictionary-form signature payload."""
        return {
            "operator_ir_fingerprint": self.operator_ir_fingerprint,
            "backend_target": self.backend_target,
            "hilbert_size": self.hilbert_size,
            "dtype_str": self.dtype_str,
            "options_signature": list(self.options_signature),
        }

    def __repr__(self) -> str:
        return (
            f"SymbolicCompilationSignature("
            f"operator_ir_fingerprint={self.operator_ir_fingerprint[:12]!r}..., "
            f"backend_target={self.backend_target!r})"
        )


@dataclasses.dataclass(frozen=True)
class SymbolicCacheKey:
    """Immutable cache key for compiled symbolic operator artifacts."""

    token: str
    namespace: str

    def __str__(self) -> str:
        return self.token


__all__ = [
    "SymbolicCacheKey",
    "SymbolicCompilationSignature",
]
