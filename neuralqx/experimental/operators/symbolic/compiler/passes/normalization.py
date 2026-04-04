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


"""Symbolic IR normalization compiler pass."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neuralqx.experimental.operators.symbolic.compiler.core.context import (
    SymbolicCompilationContext,
)
from neuralqx.experimental.operators.symbolic.compiler.passes.base import (
    AbstractSymbolicPass,
)


class SymbolicNormalizationPass(AbstractSymbolicPass):
    """
    Normalizes and fingerprints the symbolic operator IR.

    This pre-cache pass:

    1. Computes the static IR fingerprint (SHA-256 over the JSON-serialized IR)
       and stores it as analysis key ``"ir_fingerprint"``.
    2. Resolves the effective backend (``auto`` -> ``jax``) and stores it as
       analysis key ``"resolved_backend"``, also updating the context via
       :meth:`~neuralqx.experimental.operators.symbolic.compiler.core.context.SymbolicCompilationContext.set_selected_backend`.
    3. Stores a canonical term ordering (by term name, stable) as analysis key
       ``"term_order"`` - a tuple of term names in sorted order.

    Analysis keys written: ``"ir_fingerprint"``, ``"resolved_backend"``,
    ``"term_order"``
    """

    _BACKEND_RESOLUTION: dict[str, str] = {
        "auto": "jax",
        "jax": "jax",
    }

    @property
    def name(self) -> str:
        return "symbolic_normalization"

    def run(
        self,
        context: SymbolicCompilationContext,
    ) -> Mapping[str, Any] | None:
        # Static fingerprint
        fingerprint = context.ir.static_fingerprint()
        context.set_analysis("ir_fingerprint", fingerprint)

        # Backend resolution
        raw_backend = context.options.backend_preference
        resolved = self._BACKEND_RESOLUTION.get(raw_backend, raw_backend)
        context.set_analysis("resolved_backend", resolved)
        context.set_selected_backend(resolved)

        # Stable term order (sorted by name)
        term_order = tuple(
            t.name for t in sorted(context.ir.terms, key=lambda t: t.name)
        )
        context.set_analysis("term_order", term_order)

        return {
            "fingerprint_prefix": fingerprint[:16],
            "resolved_backend": resolved,
            "term_count": context.ir.term_count,
        }


__all__ = ["SymbolicNormalizationPass"]
