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


"""Symbolic compiler pass-pipeline abstraction."""

from __future__ import annotations

from collections.abc import Sequence

from neuralqx.experimental.operators.symbolic.compiler.core.context import (
    SymbolicCompilationContext,
)
from neuralqx.experimental.operators.symbolic.compiler.passes.base import (
    AbstractSymbolicPass,
)


class SymbolicPassPipeline:
    """
    Staged symbolic compiler-pass pipeline.

    The pipeline is split into two explicit stages:

    - **pre-cache passes**: run before the artifact-cache lookup, light-weight
      validation, backend selection, and normalization.
    - **post-cache passes**: run only on cache misses, heavier analysis and
      fusion planning.

    Args:
        pre_cache_passes: Passes executed before cache lookup.
        post_cache_passes: Passes executed after a cache miss.
    """

    __slots__ = ("_post_cache_passes", "_pre_cache_passes")

    def __init__(
        self,
        *,
        pre_cache_passes: Sequence[AbstractSymbolicPass],
        post_cache_passes: Sequence[AbstractSymbolicPass],
    ) -> None:
        if not pre_cache_passes:
            raise ValueError(
                "SymbolicPassPipeline requires at least one pre-cache pass."
            )
        self._pre_cache_passes = tuple(pre_cache_passes)
        self._post_cache_passes = tuple(post_cache_passes)

    @property
    def pre_cache_passes(self) -> tuple[AbstractSymbolicPass, ...]:
        """Returns pre-cache pass sequence."""
        return self._pre_cache_passes

    @property
    def post_cache_passes(self) -> tuple[AbstractSymbolicPass, ...]:
        """Returns post-cache pass sequence."""
        return self._post_cache_passes

    def run_pre_cache(self, context: SymbolicCompilationContext) -> None:
        """Runs all pre-cache passes in declaration order."""
        for pass_ in self._pre_cache_passes:
            pass_.execute(context)

    def run_post_cache(self, context: SymbolicCompilationContext) -> None:
        """Runs all post-cache passes in declaration order."""
        for pass_ in self._post_cache_passes:
            pass_.execute(context)

    def pass_names(self) -> tuple[str, ...]:
        """Returns pass names in end-to-end execution order."""
        return tuple(
            p.name for p in (*self._pre_cache_passes, *self._post_cache_passes)
        )

    def __repr__(self) -> str:
        return (
            f"SymbolicPassPipeline("
            f"pre_cache={tuple(p.name for p in self._pre_cache_passes)!r}, "
            f"post_cache={tuple(p.name for p in self._post_cache_passes)!r})"
        )


__all__ = ["SymbolicPassPipeline"]
