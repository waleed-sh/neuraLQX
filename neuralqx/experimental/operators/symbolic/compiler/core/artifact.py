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


"""Compiled symbolic operator artifact structures."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from neuralqx.experimental.operators.symbolic.compiler.core.pass_report import (
    SymbolicPassReport,
)
from neuralqx.experimental.operators.symbolic.compiler.core.signature import (
    SymbolicCacheKey,
)


@dataclasses.dataclass(frozen=True, repr=False)
class SymbolicCompiledArtifact:
    """
    Compilation artifact produced by the symbolic compiler pipeline.

    Attributes:
        operator_name: Source operator name.
        backend: Selected backend name.
        lowerer_name: Lowerer identifier used for code generation.
        compiled_operator: Executable compiled operator object.
        cache_key: Optional compilation cache key.
        pass_reports: Ordered tuple of pass-execution reports.
        metadata: Optional artifact metadata.
    """

    operator_name: str
    backend: str
    lowerer_name: str
    compiled_operator: Any  # ComputationalJaxOperator subclass
    cache_key: SymbolicCacheKey | None = None
    pass_reports: tuple = dataclasses.field(default_factory=tuple)
    metadata: tuple = dataclasses.field(default_factory=tuple)

    @classmethod
    def create(
        cls,
        *,
        operator_name: str,
        backend: str,
        lowerer_name: str,
        compiled_operator: Any,
        cache_key: SymbolicCacheKey | None = None,
        pass_reports: tuple[SymbolicPassReport, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "SymbolicCompiledArtifact":
        """Builds an artifact from user-friendly values."""
        meta_tuple: tuple
        if metadata is None:
            meta_tuple = ()
        else:
            meta_tuple = tuple(sorted(metadata.items()))
        return cls(
            operator_name=str(operator_name),
            backend=str(backend),
            lowerer_name=str(lowerer_name),
            compiled_operator=compiled_operator,
            cache_key=cache_key,
            pass_reports=pass_reports,
            metadata=meta_tuple,
        )

    def metadata_map(self) -> dict[str, Any]:
        """Returns metadata as a mutable dictionary."""
        return dict(self.metadata)

    def cache_token(self) -> str | None:
        """Returns cache token string when a cache key is present."""
        return str(self.cache_key) if self.cache_key is not None else None

    def __repr__(self) -> str:
        return (
            f"SymbolicCompiledArtifact("
            f"operator_name={self.operator_name!r}, "
            f"backend={self.backend!r}, "
            f"lowerer_name={self.lowerer_name!r}, "
            f"cache_token={self.cache_token()!r}, "
            f"pass_count={len(self.pass_reports)})"
        )


__all__ = ["SymbolicCompiledArtifact"]
