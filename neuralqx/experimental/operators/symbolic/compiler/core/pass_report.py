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


"""Compiler pass-report structures."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass(frozen=True, repr=False)
class SymbolicPassReport:
    """
    Execution report for one symbolic compiler pass.

    Attributes:
        pass_name: Stable pass identifier.
        duration_ms: Wall-clock duration in milliseconds.
        metadata: Optional pass metadata (frozen key-value tuple).
    """

    pass_name: str
    duration_ms: float
    metadata: tuple = dataclasses.field(default_factory=tuple)

    @classmethod
    def create(
        cls,
        *,
        pass_name: str,
        duration_ms: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SymbolicPassReport":
        """Builds a pass report from user-friendly values."""
        meta_tuple: tuple
        if metadata is None:
            meta_tuple = ()
        else:
            meta_tuple = tuple(sorted(metadata.items()))
        return cls(
            pass_name=str(pass_name),
            duration_ms=float(duration_ms),
            metadata=meta_tuple,
        )

    def metadata_map(self) -> dict[str, Any]:
        """Returns metadata as a mutable dictionary."""
        return dict(self.metadata)

    def __repr__(self) -> str:
        return (
            f"SymbolicPassReport("
            f"pass_name={self.pass_name!r}, "
            f"duration_ms={self.duration_ms:.3f}, "
            f"metadata_keys={tuple(k for k, _ in self.metadata)!r})"
        )


__all__ = ["SymbolicPassReport"]
