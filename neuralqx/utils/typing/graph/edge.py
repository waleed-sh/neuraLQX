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


from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from typing import overload

from .vertex import Vertex


@dataclass(frozen=True, slots=True, init=False)
class Edge:
    """A graph edge with explicit starting and ending vertices.

    Attributes:
        start: Starting vertex of the edge.
        end: Ending vertex of the edge.
        key: Non-negative integer distinguishing parallel edges.
    """

    start: Vertex
    end: Vertex
    key: int

    def __init__(
        self,
        start: object,
        end: object,
        key: int = 0,
    ) -> None:
        object.__setattr__(self, "start", Vertex.from_like(start))
        object.__setattr__(self, "end", Vertex.from_like(end))
        object.__setattr__(self, "key", self._coerce_key(key))

    @classmethod
    def from_like(
        cls,
        value: object,
        *,
        key: int | None = None,
    ) -> Edge:
        """Builds a canonical edge from an ``Edge`` or edge tuple."""
        if isinstance(value, cls):
            if key is None or key == value.key:
                return value
            return cls(value.start, value.end, key)

        if not _is_edge_sequence(value):
            raise TypeError(
                "An edge must be an Edge, ``(start, end)``, or "
                f"``(start, end, key)``; got {value!r}."
            )

        if len(value) == 2:
            start, end = value
            return cls(start, end, 0 if key is None else key)

        start, end, explicit_key = value
        return cls(start, end, explicit_key if key is None else key)

    @staticmethod
    def _coerce_key(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Edge key must be an integer; got {value!r}.")
        if value < 0:
            raise ValueError(f"Edge key must be non-negative; got {value}.")
        return value

    @property
    def vertices(self) -> tuple[Vertex, Vertex]:
        """Ordered ``(start, end)`` vertices of the edge."""
        return (self.start, self.end)

    def as_tuple(self) -> tuple[Vertex, Vertex, int]:
        """Returns the edge as ``(start, end, key)``."""
        return (self.start, self.end, self.key)

    def __repr__(self) -> str:
        return (
            "Edge(" f"start={self.start!r}, " f"end={self.end!r}, " f"key={self.key})"
        )

    __str__ = __repr__

    def __iter__(self) -> Iterator[Vertex | int]:
        yield self.start
        yield self.end
        yield self.key

    def __len__(self) -> int:
        return 3

    @overload
    def __getitem__(self, index: Literal[0]) -> Vertex: ...

    @overload
    def __getitem__(self, index: Literal[1]) -> Vertex: ...

    @overload
    def __getitem__(self, index: Literal[2]) -> int: ...

    def __getitem__(self, index: int) -> Vertex | int:
        return self.as_tuple()[index]


def _is_edge_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) in (2, 3)
    )


__all__ = [
    "Edge",
]
