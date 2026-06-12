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

from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from typing import overload
from typing import cast

from .dressing import DressingType
from .dressing import EdgeDressing
from .edge import Edge
from .vertex import Vertex


class Loop:
    """A closed ordered sequence of graph edges.

    ``Loop`` is the value object returned by graph cycle algorithms. It keeps
    the edge order explicit, validates that the path closes, and optionally
    carries one dressing entry per edge.
    """

    __slots__ = ("_edges", "_dressing")

    _edges: tuple[Edge, ...]
    _dressing: tuple[EdgeDressing, ...] | None

    def __init__(
        self,
        edges: Iterable[object],
        *,
        dressing: Iterable[object] | None = None,
    ) -> None:
        canonical_edges = tuple(Edge.from_like(edge) for edge in edges)
        self._validate_edges(canonical_edges)
        self._edges = canonical_edges
        self._dressing = None
        if dressing is not None:
            self.dressing = dressing

    @classmethod
    def from_like(cls, value: object) -> Loop:
        """Builds a canonical loop from a ``Loop`` or edge iterable."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return cls(value)
        raise TypeError(
            f"A loop must be a Loop or an iterable of edges; got {value!r}."
        )

    @classmethod
    def from_dressed(cls, dressed_edges: Iterable[object]) -> Loop:
        """Builds a loop from ``(edge, dressing)`` pairs."""
        pairs = tuple(dressed_edges)
        edges = []
        dressing = []
        for item in pairs:
            edge, edge_dressing = _coerce_dressed_edge(item)
            edges.append(edge)
            dressing.append(edge_dressing)
        return cls(edges, dressing=dressing)

    @staticmethod
    def _validate_edges(edges: tuple[Edge, ...]) -> None:
        if not edges:
            raise ValueError("A loop must contain at least one edge.")
        for index, (edge, next_edge) in enumerate(zip(edges, edges[1:])):
            if edge.end != next_edge.start:
                raise ValueError(
                    "Loop edges must be ordered head-to-tail; "
                    f"edge {index} ends at {edge.end!r}, but edge "
                    f"{index + 1} starts at {next_edge.start!r}."
                )
        if edges[-1].end != edges[0].start:
            raise ValueError(
                "Loop edges must close; the final edge ends at "
                f"{edges[-1].end!r}, but the first edge starts at {edges[0].start!r}."
            )

    @property
    def edges(self) -> tuple[Edge, ...]:
        """Ordered edges in the loop."""
        return self._edges

    @property
    def length(self) -> int:
        """Number of edges in the loop."""
        return len(self._edges)

    @property
    def n_edges(self) -> int:
        """Alias for :attr:`length`."""
        return self.length

    @property
    def vertices(self) -> tuple[Vertex, ...]:
        """Ordered vertices visited by the loop, including the closing vertex."""
        return tuple(edge.start for edge in self._edges) + (self._edges[-1].end,)

    @property
    def unique_vertices(self) -> tuple[Vertex, ...]:
        """Unique vertices in first-visit traversal order."""
        vertices = []
        seen: set[Vertex] = set()
        for vertex in self.vertices:
            if vertex in seen:
                continue
            vertices.append(vertex)
            seen.add(vertex)
        return tuple(vertices)

    @property
    def keys(self) -> tuple[int, ...]:
        """Parallel-edge keys of the loop edges."""
        return tuple(edge.key for edge in self._edges)

    @property
    def is_closed(self) -> bool:
        """Whether the edge sequence closes head-to-tail."""
        return self._edges[-1].end == self._edges[0].start

    @property
    def dressing(self) -> tuple[EdgeDressing, ...] | None:
        """Optional edge dressing metadata."""
        if self._dressing is None:
            return None
        return tuple(_copy_edge_dressing(item) for item in self._dressing)

    @dressing.setter
    def dressing(self, value: Iterable[object] | None) -> None:
        if value is None:
            self._dressing = None
            return
        self._dressing = self._coerce_dressing(value)

    @property
    def dressed_edges(self) -> tuple[tuple[Edge, EdgeDressing], ...] | None:
        """Edges paired with dressing metadata, if dressing is set."""
        if self._dressing is None:
            return None
        return tuple(
            (edge, _copy_edge_dressing(dressing))
            for edge, dressing in zip(self._edges, self._dressing)
        )

    def with_dressing(self, dressing: Iterable[object] | None) -> Loop:
        """Returns a copy of this loop with ``dressing`` attached."""
        return type(self)(self._edges, dressing=dressing)

    def without_dressing(self) -> Loop:
        """Returns a copy of this loop without dressing metadata."""
        return type(self)(self._edges)

    def copy(self) -> Loop:
        """Returns an independent copy of this loop."""
        return type(self)(self._edges, dressing=self._dressing)

    def as_tuple(self) -> tuple[Edge, ...]:
        """Returns the loop edges as a tuple."""
        return self._edges

    def as_list(self) -> list[Edge]:
        """Returns the loop edges as a mutable list."""
        return list(self._edges)

    def _coerce_dressing(
        self,
        value: Iterable[object],
    ) -> tuple[EdgeDressing, ...]:
        items = tuple(value)
        if len(items) != self.length:
            raise ValueError(
                f"Loop dressing has length {len(items)}, expected {self.length}."
            )

        out: list[EdgeDressing] = []
        for index, item in enumerate(items):
            if _is_dressed_edge(item):
                edge, dressing = _coerce_dressed_edge(item)
                if edge != self._edges[index]:
                    raise ValueError(
                        "Dressed edge does not match the loop edge at index "
                        f"{index}: expected {self._edges[index]!r}, got {edge!r}."
                    )
            else:
                dressing = _coerce_edge_dressing(item)

            if dressing["key"] != self._edges[index].key:
                raise ValueError(
                    "Dressing key does not match the loop edge key at index "
                    f"{index}: expected {self._edges[index].key}, "
                    f"got {dressing['key']}."
                )
            out.append(dressing)
        return tuple(out)

    def __repr__(self) -> str:
        dressing_state = "set" if self._dressing is not None else "unset"
        return (
            f"Loop(length={self.length}, "
            f"edges={list(self._edges)!r}, "
            f"dressing={dressing_state})"
        )

    __str__ = __repr__

    def __iter__(self) -> Iterator[Edge]:
        return iter(self._edges)

    def __len__(self) -> int:
        return self.length

    @overload
    def __getitem__(self, index: int) -> Edge: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Edge, ...]: ...

    def __getitem__(self, index: int | slice) -> Edge | tuple[Edge, ...]:
        return self._edges[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Loop):
            return NotImplemented
        return self._edges == other._edges and self._dressing == other._dressing


def _is_dressed_edge(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and isinstance(value[1], Mapping)
    )


def _coerce_dressed_edge(value: object) -> tuple[Edge, EdgeDressing]:
    if not _is_dressed_edge(value):
        raise TypeError(
            "Dressed edges must be ``(edge, dressing)`` pairs; " f"got {value!r}."
        )
    edge, dressing = cast(Sequence[object], value)
    return Edge.from_like(edge), _coerce_edge_dressing(dressing)


def _copy_edge_dressing(value: EdgeDressing) -> EdgeDressing:
    return {"type": value["type"], "key": value["key"]}


def _coerce_edge_dressing(value: object) -> EdgeDressing:
    if not isinstance(value, Mapping):
        raise TypeError(f"Edge dressing must be a mapping; got {value!r}.")

    edge_type = value.get("type")
    if edge_type not in ("creation", "annihilation"):
        raise ValueError(
            "Edge dressing field 'type' must be 'creation' or 'annihilation'; "
            f"got {edge_type!r}."
        )

    key = value.get("key")
    if isinstance(key, bool) or not isinstance(key, int):
        raise TypeError(f"Edge dressing field 'key' must be an integer; got {key!r}.")
    if key < 0:
        raise ValueError(f"Edge dressing field 'key' must be non-negative; got {key}.")

    return {"type": cast(DressingType, edge_type), "key": key}


__all__ = [
    "Loop",
]
