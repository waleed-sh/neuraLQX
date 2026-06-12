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
from numbers import Real
from typing import Literal
from typing import overload


@dataclass(frozen=True, slots=True)
class Vertex:
    """A vertex represented by Cartesian coordinates ``(x, y, z)``."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", self._coerce_coordinate(self.x, name="x"))
        object.__setattr__(self, "y", self._coerce_coordinate(self.y, name="y"))
        object.__setattr__(self, "z", self._coerce_coordinate(self.z, name="z"))

    @classmethod
    def from_like(cls, value: object) -> Vertex:
        """Builds a canonical vertex from a ``Vertex`` or coordinate triple."""
        if isinstance(value, cls):
            return value
        if not _is_coordinate_triple(value):
            raise TypeError(
                "A vertex must be a Vertex or a coordinate triple ``(x, y, z)``; "
                f"got {value!r}."
            )
        return cls(value[0], value[1], value[2])

    @staticmethod
    def _coerce_coordinate(value: object, *, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"Vertex coordinate {name!r} must be a real number; got {value!r}."
            )
        return float(value)

    def as_tuple(self) -> tuple[float, float, float]:
        """Returns the vertex as ``(x, y, z)``."""
        return (self.x, self.y, self.z)

    def __repr__(self) -> str:
        return (
            "Vertex("
            f"x={_format_coordinate(self.x)}, "
            f"y={_format_coordinate(self.y)}, "
            f"z={_format_coordinate(self.z)})"
        )

    __str__ = __repr__

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def __len__(self) -> int:
        return 3

    @overload
    def __getitem__(self, index: Literal[0]) -> float: ...

    @overload
    def __getitem__(self, index: Literal[1]) -> float: ...

    @overload
    def __getitem__(self, index: Literal[2]) -> float: ...

    def __getitem__(self, index: int) -> float:
        return self.as_tuple()[index]


def _is_coordinate_triple(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
    )


def _format_coordinate(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return format(value, ".12g")


__all__ = [
    "Vertex",
]
