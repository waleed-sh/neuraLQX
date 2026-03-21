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


"""
String-like version wrapper.
"""

from __future__ import annotations

from .coercion import VersionInput
from .coercion import coerce_version_text

from .version import Version


class NeuralqxVersion(str):
    """
    A version string that behaves like ``str`` while comparing semantically.

    This class accepts comparison against:
    - ``Version``
    - ``NeuralqxVersion``
    - version strings
    - integer tuples/lists such as ``(1, 2, 3)``
    """

    __slots__ = ("_version",)

    def __new__(cls, value: VersionInput = "0") -> NeuralqxVersion:
        text = coerce_version_text(value)
        obj = super().__new__(cls, text)
        obj._version = Version(text)
        return obj

    @property
    def version(self) -> Version:
        return self._version

    @property
    def canonical(self) -> str:
        return self.version.canonical

    @property
    def release(self) -> tuple[int, ...]:
        return self.version.release

    @property
    def major(self) -> int:
        return self.version.major

    @property
    def minor(self) -> int:
        return self.version.minor

    @property
    def patch(self) -> int:
        return self.version.patch

    def as_tuple(self, *, width: int = 3, fill: int = 0) -> tuple[int, ...]:
        return self.version.as_tuple(width=width, fill=fill)

    def _convert_to_version(self, other: object) -> Version:
        if isinstance(other, NeuralqxVersion):
            return other.version
        if isinstance(other, Version):
            return other
        return Version(other)  # type: ignore[arg-type]

    def _cmp_wrapper(self, other: object, method: str) -> bool:
        rhs = self._convert_to_version(other)
        return bool(getattr(self.version, method)(rhs))

    def __lt__(self, other: object) -> bool:
        try:
            return self._cmp_wrapper(other, "__lt__")
        except TypeError:
            return NotImplemented  # type: ignore[return-value]

    def __le__(self, other: object) -> bool:
        try:
            return self._cmp_wrapper(other, "__le__")
        except TypeError:
            return NotImplemented  # type: ignore[return-value]

    def __gt__(self, other: object) -> bool:
        try:
            return self._cmp_wrapper(other, "__gt__")
        except TypeError:
            return NotImplemented  # type: ignore[return-value]

    def __ge__(self, other: object) -> bool:
        try:
            return self._cmp_wrapper(other, "__ge__")
        except TypeError:
            return NotImplemented  # type: ignore[return-value]

    def __eq__(self, other: object) -> bool:
        try:
            return self._cmp_wrapper(other, "__eq__")
        except TypeError:
            return NotImplemented  # type: ignore[return-value]

    def __ne__(self, other: object) -> bool:
        try:
            return self._cmp_wrapper(other, "__ne__")
        except TypeError:
            return NotImplemented  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"NeuralqxVersion('{str(self)}')"
