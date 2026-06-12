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


from functools import total_ordering

from packaging.version import InvalidVersion
from packaging.version import Version as PackagingVersion

from .coercion import VersionInput
from .coercion import coerce_version_text


@total_ordering
class Version:
    """
    Strict semantic version object used by neuraLQX.

    This class is the canonical, strongly-typed representation of versions inside
    ``neuralqx.utils.module.version``. It wraps :class:`packaging.version.Version`
    and enforces full PEP 440 validation at construction time.

    This class does not perform lexical fallbacks, every comparison is semantic
    and every input must be parseable as a valid version.

    Accepted constructor inputs
        ``Version`` accepts version-like values via :class:`VersionInput`, including:
            - ``Version`` instances
            - string versions (for example ``"1.2.3"``, ``"1.2.3rc1"``, ``"1.2.3.post1"``)
            - integer versions (for example ``1`` -> ``"1"``)
            - component sequences (for example ``(1, 2, 3)`` -> ``"1.2.3"``)
            - ``packaging.version.Version`` instances

    Comparison semantics
        Ordering follows PEP 440 exactly. For the same base release ``X.Y.Z``:
            ``dev`` < ``a`` (alpha) < ``b`` (beta) < ``rc`` < final < ``post``.

    Examples:
        - ``Version("1.1.0.dev1") < Version("1.1.0.dev2") < Version("1.1.0")``
        - ``Version("1.1.0") < Version("1.1.0.post1")``
        - ``Version("1.1.0rc1") < Version("1.1.0")``

    Error behavior
        - Invalid version values raise :class:`packaging.version.InvalidVersion`.
        - Unsupported operand types in comparisons return ``NotImplemented`` so Python
           can attempt reflected operations.
        - Invalid but coercible comparison values (for example ``"parrot"``) raise
           :class:`packaging.version.InvalidVersion`.

    Exposed properties
        - ``source``: normalized textual source used to build the object.
        - ``canonical``: canonical PEP 440 form.
        - ``release``: release tuple from the parsed version.
        - ``major``, ``minor``, ``patch``: convenience release components.

    Notes
        - Use :class:`NeuralqxVersion` when you explicitly need string-subclass behavior.
        - Use :class:`Version` for strict policy checks, ordering, and dependency gating.
    """

    __slots__ = ("_source", "_parsed")

    def __init__(self, value: VersionInput = "0") -> None:
        source = coerce_version_text(value)
        self._source = source
        self._parsed = PackagingVersion(source)

    @property
    def source(self) -> str:
        """Original normalised textual source."""
        return self._source

    @property
    def canonical(self) -> str:
        """Canonical PEP 440 normalised string."""
        return str(self._parsed)

    @property
    def release(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self._parsed.release)

    @property
    def major(self) -> int:
        return self.as_tuple(width=1)[0]

    @property
    def minor(self) -> int:
        return self.as_tuple(width=2)[1]

    @property
    def patch(self) -> int:
        return self.as_tuple(width=3)[2]

    def as_tuple(self, *, width: int = 3, fill: int = 0) -> tuple[int, ...]:
        """
        Return release components padded/truncated to ``width``.
        """
        if width < 0:
            raise ValueError("width must be >= 0")

        rel = self.release
        if len(rel) >= width:
            return rel[:width]
        return rel + (fill,) * (width - len(rel))

    @classmethod
    def try_parse(cls, value: VersionInput) -> Version | None:
        try:
            return cls(value)
        except (TypeError, ValueError, InvalidVersion):
            return None

    def _coerce_other(self, other: object) -> Version | object:
        if isinstance(other, Version):
            return other

        try:
            return Version(other)  # type: ignore[arg-type]
        except TypeError:
            return NotImplemented

    def __eq__(self, other: object) -> bool:
        rhs = self._coerce_other(other)
        if rhs is NotImplemented:
            return NotImplemented  # type: ignore[return-value]
        return self._parsed == rhs._parsed

    def __lt__(self, other: object) -> bool:
        rhs = self._coerce_other(other)
        if rhs is NotImplemented:
            return NotImplemented  # type: ignore[return-value]
        return self._parsed < rhs._parsed

    def __hash__(self) -> int:
        return hash(self._parsed)

    def __repr__(self) -> str:
        return f"Version('{self.canonical}')"

    def __str__(self) -> str:
        return self.canonical


def parse_version(value: VersionInput) -> Version:
    """Parse a strict semantic version."""
    return Version(value)


def try_parse_version(value: VersionInput) -> Version | None:
    """Attempt to parse, returning ``None`` when parsing fails."""
    return Version.try_parse(value)


def coerce_version(value: VersionInput) -> Version:
    """Alias for :func:`parse_version`."""
    return parse_version(value)


def normalize_version(
    value: VersionInput,
    *,
    width: int = 3,
    strict: bool = True,
) -> tuple[int, ...]:
    """
    Convert a version-like value into a release tuple.

    If ``strict=False``, invalid values are normalised to zeros.
    """
    if strict:
        return parse_version(value).as_tuple(width=width)

    parsed = try_parse_version(value)
    if parsed is None:
        if width < 0:
            raise ValueError("width must be >= 0")
        return (0,) * width
    return parsed.as_tuple(width=width)
