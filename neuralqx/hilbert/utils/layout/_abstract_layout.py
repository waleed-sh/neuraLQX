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
Layout abstractions and concrete gauge-copy strided layout utilities.

This module defines a small but extensible abstraction for flattening layouts used by
Hilbert-space related logic in neuralqx.

Many parts of the codebase work with a flattened 1D representation of configurations.
At the same time, the physical or logical meaning of a site often lives in a richer
coordinate system, such as

- (gauge_copy, edge_index) for replicated U(1)-style gauge copies, or
- future SU(2)-specific coordinates that may include additional labels.

A layout provides the bijection between these two views.

This module provides an abstract base class that defines the contract shared by all layouts.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from typing import Generic
from typing import TypeVar
from typing import final

# TODO: move to types
CoordT = TypeVar("CoordT")


class AbstractBasisLayout(ABC, Generic[CoordT]):
    r"""
    Abstract base class for flattening layouts.

    A layout is a bijection between

    - a structured coordinate in some domain-specific coordinate space `C`, and
    - a flattened integer site index in `{0, 1, ..., N-1}`.

    In symbols, a subclass defines an invertible map

    .. math::

        \phi: C \to \{0, \ldots, N-1\}

    together with its inverse

    .. math::

        \phi^{-1}: \{0, \ldots, N-1\} \to C.

    This abstraction is meant to intentionally separates basis semantics from memory layout.

    A large part of neuralqx logic operates on flattened arrays because they are efficient
    and easy to vectorize. At the same time, many algorithms reason in structured terms such as

    - gauge copy indices,
    - edge indices,
    - irrep labels,
    - magnetic quantum number labels,
    - intertwiner labels,
    - sectors or channels.

    Hard-coding the flattening convention in many places makes the code brittle and makes
    it difficult to support new groups or basis parameterizations.

    By centralizing the flattening logic in a `BasisLayout`, code that manipulates flattened
    states can remain generic while group-specific logic is expressed through structured
    coordinates.

    Every subclass must implement the following abstract members:

    - :attr:`size`
      Returns the total number of flattened sites `N`.

    - :meth:`site_of`
      Maps a structured coordinate to a flattened site index.

    - :meth:`coord_of`
      Inverse mapping from flattened site index to structured coordinate.

    Required invariants
    ---------------------
    For every valid coordinate `c` and every valid site `s`, subclasses must satisfy

    - `0 <= site_of(c) < size`
    - `coord_of(site_of(c)) == c`
    - `site_of(coord_of(s)) == s`

    Error semantics
    -----------------
    Subclasses should raise:

    - :class:`IndexError` when a flattened site index is out of range, and
    - :class:`IndexError` when coordinate components are out of range.

    Use :class:`ValueError` only for invalid layout construction parameters, such as
    non-positive dimensions.

    Convenience methods
    --------------------
    This class provides a few helpers:

    - :meth:`validate_site` for consistent flat-index validation
    - :meth:`encode_coord` and :meth:`decode_coord` as naming aliases
    - :meth:`__len__` so `len(layout)` returns `size`
    - :meth:`contains_site` to test flat-site validity

    Notes on API naming
    --------------------
    The canonical abstract methods are `site_of(coord)` and `coord_of(site)` because they
    are explicit and coordinate-type agnostic.

    Concrete subclasses may additionally expose convenience methods such as

    - `encode(gauge_copy, edge_index) -> int`
    - `decode(site) -> tuple[...]`

    when that improves ergonomics for common call sites.
    """

    @property
    @abstractmethod
    def size(self) -> int:
        """
        Total number of flattened sites.

        :return: Integer `N > 0` such that valid flattened site indices are in
            `[0, N)`.
        """

    @abstractmethod
    def site_of(self, coord: CoordT) -> int:
        """
        Encode a structured coordinate into a flattened site index.

        :param coord: A valid coordinate in the subclass-defined coordinate space.
        :return: Flattened site index in `[0, size)`.
        :raises IndexError: If any component of `coord` is out of range.
        """

    @abstractmethod
    def coord_of(self, site: int) -> CoordT:
        """
        Decode a flattened site index into a structured coordinate.

        This is the inverse of :meth:`site_of`.

        :param site: Flattened site index.
        :return: Subclass-defined structured coordinate.
        :raises IndexError: If `site` is out of range.
        """

    @final
    def encode_coord(self, coord: CoordT) -> int:
        """
        Alias for :meth:`site_of`.

        This name can be useful in code where "encode/decode" terminology is more natural.
        """
        return self.site_of(coord)

    @final
    def decode_coord(self, site: int) -> CoordT:
        """
        Alias for :meth:`coord_of`.

        This name can be useful in code where "encode/decode" terminology is more natural.
        """
        return self.coord_of(site)

    @final
    def validate_site(self, site: int) -> None:
        """
        Validate that a flattened site index lies in `[0, size)`.

        :param site: Flattened site index.
        :raises TypeError: If `site` is not an integer.
        :raises IndexError: If `site` is outside the valid range.
        """
        if not isinstance(site, int):
            raise TypeError(f"site must be an int, got {type(site).__name__}.")
        if site < 0 or site >= self.size:
            raise IndexError(f"site={site} out of range [0,{self.size}).")

    @staticmethod
    @final
    def _validate_int(name: str, value: int) -> None:
        """
        Validate that a value is an integer.

        Subclasses can use this helper for coordinate component validation.

        :param name: Human-readable parameter name for error messages.
        :param value: Value to validate.
        :raises TypeError: If `value` is not an integer.
        """
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}.")

    @staticmethod
    @final
    def _validate_positive_int(name: str, value: int) -> None:
        """
        Validate that a layout parameter is a strictly positive integer.

        This helper is intended for constructor-time validation of shape parameters.

        :param name: Human-readable parameter name for error messages.
        :param value: Value to validate.
        :raises TypeError: If `value` is not an integer.
        :raises ValueError: If `value <= 0`.
        """
        AbstractBasisLayout._validate_int(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}.")

    @final
    def contains_site(self, site: int) -> bool:
        """
        Return whether `site` is a valid flattened site index.

        :param site: Candidate flat site index.
        :return: `True` if `site` is an integer in `[0, size)`, else `False`.
        """
        return isinstance(site, int) and (0 <= site < self.size)

    @final
    def __len__(self) -> int:
        """
        Return the total number of flattened sites.

        Equivalent to :attr:`size`.
        """
        return self.size


__all__ = [
    "AbstractBasisLayout",
]
