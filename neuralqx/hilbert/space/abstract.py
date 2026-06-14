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


"""Abstract Hilbert-space base interfaces.

The base classes in ``neuralqx.hilbert.space`` describe the minimum contract
shared by concrete Hilbert spaces. Specialized subclasses add discreteness,
graph layouts, indexing, sampling, and proposal semantics.
"""

from __future__ import annotations

import abc

from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import StructABCMeta


class AbstractHilbertSpace(Struct, metaclass=StructABCMeta):
    """Minimal public contract shared by all Hilbert spaces.

    A Hilbert space object describes a basis, whether that basis is finite, and
    whether the exact dimension is available cheaply. More specialized
    subclasses add local domains, graph layouts, indexing, random sampling, and
    proposal generation, but user code can rely on this base interface for the
    most general space-level metadata.

    Concrete subclasses must implement :attr:`is_finite` and :attr:`dimension`.
    A finite space may still return ``None`` for :attr:`dimension` when the
    exact constrained size would require expensive enumeration.
    """

    @property
    @abc.abstractmethod
    def is_finite(self) -> bool:
        """Whether the represented basis is finite."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int | None:
        """Exact basis dimension when it is cheaply known."""


__all__ = ["AbstractHilbertSpace"]
