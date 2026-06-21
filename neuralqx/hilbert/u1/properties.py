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


"""Read-only properties shared by U(1) Hilbert spaces.

The mixin exposes common scalar ranges, dimensions, and indexing capability for
both unconstrained and gauge-invariant homogeneous U(1) spaces.
"""

from __future__ import annotations

import numpy as np

from neuralqx.hilbert.local import AbstractLocalSpace
from neuralqx.hilbert.utils import INDEX_LIMIT


class U1PropertiesMixin:
    """Read-only metadata shared by public U(1) Hilbert spaces.

    The mixin assumes the concrete space has already built ``graph``,
    ``local_space``, ``layout``, ``_local_spaces``, ``_dimension``, and
    ``_dimension_pretty`` during initialization. It then exposes the common
    properties expected from ``U1Hilbert`` and ``U1GaugeInvariantHilbert``:
    flat state size, base edge count, charge range bounds, per-site local
    dimensions, dtype, exact dimension, and generic indexability.

    Users usually access these properties through a concrete U(1) space rather
    than through the mixin directly. Subclasses must preserve the expected
    attributes for the properties to remain valid.
    """

    @property
    def size(self) -> int:
        """Total number of flat scalar sites."""
        return self.layout.size

    @property
    def tiny_size(self) -> int:
        """Number of graph edges before gauge-component expansion."""
        return self.graph.n_edges

    @property
    def q_min(self) -> int | float:
        """Minimum charge value in the scalar local range."""
        return self.local_space.start

    @property
    def q_max(self) -> int | float:
        """Maximum charge value in the scalar local range."""
        return self.local_space.start + self.local_space.step * (
            self.local_space.size - 1
        )

    @property
    def q_step(self) -> int | float:
        """Charge spacing between adjacent local values."""
        return self.local_space.step

    @property
    def local_size(self) -> int:
        """Number of scalar charge values."""
        return self.local_space.local_size

    @property
    def local_sizes(self) -> tuple[int, ...]:
        """Local dimension at every flat scalar site."""
        return (self.local_size,) * self.size

    @property
    def local_spaces(self) -> tuple[AbstractLocalSpace, ...]:
        """Scalar local charge range for every flat scalar site."""
        return self._local_spaces

    @property
    def dtype(self) -> np.dtype:
        """Common dtype used for U(1) charge values."""
        return self.local_space.dtype

    @property
    def dimension(self) -> int:
        """Exact Hilbert-space dimension."""
        return self._dimension

    @property
    def dimensions_pretty(self) -> str:
        """Scientific-notation string for the exact dimension."""
        return self._dimension_pretty

    @property
    def is_indexable(self) -> bool:
        """Whether generic JAX int32 basis ranking is supported."""
        return not self.constrained and self.dimension <= INDEX_LIMIT
