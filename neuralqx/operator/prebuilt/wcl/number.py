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


"""U(1) number/charge operator."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.operator.prebuilt.utils import site_index
from neuralqx.utils.struct import field


class NumberOperator(ComputationalOperator):
    """The charge on one graph edge and one U(1) component.

    For edge $e$ and component $c$, this diagonal operator has eigenvalue
    $q_e^{(c)}$. The flat layout stores all edges of component zero first,
    then all edges of component one, and so on. An edge may be supplied as a
    graph edge or its base-edge index. No cutoff wrapping is involved in
    measuring the charge.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        edge (Any): Graph edge or base-edge index on which the operator acts.
        copy (int): Zero-based U(1) component in the component-major state layout.
    """

    edge: Any = field(static=True)
    """Graph edge or base-edge index on which the operator acts."""

    copy: int = field(static=True, default=0)
    """Zero-based U(1) component in the component-major state layout."""

    _site: int = field(static=True, init=False, default=0, repr=False)
    """Validated flat index used by the connectivity kernel."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        hilbert = require_u1_hilbert(self.hilbert, operator=type(self).__name__)
        object.__setattr__(self, "_site", site_index(hilbert, self.edge, self.copy))
        super().__post_init__()

    @property
    def max_conn_size(self) -> int:
        """Maximum number of padded connections emitted for one input configuration."""
        return 1

    @property
    def dtype(self) -> jnp.dtype:
        """Real or complex dtype used for the matrix elements."""
        return jnp.float64

    @property
    def is_hermitian(self) -> bool:
        """Whether this operator is known to equal its conjugate transpose."""
        return True

    @property
    def is_diagonal(self) -> bool:
        """Whether every nonzero matrix element connects a configuration to itself."""
        return True

    @property
    def adjoint(self) -> NumberOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return self

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        return f"NumberOperator(edge={self.edge!r}, copy={self.copy}, hilbert={self.hilbert!r})"

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        return state[None, :], jnp.asarray([state[self._site]], dtype=self.dtype)


__all__ = ["NumberOperator"]
