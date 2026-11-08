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


"""2+1-dimensional WCL Thiemann-regularised vertex constraint."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.operator.prebuilt.utils import collect_thiemann_2p1_terms
from neuralqx.operator.prebuilt.utils import q_bounds
from neuralqx.operator.prebuilt.utils import require_gauge_dimensions
from neuralqx.operator.prebuilt.utils import require_planar_graph
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.operator.prebuilt.utils import shift_sites
from neuralqx.utils.struct import field

from .volume_2p1 import planar_volume_terms
from .volume_2p1 import sqrt_planar_volume_value


class Thiemann2p1VertexConstraintOperator(ComputationalOperator):
    r"""The planar WCL Thiemann constraint at one vertex.

    Each ordered pair of distinct minimal loops meeting at the vertex supplies
    a loop holonomy and two incident segments. For component $c$, define

    $$
    D_{s,c}(q)=\sqrt{V_v(q)}-\sqrt{V_v(S_{s,c}^{-1}q)}.
    $$

    Here the segment shift is oriented relative to the vertex and wraps at
    the charge cutoff. With the loop-pair permutation signs $w_t$ and ket
    holonomy $U_{\alpha,c}|q\rangle=|q+h\ell_{\alpha,c}\rangle$, the constraint is

    $$
    H_v=N_v\sum_{t,c}w_t U_{\alpha_t,c}D_{k_t,c}D_{l_t,c}.
    $$

    The volume uses geometric orientation signs. The loop-pair weights use
    permutation signs, following the upstream triangulation convention.
    Connectivity returns $\langle q|H_v|q'\rangle$. The default therefore
    reverses the ket loop shift and evaluates both differences at $q'$.
    The adjoint uses the positive loop shift and evaluates the differences
    at $q$. Segment directions are unchanged by taking the adjoint because
    the differences are real diagonal operators. This convention corrects
    the ket action returned as rows by the upstream forward implementation.

    The operator is generally non-Hermitian. A vertex with no loop pairs
    returns one zero-amplitude padding slot.

    Attributes:
        hilbert (AbstractHilbertSpace): Hilbert space defining the operator basis.
        vertex (Any): Graph vertex or its index in the graph vertex sequence.
        adjoint_flag (bool): Whether to evaluate the conjugate-transpose operator.
        lapse (float): Real lapse multiplying this vertex contribution.
    """

    vertex: Any = field(static=True)
    """Graph vertex or its index in the graph vertex sequence."""

    adjoint_flag: bool = field(static=True, default=False)
    """Whether to evaluate the conjugate-transpose operator."""

    lapse: float = field(static=True, default=1.0)
    """Real lapse multiplying this vertex contribution."""

    _terms: tuple = field(static=True, init=False, default=(), repr=False)
    """Ordered loop-pair records with segment shifts and permutation weights."""

    _volume_terms: tuple = field(static=True, init=False, default=(), repr=False)
    """Signed incident-edge data defining the local volume."""

    _q_min: int | float = field(static=True, init=False, default=0, repr=False)
    """Smallest allowed value on the local charge lattice."""

    _q_step: int | float = field(static=True, init=False, default=1, repr=False)
    """Spacing between adjacent allowed local charges."""

    _local_size: int = field(static=True, init=False, default=0, repr=False)
    """Number of allowed charges in one local degree of freedom."""

    def __post_init__(self) -> None:
        """Validate the construction arguments and prepare static kernel metadata."""
        hilbert = require_u1_hilbert(self.hilbert, operator=type(self).__name__)
        require_gauge_dimensions(hilbert, 3, operator=type(self).__name__)
        require_planar_graph(hilbert.graph, model=type(self).__name__)
        q_min, _q_max, q_step, local_size = q_bounds(hilbert)
        object.__setattr__(
            self, "_terms", collect_thiemann_2p1_terms(hilbert, self.vertex)
        )
        object.__setattr__(
            self, "_volume_terms", planar_volume_terms(hilbert, self.vertex)
        )
        object.__setattr__(self, "_q_min", q_min)
        object.__setattr__(self, "_q_step", q_step)
        object.__setattr__(self, "_local_size", local_size)
        super().__post_init__()

    @property
    def max_conn_size(self) -> int:
        """Maximum number of padded connections emitted for one input configuration."""
        return max(1, 3 * len(self._terms))

    @property
    def dtype(self) -> jnp.dtype:
        """Real or complex dtype used for the matrix elements."""
        return jnp.float64

    @property
    def is_hermitian(self) -> bool:
        """Whether this operator is known to equal its conjugate transpose."""
        return False

    @property
    def adjoint(self) -> Thiemann2p1VertexConstraintOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return Thiemann2p1VertexConstraintOperator(
            self.hilbert,
            self.vertex,
            adjoint_flag=not self.adjoint_flag,
            lapse=self.lapse,
        )

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        side = "adjoint" if self.adjoint_flag else "forward"
        return (
            "Thiemann2p1VertexConstraintOperator("
            f"vertex={self.vertex!r}, n_terms={len(self._terms)}, side={side})"
        )

    def _shift_one(self, state: jax.Array, site: int, direction: int) -> jax.Array:
        """Shift one flat component by the signed number of local lattice steps, cyclically."""
        return shift_sites(
            state,
            (site,),
            (direction,),
            q_min=self._q_min,
            q_step=self._q_step,
            local_size=self._local_size,
        )

    def _sqrt_volume(self, state: jax.Array) -> jax.Array:
        """Evaluate the square root of the planar vertex volume on one configuration."""
        return sqrt_planar_volume_value(state, self._volume_terms)

    def _get_conn_padded_kernel(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the connected row for one configuration.

        Args:
            state: Flat configuration with shape `(hilbert.size,)`.

        Returns:
            result (tuple[jax.Array, jax.Array]): Targets of shape `(max_conn_size, hilbert.size)` and their row matrix
                elements of shape `(max_conn_size,)`. Zero amplitudes mark padding.
        """
        if not self._terms:
            return state[None, :], jnp.zeros((1,), dtype=self.dtype)

        x_primes = []
        mels = []
        sign = +1 if self.adjoint_flag else -1
        for term in self._terms:
            for copy in range(3):
                loop_directions = tuple(
                    sign * int(direction) for direction in term.loop_directions
                )
                final = shift_sites(
                    state,
                    term.loop_sites_by_copy[copy],
                    loop_directions,
                    q_min=self._q_min,
                    q_step=self._q_step,
                    local_size=self._local_size,
                )
                base = state if self.adjoint_flag else final
                v0 = self._sqrt_volume(base)
                sig_k = self._shift_one(base, term.seg_k_sites[copy], term.seg_k_shift)
                sig_l = self._shift_one(base, term.seg_l_sites[copy], term.seg_l_shift)
                delta_k = v0 - self._sqrt_volume(sig_k)
                delta_l = v0 - self._sqrt_volume(sig_l)
                x_primes.append(final)
                mels.append(
                    jnp.asarray(self.lapse * term.weight, dtype=self.dtype)
                    * delta_k
                    * delta_l
                )
        return jnp.stack(x_primes, axis=0), jnp.stack(mels, axis=0)


__all__ = ["Thiemann2p1VertexConstraintOperator"]
