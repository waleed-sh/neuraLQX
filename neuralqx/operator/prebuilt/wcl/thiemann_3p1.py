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


"""3+1-dimensional WCL Thiemann-regularised vertex constraint."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.operator.prebuilt.utils import collect_thiemann_3p1_terms
from neuralqx.operator.prebuilt.utils import q_bounds
from neuralqx.operator.prebuilt.utils import require_gauge_dimensions
from neuralqx.operator.prebuilt.utils import require_nonplanar_graph
from neuralqx.operator.prebuilt.utils import require_u1_hilbert
from neuralqx.operator.prebuilt.utils import shift_sites
from neuralqx.utils.struct import field

from .volume_3p1 import volume_terms_3p1
from .volume_3p1 import volume_value_3p1


class Thiemann3p1VertexConstraintOperator(ComputationalOperator):
    r"""The spatially three-dimensional WCL Thiemann vertex constraint.

    For a segment $s$ and component $c$, let
    $D_{s,c}(q)=V_v(q)-V_v(q-d_s h e_{s,c})$, with cyclic charge arithmetic.
    Here $d_s$ is positive for a stored edge pointing away from the vertex
    and negative for an incoming edge. Write $U_{\alpha,c}|q\rangle
    =|q+h\ell_{\alpha,c}\rangle$ for the ket holonomy. The constraint is

    $$
    H_v=N_v\sum_{t,c}\frac{r_t\epsilon_t}{T_v L_t}
    (U_{\alpha_t,c}-U_{\alpha_t,c}^\dagger)D_{s_t,c}.
    $$

    The first two edges of each triple choose a loop that leaves the vertex
    along the first edge and returns along the second. $L_t$ counts those
    loops, $r_t$ accounts for reorienting the triple, and $\epsilon_t$ is its
    permutation parity. $T_v$ counts the positive-parity triples after
    identifying the first-two-edge swap, following the upstream convention.
    These combinatorial signs are distinct from the geometric signs in $V_v$.

    Connectivity returns $\langle q|H_v|q'\rangle$, as required by Monte Carlo
    estimators. Thus a row target $q'=q+b h\ell_{\alpha,c}$ has coefficient
    $-b N_v r_t\epsilon_t D_{s_t,c}(q')/(T_v L_t)$. The adjoint row instead
    has coefficient $b N_v r_t\epsilon_t D_{s_t,c}(q)/(T_v L_t)$.
    The default is the constraint above, and `adjoint` is its mathematical
    conjugate transpose. Upstream's ket-based forward/adjoint names were
    reversed when interpreted as rows and are not used here.

    Vertices with no admissible loop terms return one zero-amplitude padding
    slot. The operator is generally non-Hermitian.

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
    """Loop-and-segment records including vertex and loop normalization."""

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
        require_nonplanar_graph(hilbert.graph, model=type(self).__name__)
        q_min, _q_max, q_step, local_size = q_bounds(hilbert)
        object.__setattr__(
            self, "_terms", collect_thiemann_3p1_terms(hilbert, self.vertex)
        )
        object.__setattr__(
            self, "_volume_terms", volume_terms_3p1(hilbert, self.vertex)
        )
        object.__setattr__(self, "_q_min", q_min)
        object.__setattr__(self, "_q_step", q_step)
        object.__setattr__(self, "_local_size", local_size)
        super().__post_init__()

    @property
    def max_conn_size(self) -> int:
        """Maximum number of padded connections emitted for one input configuration."""
        return max(1, 2 * 3 * len(self._terms))

    @property
    def dtype(self) -> jnp.dtype:
        """Real or complex dtype used for the matrix elements."""
        return jnp.float64

    @property
    def is_hermitian(self) -> bool:
        """Whether this operator is known to equal its conjugate transpose."""
        return False

    @property
    def adjoint(self) -> Thiemann3p1VertexConstraintOperator:
        r"""Return the conjugate transpose, with entries $A^\dagger_{q,q'}=A^*_{q',q}$."""
        return Thiemann3p1VertexConstraintOperator(
            self.hilbert,
            self.vertex,
            adjoint_flag=not self.adjoint_flag,
            lapse=self.lapse,
        )

    def __repr__(self) -> str:
        """Return the operator type and its defining parameters."""
        side = "adjoint" if self.adjoint_flag else "forward"
        return (
            "Thiemann3p1VertexConstraintOperator("
            f"vertex={self.vertex!r}, n_terms={len(self._terms)}, side={side})"
        )

    def _shift_segment_inverse(
        self, state: jax.Array, site: int, direction: int
    ) -> jax.Array:
        """Apply the inverse of the segment holonomy oriented away from the vertex."""
        return shift_sites(
            state,
            (site,),
            (direction,),
            q_min=self._q_min,
            q_step=self._q_step,
            local_size=self._local_size,
        )

    def _volume(self, state: jax.Array) -> jax.Array:
        """Evaluate the local 3+1 vertex-volume eigenvalue."""
        return volume_value_3p1(state, self._volume_terms)

    def _loop_branch(
        self,
        state: jax.Array,
        sites: tuple[int, ...],
        directions: tuple[int, ...],
        branch_sign: int,
    ) -> jax.Array:
        """Apply the stored loop shifts or their inverses according to `branch_sign`."""
        return shift_sites(
            state,
            sites,
            directions,
            q_min=self._q_min,
            q_step=self._q_step,
            local_size=self._local_size,
            scale=branch_sign,
        )

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
        base_volume = self._volume(state)
        scale = jnp.asarray(self.lapse, dtype=self.dtype)
        for term in self._terms:
            coeff = jnp.asarray(term.coefficient, dtype=self.dtype)
            for copy in range(3):
                segment_site = term.segment_sites[copy]
                loop_sites = term.loop_sites_by_copy[copy]
                for branch_sign, mel_sign in ((+1, +1.0), (-1, -1.0)):
                    branch = self._loop_branch(
                        state, loop_sites, term.loop_directions, branch_sign
                    )
                    if not self.adjoint_flag:
                        # A row reverses the ket loop shift and uses its source coefficient.
                        segment_base = branch
                        delta_v = self._volume(segment_base) - self._volume(
                            self._shift_segment_inverse(
                                segment_base, segment_site, term.segment_shift
                            )
                        )
                        mel = -mel_sign * coeff * delta_v
                    else:
                        # Transposition puts the same diagonal difference on the row state.
                        delta_v = base_volume - self._volume(
                            self._shift_segment_inverse(
                                state, segment_site, term.segment_shift
                            )
                        )
                        mel = mel_sign * coeff * delta_v
                    x_primes.append(branch)
                    mels.append(scale * mel)
        return jnp.stack(x_primes, axis=0), jnp.stack(mels, axis=0)


__all__ = ["Thiemann3p1VertexConstraintOperator"]
