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

from functools import partial
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
from neuralqx.operator.prebuilt.utils.u1 import wrap_add_value
from neuralqx.utils.struct import field

from .volume_3p1 import volume_terms_3p1
from .volume_3p1 import volume_value_3p1


def _connectivity_tables(terms, volume_terms, n_sites):
    """Pack the geometry into arrays without expanding arithmetic in the JAX trace."""
    loops, segments, segment_ids, coefficients = [], [], [], []
    segment_lookup = {}
    for term in terms:
        for copy in range(3):
            loop = [0] * n_sites
            for site, direction in zip(
                term.loop_sites_by_copy[copy], term.loop_directions, strict=True
            ):
                # Repeated traversal accumulates the winding before cyclic wrapping.
                loop[site] += direction
            loops.append(loop)
            coefficients.append(term.coefficient)
            key = (term.segment_sites[copy], term.segment_shift)
            if key not in segment_lookup:
                segment_lookup[key] = len(segments)
                segment = [0] * n_sites
                segment[key[0]] = key[1]
                segments.append(segment)
            segment_ids.append(segment_lookup[key])
    volume_weights = {}
    for *sites, sign in volume_terms:
        # A determinant changes by the permutation parity. Combine equivalent
        # orderings while retaining their full weight in the volume sum.
        inversions = sum(sites[i] > sites[j] for i in range(3) for j in range(i + 1, 3))
        key = tuple(sorted(sites))
        volume_weights[key] = volume_weights.get(key, 0) + sign * (-1) ** inversions
    volume_weights = {
        sites: weight for sites, weight in volume_weights.items() if weight
    }
    return (
        jnp.asarray(loops, dtype=jnp.int32).reshape((-1, n_sites)),
        jnp.asarray(segments, dtype=jnp.int32).reshape((-1, n_sites)),
        jnp.asarray(segment_ids, dtype=jnp.int32),
        jnp.asarray(coefficients, dtype=jnp.float64),
        jnp.asarray(tuple(volume_weights), dtype=jnp.int32).reshape((-1, 3, 3)),
        jnp.asarray(tuple(volume_weights.values()), dtype=jnp.float64),
    )


def _volume_from_tables(states, sites, signs):
    """Evaluate the ordered triple sum for any leading configuration axes."""
    charges = jnp.asarray(states[..., sites], dtype=jnp.float64)
    triples = jnp.sum(
        charges[..., 0, :] * jnp.cross(charges[..., 1, :], charges[..., 2, :]), axis=-1
    )
    return jnp.sqrt(jnp.abs(jnp.sum(signs * triples, axis=-1)))


@partial(jax.jit, static_argnames=("adjoint",))
def _vertex_connections(states, tables, q_min, q_step, local_size, lapse, *, adjoint):
    """Share one executable across vertices with matching table shapes."""
    loops, segments, segment_ids, coefficients, volume_sites, volume_signs = tables

    def shifted(configurations, winding):
        values = wrap_add_value(
            configurations,
            winding * jnp.asarray(q_step, dtype=configurations.dtype),
            q_min=q_min,
            q_step=q_step,
            local_size=local_size,
        )
        return jnp.where(winding != 0, values, configurations)

    def volume(configurations):
        return _volume_from_tables(configurations, volume_sites, volume_signs)

    # Keep the public connection order: term, gauge copy, then + and - branches.
    branch_signs = jnp.asarray([1, -1], dtype=jnp.int32)
    branches = shifted(
        states[:, None, None, :],
        loops[None, :, None, :] * branch_signs[None, None, :, None],
    )
    if adjoint:
        # The adjoint's volume difference depends only on the source row and
        # segment, so each distinct segment is evaluated once, not per loop.
        delta = volume(states)[:, None] - volume(shifted(states[:, None, :], segments))
        values = delta[:, segment_ids, None] * branch_signs[None, None, :]
    else:
        # Forward rows evaluate the diagonal difference at the loop target.
        delta = volume(branches) - volume(
            shifted(branches, segments[segment_ids][None, :, None, :])
        )
        values = -delta * branch_signs[None, None, :]
    values = values * (lapse * coefficients)[None, :, None]
    return branches.reshape((states.shape[0], -1, states.shape[-1])), values.reshape(
        (states.shape[0], -1)
    )


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

    _kernel_tables: tuple = field(init=False, default=(), repr=False)
    """Geometry arrays passed as data to the shared connectivity executable."""

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
        object.__setattr__(
            self,
            "_kernel_tables",
            _connectivity_tables(self._terms, self._volume_terms, hilbert.size),
        )
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
        targets, values = self._get_conn_padded_batch_kernel(state[None, :])
        return targets[0], values[0]

    def _get_conn_padded_batch_kernel(self, states_2d):
        """Evaluate all rows using the same geometry tables."""
        if not self._terms:
            return states_2d[:, None, :], jnp.zeros(
                (states_2d.shape[0], 1), dtype=self.dtype
            )
        return _vertex_connections(
            states_2d,
            self._kernel_tables,
            self._q_min,
            self._q_step,
            self._local_size,
            self.lapse,
            adjoint=self.adjoint_flag,
        )

    def _get_conn_padded_direct(self, states_2d):
        """Keep vertex identity outside the shared kernel's compilation cache key."""
        return self._get_conn_padded_batch_kernel(states_2d)


__all__ = ["Thiemann3p1VertexConstraintOperator"]
