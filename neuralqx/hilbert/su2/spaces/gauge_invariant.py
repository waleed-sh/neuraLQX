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


"""Gauge-invariant SU(2) spin-network Hilbert space.

This module defines the spin-network Hilbert space used by the SU(2) API.
Flat states contain doubled-spin edge labels followed by left-comb recoupling
channels at graph vertices. The space can optionally enumerate the exact valid
basis for indexing and direct sampling.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

import jax
from neuralqx.graph import AbstractGraph
from neuralqx.hilbert.local import AbstractLocalSpace
from neuralqx.hilbert.local import LocalRange
from neuralqx.hilbert.space import DiscreteHilbertSpace
from neuralqx.hilbert.su2.constraints import SU2GaugeConstraint
from neuralqx.hilbert.su2.errors import InvalidSU2CutoffError
from neuralqx.hilbert.su2.errors import SU2ChannelConstructionFailureError
from neuralqx.hilbert.su2.errors import SU2UnconstrainedNotImplementedError
from neuralqx.hilbert.su2.errors import UnsupportedRecouplingSchemeError
from neuralqx.hilbert.su2.index import enumerate_exact_states
from neuralqx.hilbert.su2.layout import SpinNetworkLayout
from neuralqx.hilbert.su2.metadata import SU2GraphMetadata
from neuralqx.hilbert.su2.metadata import compile_su2_graph_metadata
from neuralqx.hilbert.su2.metadata import graph_loop_edge_indices
from neuralqx.hilbert.su2.recoupling import first_left_comb_channels
from neuralqx.hilbert.su2.recoupling import left_comb_valid_py
from neuralqx.hilbert.su2.sampling import SU2SamplingMixin
from neuralqx.hilbert.su2.state import SU2StateMixin
from neuralqx.hilbert.su2.utils import validate_two_j_max
from neuralqx.hilbert.u1.utils import scientific_int
from neuralqx.hilbert.utils import INDEX_LIMIT
from neuralqx.hilbert.utils import product
from neuralqx.utils.struct import field


class SU2GaugeInvariantHilbert(
    SU2StateMixin,
    SU2SamplingMixin,
    DiscreteHilbertSpace,
):
    """Gauge-invariant SU(2) spin-network Hilbert space.

    ``SU2GaugeInvariantHilbert`` represents the spin-network basis of an
    SU(2) lattice gauge theory on a graph. Edge sites store doubled-spin labels
    ``two_j`` with values from ``0`` through ``two_j_max``. Vertices with
    valence greater than three receive additional left-comb recoupling-channel
    sites, and a flat state is ordered as
    ``[edge two_j labels | vertex left-comb channels]``.

    Gauge invariance is enforced locally at every vertex by SU(2)
    admissibility. The edge labels incident on a vertex and the vertex's
    channel labels must form a valid recoupling tree. This is stronger than a
    simple rectangular local-domain check, so the class owns compiled graph
    metadata, a spin-network layout, a gauge constraint, optional exact-basis
    enumeration, and constructive state-completion helpers.

    When ``exact`` is true the valid basis is enumerated up to
    ``max_exact_states`` and can be used for direct sampling, ranking, and
    unranking. Without exact enumeration the dimension is reported as unknown,
    but the space still supports valid-state generation through a compiled
    Metropolis warmup. Wavefunction sampling uses ``MetropolisSU2`` and its
    shared proposal kernel with explicit Hastings corrections.

    This class is the gauge-invariant SU(2) Hilbert space. Passing
    ``is_gauge_invariant=False`` raises an explicit error because the
    unconstrained magnetic-index link basis is a different representation.
    Use SU2PeterWeylHilbert explicitly for that representation.

    Args:
        graph: Graph whose edges and vertices define the spin network.
        two_j_max: Maximum doubled-spin label.
        cutoff: Compatibility alias for ``two_j_max``.
        is_gauge_invariant: Whether to enforce SU(2) gauge invariance.
        recoupling: Recoupling scheme. Currently only ``"left_comb"`` is
            supported.
        exact: Whether to enumerate and cache the exact valid basis.
        max_exact_states: Safety limit for exact enumeration.
        random_init: Random-state initialization strategy.
        init_warmup: Optional number of attempted initialization updates per state.
        dtype: Dtype used for state values.
        dtype_like: Compatibility dtype override.

    Attributes:
        graph: Graph whose edges and vertices define the spin network.
        two_j_max: Maximum doubled-spin label.
        is_gauge_invariant: Whether SU(2) gauge invariance is enforced.
        recoupling: Recoupling scheme used for vertex channels.
        exact: Whether the exact valid basis is cached.
        max_exact_states: Safety limit for exact enumeration.
        random_init: Random-state initialization strategy.
        init_warmup: Optional number of attempted initialization updates per state.
        dtype_like: Dtype-like object used to resolve state dtype.
        metadata: Compiled graph incidence metadata.
        layout: Spin-network flat-state layout.
        constraint: SU(2) gauge constraint.
        _local_spaces: Local range for every flat site.
        _dtype: Common dtype used for state values.
        _dimension: Exact dimension when known.
        _dimension_pretty: Human-readable dimension string.
        _exact_states: Cached exact basis states when enumeration is enabled.
        _loop_edge_indices: Minimal graph loops represented by edge indices.
    """

    graph: AbstractGraph = field(pytree=False)
    """Graph whose edges and vertices define the spin network."""

    two_j_max: int = field(static=True)
    """Maximum doubled-spin label."""

    is_gauge_invariant: bool = field(static=True, default=True)
    """Whether SU(2) gauge invariance is enforced."""

    recoupling: str = field(static=True, default="left_comb")
    """Recoupling scheme used for vertex channels."""

    exact: bool = field(static=True, default=False)
    """Whether the exact valid basis is cached."""

    max_exact_states: int = field(static=True, default=100_000)
    """Safety limit for exact enumeration."""

    random_init: str = field(static=True, default="metropolis")
    """Random-state initialization strategy."""

    init_warmup: int | None = field(static=True, default=None)
    """Optional number of attempted initialization updates per state."""

    dtype_like: Any = field(static=True, default=jnp.int32)
    """Dtype-like object used to resolve state dtype."""

    metadata: SU2GraphMetadata = field(pytree=False, init=False, default=None)
    """Compiled graph incidence metadata."""

    layout: SpinNetworkLayout = field(pytree=False, init=False, default=None)
    """Spin-network flat-state layout."""

    constraint: SU2GaugeConstraint = field(static=True, init=False, default=None)
    """SU(2) gauge constraint."""

    _local_spaces: tuple[AbstractLocalSpace, ...] = field(
        static=True,
        init=False,
        default_factory=tuple,
        repr=False,
    )
    """Local range for every flat site."""

    _dtype: np.dtype = field(
        static=True, init=False, default=np.dtype(np.int32), repr=False
    )
    """Common dtype used for state values."""

    _dimension: int | None = field(static=True, init=False, default=None, repr=False)
    """Exact dimension when known."""

    _dimension_pretty: str = field(
        static=True, init=False, default="unknown", repr=False
    )
    """Readable dimension string."""

    _exact_states: np.ndarray | None = field(
        pytree=False,
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    """Cached exact basis states when enumeration is enabled."""

    _loop_edge_indices: tuple[tuple[int, ...], ...] = field(
        static=True,
        init=False,
        default_factory=tuple,
        repr=False,
    )
    """Minimal graph loops represented by edge indices."""

    def __init__(
        self,
        graph: AbstractGraph,
        two_j_max: int | None = None,
        *,
        cutoff: int | None = None,
        is_gauge_invariant: bool = True,
        recoupling: str = "left_comb",
        exact: bool = False,
        max_exact_states: int = 100_000,
        random_init: str = "metropolis",
        init_warmup: int | None = None,
        dtype: Any = jnp.int32,
        dtype_like: Any | None = None,
    ) -> None:
        """Initializes a gauge-invariant SU(2) spin-network Hilbert space.

        Args:
            graph: Graph whose edges and vertices define the spin network.
            two_j_max: Maximum doubled-spin edge label.
            cutoff: Compatibility alias for ``two_j_max``.
            is_gauge_invariant: Whether to construct the gauge-invariant
                spin-network basis.
            recoupling: Vertex recoupling scheme. Only ``"left_comb"`` is
                currently supported.
            exact: Whether to enumerate and cache the exact physical basis.
            max_exact_states: Guard for exact-basis enumeration.
            random_init: Valid-state initialization strategy.
            init_warmup: Optional number of attempted initialization updates per state.
            dtype: Dtype used for state values.
            dtype_like: Compatibility dtype override.

        Raises:
            InvalidSU2CutoffError: If no valid cutoff is supplied or the cutoff
            arguments conflict.
        """
        if two_j_max is None:
            if cutoff is None:
                raise InvalidSU2CutoffError(None)
            two_j_max = cutoff
        elif cutoff is not None and int(two_j_max) != int(cutoff):
            raise InvalidSU2CutoffError(
                f"conflicting two_j_max={two_j_max!r} and cutoff={cutoff!r}"
            )
        if dtype_like is not None:
            dtype = dtype_like
        self._init_fields(
            graph=graph,
            two_j_max=two_j_max,
            is_gauge_invariant=is_gauge_invariant,
            recoupling=recoupling,
            exact=exact,
            max_exact_states=max_exact_states,
            random_init=random_init,
            init_warmup=init_warmup,
            dtype_like=dtype,
        )

    def __post_init__(self) -> None:
        """Compiles metadata, layout, constraints, and optional exact states."""
        if not self.is_gauge_invariant:
            raise SU2UnconstrainedNotImplementedError()
        two_j_max = validate_two_j_max(self.two_j_max)
        if self.recoupling != "left_comb":
            raise UnsupportedRecouplingSchemeError(self.recoupling)
        if self.random_init not in ("metropolis", "vacuum"):
            raise ValueError("random_init must be 'metropolis' or 'vacuum'.")
        if self.init_warmup is not None and (
            isinstance(self.init_warmup, (bool, np.bool_))
            or not isinstance(self.init_warmup, (int, np.integer))
            or self.init_warmup < 0
        ):
            raise ValueError("init_warmup must be a nonnegative integer or None.")
        if self.max_exact_states < 1:
            raise ValueError("max_exact_states must be >= 1.")

        dtype = np.dtype(self.dtype_like)
        metadata = compile_su2_graph_metadata(self.graph)
        layout = SpinNetworkLayout(
            n_edges=metadata.n_edges,
            vertex_ids=metadata.vertex_ids,
            valence_by_vertex=metadata.valence_by_vertex,
            two_j_max=two_j_max,
        )
        local_spaces = tuple(
            LocalRange(0, 1, int(max_label) + 1, dtype=dtype)
            for max_label in layout.site_max_label_tuple
        )
        exact_states = (
            enumerate_exact_states(
                metadata=metadata,
                layout=layout,
                two_j_max=two_j_max,
                max_states=self.max_exact_states,
            )
            if self.exact
            else None
        )
        dimension = None if exact_states is None else int(exact_states.shape[0])
        constraint = SU2GaugeConstraint(
            incident_edges_by_vertex=metadata.incident_edges_by_vertex,
            vertex_channel_sites=layout.vertex_channel_sites,
            site_max_label=layout.site_max_label_tuple,
            dimension_value=dimension,
            recoupling=self.recoupling,
        )

        object.__setattr__(self, "two_j_max", two_j_max)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "_local_spaces", local_spaces)
        object.__setattr__(self, "_dtype", dtype)
        object.__setattr__(self, "_exact_states", exact_states)
        object.__setattr__(self, "_dimension", dimension)
        object.__setattr__(
            self,
            "_dimension_pretty",
            "unknown" if dimension is None else scientific_int(dimension),
        )
        object.__setattr__(
            self, "_loop_edge_indices", graph_loop_edge_indices(self.graph)
        )
        object.__setattr__(self, "constraint", constraint)
        constraint.validate_hilbert(self)

    def __hash__(self) -> int:
        """Returns a structural hash for the SU(2) Hilbert space."""
        return hash(
            (
                type(self),
                self.graph,
                self.two_j_max,
                self.is_gauge_invariant,
                self.recoupling,
                self.exact,
                self.max_exact_states,
                self.random_init,
                self.init_warmup,
                str(self.dtype),
            )
        )

    @property
    def graph_metadata(self) -> SU2GraphMetadata:
        """Compiled graph incidence metadata."""
        return self.metadata

    @property
    def n_edges(self) -> int:
        """Number of spin-network edge sites."""
        return self.layout.n_edges

    @property
    def n_vertices(self) -> int:
        """Number of graph vertices participating in channel layout."""
        return self.layout.n_vertices

    @property
    def n_channel_sites(self) -> int:
        """Number of vertex recoupling-channel sites."""
        return self.layout.n_channel_sites

    @property
    def size(self) -> int:
        """Total flat state size."""
        return self.layout.size

    @property
    def tiny_size(self) -> int:
        """Number of edge spin sites before channel expansion."""
        return self.n_edges

    @property
    def local_size(self) -> int:
        """Largest local dimension among edge and channel sites."""
        return max(self.local_sizes, default=0)

    @property
    def local_sizes(self) -> tuple[int, ...]:
        """Local dimension at every flat spin-network site."""
        return tuple(max_label + 1 for max_label in self.layout.site_max_label_tuple)

    @property
    def local_spaces(self) -> tuple[AbstractLocalSpace, ...]:
        """Local range for every flat site."""
        return self._local_spaces

    @property
    def dtype(self) -> np.dtype:
        """Common dtype used for state values."""
        return self._dtype

    @property
    def site_max_label(self) -> jax.Array:
        """Maximum allowed label at every flat site."""
        return self.layout.site_max_label

    @property
    def loop_edge_indices(self) -> tuple[tuple[int, ...], ...]:
        """Minimal graph loops represented by edge indices."""
        return self._loop_edge_indices

    @property
    def dimension(self) -> int | None:
        """Exact dimension when exact enumeration is available."""
        return self._dimension

    @property
    def dimensions_pretty(self) -> str:
        """Human-readable exact dimension or ``"unknown"``."""
        return self._dimension_pretty

    @property
    def rectangular_dimension(self) -> int:
        """Dimension of the unconstrained rectangular embedding."""
        return product(self.local_sizes)

    @property
    def is_indexable(self) -> bool:
        """Whether exact-basis ranking is available and int32-safe."""
        return self._dimension is not None and self._dimension <= INDEX_LIMIT

    def is_gauge_invariant_state(self, states: Any) -> jax.Array:
        """Checks the SU(2) gauge constraint using physics terminology.

        Args:
            states: Candidate flat spin-network state array.

        Returns:
            Boolean mask over the leading batch axes indicating which rows
            satisfy local bounds and SU(2) vertex admissibility.
        """
        return self.constraint(states)

    def check_states(self, states: Any) -> jax.Array:
        """Returns whether states are valid spin-network basis states.

        Args:
            states: Candidate flat spin-network state array.

        Returns:
            Boolean mask from :meth:`is_valid`, combining per-site local bounds
            with the SU(2) gauge constraint.
        """
        return self.is_valid(states)

    def all_states(self, *, max_states: int = 1_000_000) -> jax.Array:
        """Materializes exact valid spin-network basis states.

        Args:
            max_states: Maximum number of states to materialize.

        Returns:
            Exact valid basis as a JAX array ordered by the enumerator used by
            this Hilbert space.

        Raises:
            RuntimeError: If a cached exact basis exceeds ``max_states``. The
            same guard is also passed to on-demand enumeration when no cached
            basis exists.
        """
        if self._exact_states is not None:
            if self._exact_states.shape[0] > max_states:
                raise RuntimeError(
                    f"Refusing to materialize {self._exact_states.shape[0]} states, "
                    f"increase max_states={max_states} explicitly if intentional."
                )
            return jnp.asarray(self._exact_states, dtype=self.dtype)
        states = enumerate_exact_states(
            metadata=self.metadata,
            layout=self.layout,
            two_j_max=self.two_j_max,
            max_states=max_states,
        )
        return jnp.asarray(states, dtype=self.dtype)

    def states_to_numbers(self, states: Any, *, validate: bool = False) -> jax.Array:
        """Ranks states against the cached exact SU(2) basis.

        Args:
            states: Valid spin-network states to rank.
            validate: Whether to check membership before returning labels.

        Returns:
            Integer labels locating each state in the cached exact basis.

        Raises:
            RuntimeError: If the space was not constructed with ``exact=True``.
            ValueError: If validation is enabled and any row is invalid or not
            present in the cached basis.
        """
        if self._exact_states is None:
            raise RuntimeError("Constrained SU(2) ranking requires exact=True.")
        arr = jnp.asarray(states, dtype=jnp.int32)
        if validate and not bool(jnp.all(self.is_valid(arr))):
            raise ValueError("states contain values outside the SU(2) Hilbert space.")
        exact = jnp.asarray(self._exact_states, dtype=jnp.int32)
        single = arr.ndim == 1
        rows = arr[None, :] if single else arr.reshape((-1, self.size))
        matches = jnp.all(rows[:, None, :] == exact[None, :, :], axis=-1)
        found = jnp.any(matches, axis=-1)
        if validate and not bool(jnp.all(found)):
            raise ValueError(
                "states are valid-shaped but not present in exact SU(2) basis."
            )
        out = jnp.argmax(matches, axis=-1).astype(jnp.int32)
        return out[0] if single else out.reshape(arr.shape[:-1])

    def numbers_to_states(self, numbers: Any) -> jax.Array:
        """Unranks exact-basis labels into SU(2) spin-network states.

        Args:
            numbers: Integer labels into the cached exact basis.

        Returns:
            State array selected from the cached exact basis.

        Raises:
            RuntimeError: If the space was not constructed with ``exact=True``.
        """
        if self._exact_states is None:
            raise RuntimeError("Constrained SU(2) unranking requires exact=True.")
        nums = jnp.asarray(numbers, dtype=jnp.int32)
        exact = jnp.asarray(self._exact_states, dtype=self.dtype)
        return exact[nums]

    def states_to_numbers_python(self, states: Any, *, validate: bool = False) -> Any:
        """Ranks states using the exact-basis implementation.

        Args:
            states: Valid spin-network states to rank.
            validate: Whether to check membership before returning labels.

        Returns:
            Integer labels from :meth:`states_to_numbers`. The method exists to
            satisfy the discrete Hilbert Python-ranking API for exact SU(2)
            spaces.
        """
        return self.states_to_numbers(states, validate=validate)

    def numbers_to_states_python(self, numbers: Any) -> jax.Array:
        """Unranks exact-basis labels through the SU(2) exact basis.

        Args:
            numbers: Integer labels into the cached exact basis.

        Returns:
            State array selected from the cached exact basis.
        """
        return self.numbers_to_states(numbers)

    def complete_channels_for_edges(self, edge_values: Any) -> np.ndarray | None:
        """Builds a full state for a fixed edge-label sector.

        Args:
            edge_values: Doubled-spin labels for every graph edge.

        Returns:
            NumPy state array containing the supplied edge labels and the first
            admissible channel assignment at each vertex, or ``None`` when the
            edge sector cannot be completed.
        """
        edges = np.asarray(edge_values, dtype=np.int32).reshape(-1)
        if edges.shape[0] != self.n_edges:
            raise ValueError(
                f"Expected {self.n_edges} edge labels, got {edges.shape[0]}."
            )
        if not np.all((edges >= 0) & (edges <= self.two_j_max)):
            return None
        state = np.zeros(self.size, dtype=np.int32)
        state[: self.n_edges] = edges
        for vertex, edge_sites in enumerate(self.metadata.incident_edges_by_vertex):
            local_edges = tuple(int(edges[site]) for site in edge_sites)
            maxima = self.layout.vertex_channel_max_labels[vertex]
            channels = first_left_comb_channels(local_edges, maxima)
            if channels is None:
                return None
            for site, value in zip(
                self.layout.vertex_channel_sites[vertex], channels, strict=True
            ):
                state[site] = int(value)
        return state

    def explain_invalid_state(self, state: Any) -> dict[str, Any]:
        """Returns a diagnostic for the first invalid spin-network condition.

        Args:
            state: Candidate flat spin-network state.

        Returns:
            Dictionary describing whether the state is valid. Invalid results
            include a reason such as shape mismatch, site bounds, or vertex
            admissibility, plus the relevant site or vertex metadata.
        """
        arr = np.asarray(jnp.asarray(state), dtype=np.int32).reshape(-1)
        if arr.shape[0] != self.size:
            return {"valid": False, "reason": "shape", "shape": arr.shape}
        for site, (value, max_label) in enumerate(
            zip(arr, self.layout.site_max_label_tuple, strict=True)
        ):
            if value < 0 or value > max_label:
                return {
                    "valid": False,
                    "reason": "site_bounds",
                    "site": site,
                    "value": int(value),
                    "max_label": int(max_label),
                }
        for vertex, edge_sites in enumerate(self.metadata.incident_edges_by_vertex):
            edges = tuple(int(arr[site]) for site in edge_sites)
            channel_sites = self.layout.vertex_channel_sites[vertex]
            channels = tuple(int(arr[site]) for site in channel_sites)
            if not left_comb_valid_py(edges, channels):
                return {
                    "valid": False,
                    "reason": "vertex_admissibility",
                    "vertex": self.layout.vertex_ids[vertex],
                    "edge_spins": edges,
                    "channels": channels,
                }
        return {"valid": True}

    def require_completed_state(
        self,
        edge_values: Any,
        *,
        vertex_hint: Any | None = None,
    ) -> np.ndarray:
        """Returns a completed state or raises a physics-specific error.

        Args:
            edge_values: Doubled-spin labels for every graph edge.
            vertex_hint: Optional vertex identifier included in the error when
                completion fails.

        Returns:
            Full flat spin-network state with admissible vertex channels.

        Raises:
            SU2ChannelConstructionFailureError: If no admissible channel
            completion exists for the supplied edge sector.
        """
        state = self.complete_channels_for_edges(edge_values)
        if state is None:
            raise SU2ChannelConstructionFailureError(vertex_hint, edge_values)
        return state

    def site_info(self, site: int):
        """Returns layout metadata for one flat spin-network site.

        Args:
            site: Flat site index in the spin-network layout.

        Returns:
            :class:`SU2SiteInfo` describing whether the site is an edge or
            channel site, its maximum label, and its graph coordinate.
        """
        return self.layout.site_info(site)

    def edge_to_site(self, edge: Any) -> int:
        """Maps a graph edge reference to its flat spin site.

        Args:
            edge: Edge index or graph edge object accepted by the underlying
                graph.

        Returns:
            Flat site index containing that edge's doubled-spin label.
        """
        edge_index = (
            int(edge) if isinstance(edge, int) else self.graph.edge_to_index(edge)
        )
        return self.layout.edge_site(edge_index)

    def vertex_channel_site(self, vertex: Any, channel_index: int) -> int:
        """Maps a vertex channel reference to its flat site index.

        Args:
            vertex: Vertex id or layout position.
            channel_index: Zero-based left-comb channel index at the vertex.

        Returns:
            Flat site index containing the requested recoupling-channel label.
        """
        return self.layout.channel_site(vertex, channel_index)

    def model_input_metadata(self) -> dict[str, Any]:
        """Returns static metadata useful for model input encodings.

        Returns:
            Dictionary containing group name, basis name, cutoff, recoupling
            scheme, layout sizes, per-site maximum labels, vertex channel
            counts, and graph incidence data. The dictionary is intended for
            neural ansatz builders that need to interpret flat SU(2) states.
        """
        return {
            "group": "SU(2)",
            "basis": "gauge_invariant_spin_network",
            "two_j_max": self.two_j_max,
            "recoupling": self.recoupling,
            "n_edges": self.n_edges,
            "n_vertices": self.n_vertices,
            "n_channel_sites": self.n_channel_sites,
            "site_max_label": self.layout.site_max_label_tuple,
            "vertex_num_channels": self.layout.vertex_num_channels,
            "incident_edges": self.metadata.incident_edges_by_vertex,
            "incident_orientations": self.metadata.incident_orient_by_vertex,
        }

    def __repr__(self) -> str:
        """Returns a compact representation of SU(2) Hilbert metadata.

        Returns:
            String containing cutoff, graph size, flat state size, channel
            count, recoupling scheme, exact-enumeration flag, and dimension
            summary for interactive debugging.
        """
        return (
            "SU2GaugeInvariantHilbert("
            f"two_j_max={self.two_j_max}, "
            f"n_edges={self.n_edges}, "
            f"n_vertices={self.n_vertices}, "
            f"size={self.size}, "
            f"n_channel_sites={self.n_channel_sites}, "
            f"recoupling={self.recoupling!r}, "
            f"exact={self.exact}, "
            f"dimension={self.dimensions_pretty})"
        )


SU2Hilbert = SU2GaugeInvariantHilbert
SpinNetworkHilbert = SU2GaugeInvariantHilbert


__all__ = [
    "SU2GaugeInvariantHilbert",
    "SU2Hilbert",
    "SpinNetworkHilbert",
]
