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
Core Hilbert-space implementation.

This module defines :class:`~neuralqx.hilbert._abstract_hilbert_core.AbstractHilbertSpace`, the
implementation-level base class for Hilbert spaces in neuraLQX.

The core is responsible for:

- defining the local quantum-number domain (cutoff/step and the resulting discrete set),
- storing graph/layout metadata for flattened configurations with multiple gauge copies,
- constructing and exposing a NetKet Hilbert instance for sampling and state representation,
- providing indexing helpers between graph edges and flattened NetKet site indices,
- delegating state generation and proposal moves to the dispatched operations in ``.operations``.

The user-facing interface class wraps a concrete core instance and provides a stable public API.
"""

from __future__ import annotations

import abc
import logging
from typing import Literal
from typing import Optional
from typing import Union

import jax
import jax.numpy as jnp
import netket as nk
from netket.utils import StaticRange

from neuralqx.graph.core import AbstractGraph
from .utils.layout import StridedGaugeCopyLayout
from .utils.index import states_to_numbers as _states_to_numbers
from .utils.index import numbers_to_states as _numbers_to_states
from neuralqx.debug import event


class AbstractHilbertSpace(abc.ABC):
    """
    Core Hilbert-space base class.

    This class represents the implementation-level Hilbert space (as opposed to the user-facing
    interface). It owns:

    - the graph and local quantum-number domain (cutoff/step, allowed basis states),
    - metadata needed by move/state operations (dtype, q-range),
    - a NetKet Hilbert instance (``self._hilbert``),
    - the total dimension (or an estimate) of the represented space (``self._dimensions``).

    Subclasses are responsible for constructing ``self._hilbert`` and setting ``self._dimensions``.
    This base class provides:

    - consistent bookkeeping for local degrees of freedom,
    - a standard flattened layout for multi-copy gauge dimensions,
    - convenience helpers for mapping graph edges to flattened NetKet site indices,
    - thin method wrappers (``random_state``, ``flip_state``) that delegate to dispatch-based
      operations defined in ``.state_ops`` (allowing different cores to customize behavior).

    No public attributes are intended, access all state via properties.

    :param graph: Graph defining the degrees of freedom (edges) and mappings
        (edge_to_index/index_to_edge).
    :param cutoff: Cutoff controlling the local quantum number range
        (interpretation is model-specific).
    :param step: Step size between allowed local quantum numbers.
    :param gauge_dimensions: Number of gauge copies stored in block layout, must be >= 1.
    :param positive_qn: If True, restrict the local quantum numbers to non-negative values starting
        at ``qn_start`` (or 0 if ``qn_start`` is None).
    :param qn_start: Starting quantum number when ``positive_qn=True``. Defaults to 0 if omitted.

    :raises ValueError: If ``gauge_dimensions < 1``.
    """

    __slots__ = (
        "_graph",
        "_cutoff",
        "_step",
        "_gauge_dimensions",
        "_allowed_basis_states",
        "_dtype",
        "_q_min",
        "_q_max",
        "_q_step",
        "_hilbert",
        "_dimensions",
    )

    def __init__(
        self,
        graph: AbstractGraph,
        cutoff: Union[int, float],
        *,
        step: Union[int, float],
        gauge_dimensions: int,
        positive_qn: bool = False,
        qn_start: Optional[int] = None,
    ):
        if gauge_dimensions < 1:
            raise ValueError("gauge_dimensions must be >= 1.")

        self._graph = graph
        self._cutoff = cutoff
        self._step = step
        self._gauge_dimensions = int(gauge_dimensions)

        if isinstance(cutoff, float) or isinstance(step, float):
            self._dtype = jnp.float64
        else:
            self._dtype = jnp.int64

        # build local states
        self._allowed_basis_states = nk.utils.StaticRange(
            (-cutoff if not positive_qn else (qn_start if qn_start is not None else 0)),
            step,
            ((2 * cutoff) // step) + 1,
            dtype=self._dtype,
        )

        all_states = self._allowed_basis_states.all_states()
        self._q_min = float(all_states[0])
        self._q_max = float(all_states[-1])
        self._q_step = float(self._allowed_basis_states.step)

        # subclasses must build self._hilbert and self._dimensions
        self._hilbert = None  # type: ignore[assignment]
        self._dimensions = None  # type: ignore[assignment]

        event(
            msg="INITIALIZING HILBERT CORE",
            tag="HILBERT:CORE:INIT",
            level=logging.INFO,
        )

    #
    #
    #   Core metadata

    @property
    def graph(self) -> AbstractGraph:
        """
        Underlying graph defining the degrees of freedom and index mappings.

        The graph supplies a bijection between user-facing edge tokens and integer indices, which are
        then embedded into the flattened configuration layout.

        :return: The associated :class:`~neuralqx.graph.core.AbstractGraph`.
        """
        return self._graph

    @property
    def cutoff(self) -> Union[int, float]:
        """
        Cutoff controlling the local quantum-number range.

        The precise physical meaning is model-dependent, but the cutoff always participates in defining
        the discrete set of allowed local values (together with :attr:`step` and the ``positive_qn`` flag
        used during construction).

        :return: The cutoff value.
        """

        return self._cutoff

    @property
    def step(self) -> Union[int, float]:
        """
        Step size between allowed local quantum numbers.

        :return: The step size.
        """
        return self._step

    @property
    def gauge_dimensions(self) -> int:
        """
        Number of gauge copies stored in the flattened configuration.

        If each gauge copy contains :math:`E` edge sites, the flattened configuration length is

        .. math::

            N = E \\, G,

        where :math:`G =` ``gauge_dimensions``.

        :return: Number of gauge copies (``G``), always ``>= 1``.
        """
        return self._gauge_dimensions

    @property
    def dtype(self):
        """
        Dtype used to represent configurations.

        This is selected during construction from the types of ``cutoff`` and ``step`` (integer-like
        parameters lead to an integer dtype, otherwise a floating dtype).

        :return: A JAX dtype (for example ``jnp.int64`` or ``jnp.float64``).
        """
        return self._dtype

    @property
    def allowed_basis_states(self) -> StaticRange:
        """
        Allowed local basis values as a NetKet :class:`netket.utils.StaticRange`.

        This object compactly represents the discrete local domain (start, step, number of values) and
        can be used to obtain the explicit array of allowed local states.

        :return: A :class:`netket.utils.StaticRange` describing the local basis.
        """
        return self._allowed_basis_states

    @property
    def q_min(self) -> float:
        """
        Minimum allowed local quantum number.

        :return: The minimum local value as a float.
        """
        return self._q_min

    @property
    def q_max(self) -> float:
        """
        Maximum allowed local quantum number.

        :return: The maximum local value as a float.
        """
        return self._q_max

    @property
    def q_step(self) -> float:
        """
        Step size between allowed local quantum numbers (as a float).

        :return: The local step size.
        """
        return self._q_step

    @property
    def tiny_size(self) -> int:
        """
        Number of edge degrees of freedom in a single gauge copy.

        In the neuraLQX layout, a single gauge copy typically corresponds to the set of edges in the
        chosen edge representation (often the dual graph’s vertices).

        :return: Number of sites per gauge copy (``E``).
        """
        return int(self._graph.dual_nk_graph.n_nodes)

    @property
    def size(self) -> int:
        """
        Total number of sites in the flattened configuration.

        This is the length ``N`` of a single basis state ``sigma`` in NetKet layout. With
        :math:`E =` :attr:`tiny_size` and :math:`G =` :attr:`gauge_dimensions`:

        .. math::

            N = E \\, G.

        :return: Total configuration length (``N``).
        """
        return int(self.hilbert.size)

    @property
    def local_size(self) -> int:
        """
        Number of local basis states per site.

        :return: Local Hilbert dimension per site.
        """
        return int(self.hilbert.local_size)

    @property
    def dimensions(self) -> int:
        """
        Total Hilbert-space dimension.

        For indexable spaces this is exact; otherwise subclasses may provide an estimate appropriate for
        the represented configuration space.

        :return: The Hilbert-space dimension (or an estimate), as an integer.
        """

        return int(self._dimensions)

    @property
    def hilbert(self) -> nk.hilbert.AbstractHilbert:
        """
        NetKet Hilbert space instance representing the full space.

        Subclasses are responsible for constructing this object. It is the canonical Hilbert instance
        used for sampling and state representation within NetKet.

        :return: A :class:`netket.hilbert.AbstractHilbert` instance.
        """

        return self._hilbert

    @property
    def layout(self) -> StridedGaugeCopyLayout:
        """
        Gauge layout describing the block structure of the flattened configuration.

        The convention is a contiguous block layout:

        .. math::

            \\sigma = (\\sigma^{(0)}, \\sigma^{(1)}, \\ldots, \\sigma^{(G-1)}),

        where each block :math:`\\sigma^{(g)}` contains :math:`E` edge sites.

        :return: A :class:`~neuralqx.hilbert.utils.layout.StridedGaugeCopyLayout` instance.
        """

        return StridedGaugeCopyLayout(
            edges_per_copy=self.tiny_size, gauge_dimensions=self.gauge_dimensions
        )

    @property
    def tiny_hilbert(self) -> nk.hilbert.AbstractHilbert:
        """
        Single-copy NetKet Hilbert space.

        This returns a homogeneous Hilbert space of length :attr:`tiny_size` with local states given by
        :attr:`allowed_basis_states`. It is primarily intended for internal helpers and compatibility
        with code paths that operate on a single gauge copy.

        :return: A :class:`netket.hilbert.HomogeneousHilbert` instance.
        """

        return nk.hilbert.HomogeneousHilbert(
            local_states=self.allowed_basis_states,
            N=self.tiny_size,
        )

    #
    #
    #   Edge/site mapping helpers

    def edge_to_site(self, edge, gauge_copy: int = 0) -> int:
        """
        Map a graph edge token to a flattened site index.

        This first maps the edge token to an edge index using the graph, and then embeds that index into
        the chosen gauge-copy block.

        :param edge: Edge token accepted by ``graph.edge_to_index``.
        :param gauge_copy: Gauge-copy index ``g`` with ``0 <= g < gauge_dimensions``.
        :return: Flattened site index in ``[0, size)``.
        """

        edge_idx = self.graph.edge_to_index(edge)
        return self.layout.encode(gauge_copy=gauge_copy, edge_index=edge_idx)

    def site_to_edge(self, site: int):
        """
        Inverse map from a flattened site index to ``(gauge_copy, edge_token)``.

        :param site: Flattened site index in ``[0, size)``.
        :return: A pair ``(gauge_copy, edge_token)`` where ``edge_token`` is produced by
            ``graph.index_to_edge``.
        """

        _coord = self.layout.coord_of(site)
        gd, edge_idx = _coord.gauge_copy, _coord.edge_index
        return gd, self.graph.index_to_edge(edge_idx)

    def view(self, sigma: jax.Array) -> jax.Array:
        """
        Reshape a flattened configuration into an explicit gauge-block view.

        If ``sigma`` has shape ``(N,)``, this returns shape ``(G, E)``.
        If ``sigma`` has shape ``(B, N)``, this returns shape ``(B, G, E)``.

        :param sigma: A single configuration or batch in flattened layout.
        :return: A reshaped view with explicit gauge and edge axes.
        """

        E = self.tiny_size
        G = self.gauge_dimensions
        if sigma.ndim == 1:
            return sigma.reshape(G, E)
        return sigma.reshape(sigma.shape[0], G, E)

    def flatten(self, sigma_view: jax.Array) -> jax.Array:
        """
        Flatten a gauge-block view back to the standard NetKet layout.

        This is the inverse of :meth:`view`.

        :param sigma_view: Array of shape ``(G, E)`` or ``(B, G, E)``.
        :return: Flattened array of shape ``(N,)`` or ``(B, N)``.
        """

        if sigma_view.ndim == 2:
            return sigma_view.reshape(-1)
        return sigma_view.reshape(sigma_view.shape[0], -1)

    #
    #
    #   Public API (dispatch-backed)

    def random_state(self, key: jax.Array, size: int = 1) -> jax.Array:
        """
        Generate random basis state(s) using the dispatched random-state operation.

        Concrete core types may customise how random states are produced by providing specialised
        implementations in ``.operations.random``.

        :param key: JAX PRNGKey.
        :param size: Number of states to generate.
        :return: A state of shape ``(N,)`` or a batch of shape ``(B, N)``.
        """

        from .operations.random import random_state

        return random_state(self, key, size=size)

    def flip_state(
        self,
        sigma: jax.Array,
        key: jax.Array,
        number_of_edges: int = 1,
        *,
        adjacency: bool = False,
        scope: str = "single",
    ) -> jax.Array:
        """
        Propose a new configuration by updating one or more local degrees of freedom.

        This delegates to the dispatched flip operation in ``.operations.flip``. Constrained cores may
        specialise the dispatched implementation to ensure proposals remain within the allowed subspace.

        :param sigma: A state of shape ``(N,)`` or a batch of shape ``(B, N)``.
        :param key: JAX PRNGKey.
        :param number_of_edges: Number of sites/edges to modify (interpretation depends on the dispatched
            implementation and may depend on ``scope``).
        :param adjacency: If True, propose adjacent updates (for example :math:`q \\to q \\pm \\Delta q`);
            if False, propose values by resampling within the local domain.
        :param scope: Move-scope hint interpreted by the dispatched implementation.
        :return: Proposed state(s) with the same shape as ``sigma``.
        """

        from .operations.flip import flip_state

        return flip_state(
            self,
            sigma,
            key,
            number_of_edges=number_of_edges,
            adjacency=adjacency,
            scope=scope,
        )

    def states_to_numbers(
        self,
        states: jax.Array,
        *,
        backend: Literal["auto", "netket", "python"] = "auto",
        return_dtype: str = "auto",
        validate: bool = False,
    ):
        """
        Convert basis states to sequential numbers with NetKet-compatible ordering.

        :param states: A single state ``(N,)`` or a batch ``(B, N)``.
        :param backend: Conversion backend selector: ``"auto"``, ``"netket"``, or ``"python"``.
        :param return_dtype: Output dtype selector (implementation-defined; ``"auto"`` chooses a sensible
            default).
        :param validate: If True, validate that inputs are legal basis states before conversion.
        :return: Integer label(s) corresponding to the provided basis state(s).
        """

        return _states_to_numbers(
            self, states, backend=backend, return_dtype=return_dtype, validate=validate
        )

    def numbers_to_states(
        self,
        numbers,
        *,
        backend: Literal["auto", "netket", "python"] = "auto",
        validate: bool = False,
    ) -> jax.Array:
        """
        Convert sequential numbers to basis states with NetKet-compatible ordering.

        :param numbers: Integer label or array of labels.
        :param backend: Conversion backend selector: ``"auto"``, ``"netket"``, or ``"python"``.
        :param validate: If True, validate that inputs are within the representable range.
        :return: Basis state(s) corresponding to the provided label(s).
        """

        return _numbers_to_states(self, numbers, backend=backend, validate=validate)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"dim={self.dimensions}, size={self.size}, tiny_size={self.tiny_size}, "
            f"gauge_dims={self.gauge_dimensions})"
        )
