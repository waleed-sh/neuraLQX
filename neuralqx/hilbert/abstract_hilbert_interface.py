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
User-facing Hilbert-space interface.

This module defines :class:`~neuralqx.hilbert.abstract_hilbert_interface.AbstractHilbertInterface`,
a thin wrapper around a concrete core implementation
(:class:`~neuralqx.hilbert._abstract_hilbert_core.AbstractHilbertSpace`).

The interface exists to provide a stable public API (validation, configuration, convenience
methods) while allowing the internal core to evolve independently. Most numerical logic and all
backend-specific details (NetKet Hilbert construction, constraints, move proposals, indexing) live
in the core and are exposed here via delegation.
"""

import abc
import logging

from typing import Any
from typing import Generic
from typing import Literal, Union
from typing import TypeVar
from humanize import scientific

import jax
import jax.numpy as jnp

import netket as nk

from .abstract_hilbert_core import AbstractHilbertSpace

from neuralqx.graph.core import AbstractGraph
from neuralqx.debug import event

CoreT = TypeVar("CoreT", bound=AbstractHilbertSpace)


class AbstractHilbertInterface(abc.ABC, Generic[CoreT]):
    """
    User-facing interface around a concrete Hilbert-space core implementation.

    This interface is intentionally thin: it owns a core object (an
    :class:`neuralqx.hilbert._abstract_hilbert_core.AbstractHilbertSpace`) and exposes a stable,
    user-friendly API while allowing the internal core to evolve.

    The key design goal is separation of concerns:

    - The interface handles user-facing validation, configuration, and a clean attribute surface.
    - The core implements the actual physics/constraints, NetKet Hilbert construction, moves, etc.

    Core construction:
    Subclasses must implement :meth:`_build_core` and return a fully-initialised core instance.
    All properties and convenience methods delegate to that core.

    :param graph: The graph on which the Hilbert space is defined.
    :param cutoff: Cutoff used by the underlying core (semantics depend on the concrete core).
    :param step: Step size between allowed local quantum numbers (semantics depend on the core).
    :param gauge_dimensions: Number of gauge copies (blocks) stored in the flattened configuration.
    :param is_gauge_invariant: If True, construct a gauge-invariant (constrained) Hilbert space.
    :param kwargs: Forwarded to :meth:`_build_core` for backend-specific options.
        Concrete interfaces define and validate any group-specific constructor arguments.
    """

    _core: CoreT
    """Concrete Hilbert core implementation wrapped by this interface."""

    _is_gauge_invariant: bool
    """Whether this interface represents a gauge-invariant Hilbert space."""

    def __init__(
        self,
        graph: AbstractGraph,
        cutoff: Union[int, float] = jnp.inf,
        *,
        step: Union[int, float] = 1,
        gauge_dimensions: int = 1,
        is_gauge_invariant: bool = False,
        **kwargs,
    ):

        event(
            msg="INITIALIZING HILBERT INTERFACE",
            tag="HILBERT:INIT",
            level=logging.INFO,
            cutoff=cutoff,
            step=step,
            gauge_dimensions=gauge_dimensions,
            is_gauge_invariant=is_gauge_invariant,
        )

        self._is_gauge_invariant = bool(is_gauge_invariant)
        self._core = self._build_core(
            graph=graph,
            cutoff=cutoff,
            step=step,
            gauge_dimensions=gauge_dimensions,
            is_gauge_invariant=is_gauge_invariant,
            **kwargs,
        )

    @abc.abstractmethod
    def _build_core(self, **kwargs) -> CoreT:
        """
        Build and return the concrete Hilbert-space core.

        Subclasses must implement this to select and configure the appropriate core (for example,
        unconstrained vs constrained, or different gauge groups). The returned object must be a fully
        initialised instance of
        :class:`~neuralqx.hilbert._abstract_hilbert_core.AbstractHilbertSpace`.

        :param kwargs: Constructor arguments forwarded from :meth:`__init__`.
        :return: A ready-to-use core Hilbert-space implementation.
        """

    @property
    def hilbert(self) -> CoreT:
        """
        Underlying core Hilbert-space implementation.

        This object owns the NetKet Hilbert instance, constraint logic, move proposals, and all indexing
        utilities. The interface delegates most functionality to this core.

        :return: The concrete core instance.
        """

        return self._core

    @property
    def hilbert_netket(self) -> nk.hilbert.AbstractHilbert | nk.hilbert.DiscreteHilbert:
        """
        NetKet Hilbert object used for sampling and operator construction.

        :return: A NetKet Hilbert instance.
        """

        return self.hilbert.hilbert

    @property
    def is_gauge_invariant(self) -> bool:
        """
        Whether this interface represents a gauge-invariant (constrained) Hilbert space.

        :return: True if the Hilbert space is constrained by gauge invariance, otherwise False.
        """

        return self._is_gauge_invariant

    @property
    def graph(self) -> AbstractGraph:
        """
        Graph on which the Hilbert space is defined.

        This is delegated to the underlying core and is guaranteed to be an
        :class:`~neuralqx.graph.core.AbstractGraph`.

        :return: The associated graph.
        """
        return self.hilbert.graph

    @property
    def tiny_hilbert(self) -> nk.hilbert.AbstractHilbert:
        """
        A smaller/base NetKet Hilbert object used internally by some cores.

        Typically this corresponds to a single "gauge copy" (one block of edge degrees of freedom) and is
        used to build or reason about the full Hilbert space.

        :return: A NetKet Hilbert instance.
        """

        return self.hilbert.tiny_hilbert

    @property
    def tiny_size(self) -> int:
        """
        Number of sites (edge degrees of freedom) in a single gauge copy.

        If the configuration is stored as ``gauge_dimensions`` consecutive blocks, then

        .. math::

            \\text{size} = \\text{gauge\\_dimensions} \\times \\text{tiny\\_size}.

        :return: Number of sites in one gauge copy.
        """
        return self.hilbert.tiny_size

    @property
    def size(self) -> int:
        """
        Total number of sites in the flattened configuration.

        This is the length ``N`` of a single basis state ``sigma`` (shape ``(N,)``), including all gauge
        copies.

        :return: Total configuration length.
        """
        return self.hilbert.size

    @property
    def local_size(self) -> int:
        """
        Number of local basis states per site.

        :return: Local Hilbert dimension per site.
        """
        return self.hilbert.local_size

    @property
    def dimensions(self) -> int:
        """
        Total Hilbert-space dimension (or an estimate if the space is not indexable).

        :return: The dimension (exact for finite indexable spaces, otherwise core-defined).
        """
        return self.hilbert.dimensions

    @property
    def gauge_dimensions(self) -> int:
        """
        Number of gauge copies (blocks) stored in the flattened configuration.

        :return: Gauge dimension as defined by the core.
        """
        return self.hilbert.gauge_dimensions

    @property
    def is_indexable(self) -> bool:
        """
        Whether the underlying NetKet Hilbert space supports indexing.

        :return: True if indexable, otherwise False.
        """
        return self.hilbert_netket.is_indexable

    @property
    def is_finite(self) -> bool:
        """
        Whether the underlying NetKet Hilbert space is finite.

        :return: True if finite, otherwise False.
        """
        return self.hilbert_netket.is_finite

    def random_state(self, key: jax.Array, size: int = 1) -> jax.Array:
        """
        Generate random basis state(s) via the core implementation.

        :param key: JAX PRNGKey.
        :param size: Number of states to generate (batch size).

        :return: A single state of shape ``(N,)`` or a batch of states of shape ``(B, N)``,
            depending on the core and ``size``.
        """

        return self.hilbert.random_state(key, size=size)

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
        Propose a new configuration by modifying one or more local degrees of freedom.

        This is a user-facing wrapper over the core’s move proposal logic. The precise semantics of
        ``number_of_edges``, ``adjacency``, and ``scope`` are core-dependent (and may differ between
        constrained and unconstrained implementations).

        :param sigma: A state of shape ``(N,)`` or a batch of states of shape ``(B, N)``.
        :param key: JAX PRNGKey.
        :param number_of_edges: Number of sites/edges to modify (core-dependent).
        :param adjacency: If True, propose adjacent updates (for example, :math:`q \\to q \\pm \\Delta`)
            rather than resampling.
        :param scope: Optional selector used by some constrained cores (for example, ``"single"`` vs
            ``"all"``). Ignored by cores that do not support it.

        :return: A proposed state (or batch) with the same shape as ``sigma``.
        """

        return self.hilbert.flip_state(
            sigma,
            key,
            number_of_edges=number_of_edges,
            adjacency=adjacency,
            scope=scope,
        )

    def edge_to_site(self, edge: Any, gauge_copy: int = 0) -> int:
        """
        Map a graph edge token to a flattened site index.

        Conceptually, this addresses an edge degree of freedom inside a chosen gauge copy.

        :param edge: Edge identifier accepted by the graph/core (for example a keyed edge tuple).
        :param gauge_copy: Index of the gauge copy/block to address (0-based).

        :return: Flattened site index in ``[0, size)``.
        """

        return self.hilbert.edge_to_site(edge, gauge_copy=gauge_copy)

    def site_to_edge(self, site: int) -> tuple[int, Any]:
        """
        Inverse mapping from a flattened site index to a structured edge representation.

        :param site: Flattened site index in ``[0, size)``.

        :return: A core-defined representation identifying the gauge copy and the underlying graph edge.
        """

        return self.hilbert.site_to_edge(site)

    def states_to_numbers(
        self,
        states: Any,
        *,
        backend: Literal["auto", "netket", "python"] = "auto",
        return_dtype: str = "auto",
        validate: bool = False,
    ) -> Any:
        """
        Convert basis states to sequential integers using NetKet-compatible ordering.

        Depending on ``backend``, this conversion may be delegated to NetKet, performed in pure Python,
        or chosen automatically by the core. For large batches, the backend choice can materially affect
        performance and memory use.

        :param states: A single state ``(N,)`` or a batch ``(B, N)``.
        :param backend: Conversion backend: ``"auto"``, ``"netket"``, or ``"python"``.
        :param return_dtype: Output dtype selector (core-defined; ``"auto"`` chooses a sensible default).
        :param validate: If True, validate that inputs are legal basis states before conversion.

        :return: Integer label(s) corresponding to the provided basis state(s).
        """

        return self.hilbert.states_to_numbers(
            states, backend=backend, return_dtype=return_dtype, validate=validate
        )

    def numbers_to_states(
        self,
        numbers: Any,
        *,
        backend: Literal["auto", "netket", "python"] = "auto",
        validate: bool = False,
    ) -> jax.Array:
        """
        Convert sequential integers to basis states using NetKet-compatible ordering.

        Depending on ``backend``, this conversion may be delegated to NetKet, performed in pure Python,
        or chosen automatically by the core.

        :param numbers: Integer label or array of labels.
        :param backend: Conversion backend: ``"auto"``, ``"netket"``, or ``"python"``.
        :param validate: If True, validate that inputs are within the representable range.

        :return: Basis state(s) corresponding to the provided label(s).
        """

        return self.hilbert.numbers_to_states(
            numbers, backend=backend, validate=validate
        )

    def __getattr__(self, name: str) -> Any:
        """
        Forward unknown *public* attribute access to the core.

        This supports legacy code that accessed core methods directly on the interface. Private names
        (starting with ``"_"``) are intentionally not forwarded.

        :param name: Requested attribute name.
        :return: The corresponding attribute on the core.

        :raises AttributeError: If ``name`` is private or does not exist on the core.
        """

        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.hilbert, name)

    def __dir__(self) -> list[str]:
        """
        Return an augmented attribute listing including both interface and core attributes.

        This improves tab-completion and interactive discoverability.

        :return: Sorted list of attribute names.
        """

        return sorted(set(super().__dir__()) | set(dir(self.hilbert)))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"dimensions={scientific(self.hilbert.dimensions)}, "
            f"cutoff={self.hilbert.cutoff}, "
            f"dofs={self.hilbert.allowed_basis_states}, "
            f"is_gauge_invariant={self.is_gauge_invariant}, "
            f"is_finite={self.is_finite}, "
            f"is_indexable={self.is_indexable}"
            f")"
        )
