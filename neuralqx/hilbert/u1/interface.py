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
U(1) Hilbert-space interface.

This module defines :class:`HilbertU1`, the user-facing entry point for constructing Hilbert spaces
with local :math:`U(1)` degrees of freedom on a graph.

A basis configuration is represented as a 1D array ``sigma`` whose entries encode edge charges
(selected from a discrete set determined by ``cutoff`` and ``step``). When ``gauge_dimensions = G``,
the configuration is stored in a block (strided) layout with :math:`G` consecutive copies of the
same edge set. Writing :math:`E` for the number of edge sites in one copy, the flattened state has
length :math:`N = G E` and can be viewed as a tensor :math:`q_{g,e}` with shape :math:`(G, E)`.

The interface delegates all implementation details to a concrete core:

- An unconstrained core for kinematical spaces (independent edge degrees of freedom).
- A constrained core for gauge-invariant spaces, where gauge fixing is imposed and non-free edge
  variables are reconstructed deterministically.
"""

from __future__ import annotations

from typing import List
from typing import Optional
from typing import Union

import jax.numpy
import jax.numpy as jnp

from neuralqx.graph.core import AbstractGraph
from neuralqx.hilbert import AbstractHilbertInterface
from .unconstrained_core import UnconstrainedHilbertU1Core
from .constrained_core import ConstrainedHilbertU1Core


class HilbertU1(AbstractHilbertInterface):
    r"""
    Concrete :math:`U(1)` Hilbert-space interface.

    This class provides a stable, user-facing API for constructing Hilbert spaces on a given graph with
    local :math:`U(1)` edge degrees of freedom (integer or real-valued charges depending on
    ``cutoff``/``step``). It is intentionally thin: it validates high-level configuration and then
    selects an appropriate *core* implementation which owns the NetKet Hilbert object, indexing logic,
    and move/state operations.

    Core selection:
        - If ``is_gauge_invariant=False`` (default), :class:`UnconstrainedHilbertU1Core` is used.
          The space is kinematical: each edge carries an independent local variable.
        - If ``is_gauge_invariant=True``, :class:`ConstrainedHilbertU1Core` is used.
          A gauge-fixing specification must be provided via ``gauge_fixing`` or generated via
          ``auto_constraint=True``. The constrained core typically treats a subset of edge variables as
          *free* and reconstructs the remaining *slave* variables to satisfy the constraint.

    Gauge copies:
        If ``gauge_dimensions = G``, a configuration is stored as :math:`G` consecutive blocks, each of
        length :math:`E` (the number of edge sites in one copy). Equivalently, the flattened state
        ``sigma`` can be reshaped to :math:`q_{g,e}` with shape :math:`(G, E)`.

    :param graph: Graph on which the Hilbert space is defined (edges provide the sites).
    :param cutoff: Cut-off controlling the local quantum-number domain (model-specific semantics).
    :param step: Step size between allowed local quantum numbers.
    :param gauge_dimensions: Number of gauge copies :math:`G` stored in the flattened configuration.
    :param is_gauge_invariant: If ``True``, construct a gauge-invariant (constrained) space.
    :param gauge_fixing: Gauge-fixing specification used when ``is_gauge_invariant=True``. The expected
      structure is core-defined; for the U(1) constrained core it is typically a list describing which
      edges are fixed or how constraints are imposed.
    :param auto_constraint: If ``True`` and ``is_gauge_invariant=True``, attempt to auto-generate a
      gauge fixing from the graph.
    :param positive_qn: If ``True``, restrict allowed local quantum numbers to be non-negative.
    :param qn_start: Starting quantum number when ``positive_qn=True`` (defaults to 0 if omitted).
    :param kwargs: Additional backend-specific options forwarded to the core builder.
    :return: None.
    :raises UnspecifiedGaugeFixingError: If ``is_gauge_invariant=True`` and neither ``gauge_fixing`` nor
      ``auto_constraint`` is provided.
    :raises AutoConstraintGaugeFixingConflictError: If ``is_gauge_invariant=True`` and both an explicit
      ``gauge_fixing`` and ``auto_constraint=True`` are provided.
    """

    def _build_core(
        self,
        *,
        graph: AbstractGraph,
        cutoff: Union[int, float] = jnp.inf,
        step: Union[int, float] = 1,
        gauge_dimensions: int = 1,
        is_gauge_invariant: bool = False,
        gauge_fixing: List[List[List[Union[int, str]]]] = None,
        auto_constraint: Optional[bool] = False,
        positive_qn: bool = False,
        qn_start: Optional[int] = None,
        **_unused,
    ):
        """
        Construct and return the concrete U(1) Hilbert-space core.

        This method is called by :class:`neuralqx.hilbert.AbstractHilbertInterface` during initialisation.
        It selects the appropriate backend core based on the gauge-invariance flag:

        - If ``is_gauge_invariant=True``, return a :class:`ConstrainedHilbertU1Core` configured to enforce
          the supplied gauge fixing (or to generate one if ``auto_constraint=True``).
        - Otherwise, return an :class:`UnconstrainedHilbertU1Core` representing the unconstrained (kinematic)
          Hilbert space.

        The returned object must be a fully initialised core that provides the NetKet Hilbert instance,
        local basis construction (from ``cutoff``/``step``), and all state/move utilities expected by the
        interface.

        :param graph: Graph defining edge degrees of freedom and edge indexing.
        :param cutoff: Cut-off controlling the local quantum-number domain.
        :param step: Step size between allowed local quantum numbers.
        :param gauge_dimensions: Number of gauge copies :math:`G` stored in the flattened configuration.
        :param is_gauge_invariant: If ``True``, build a constrained core; otherwise build an unconstrained
          core.
        :param gauge_fixing: Gauge-fixing specification for constrained construction (ignored when
          ``is_gauge_invariant=False``).
        :param auto_constraint: If ``True``, request gauge-fixing auto-generation in the constrained core.
        :param positive_qn: If ``True``, restrict allowed local quantum numbers to non-negative values.
        :param qn_start: Starting quantum number when ``positive_qn=True``.
        :param _unused: Extra keyword arguments forwarded by the interface and intentionally ignored here.
        :return: A concrete core instance, either :class:`ConstrainedHilbertU1Core` or
          :class:`UnconstrainedHilbertU1Core`.
        :raises ValueError: If core construction fails due to invalid parameters (for example, an invalid
          local basis implied by ``cutoff``/``step`` or an incompatible gauge-fixing specification).
        """

        if is_gauge_invariant:
            return ConstrainedHilbertU1Core(
                graph,
                cutoff,
                step=step,
                gauge_dimensions=gauge_dimensions,
                constraint=gauge_fixing,
                auto_constraint=auto_constraint,
                positive_qn=positive_qn,
                qn_start=qn_start,
            )
        return UnconstrainedHilbertU1Core(
            graph,
            cutoff,
            step=step,
            gauge_dimensions=gauge_dimensions,
            positive_qn=positive_qn,
            qn_start=qn_start,
        )

    def edge_charges(
        self,
        sigma: jax.numpy.ndarray,
        edge: Union[tuple, list],
    ) -> jax.numpy.ndarray:
        r"""
        Return the charge values carried by a given graph edge in one or more configurations.

        A U(1) configuration can be viewed (via the underlying core layout) as a tensor
        :math:`q_{g,e}` with shape :math:`(G, E)`, where :math:`G` is the number of gauge copies and
        :math:`E` is the number of edge sites per copy. For a fixed edge (site) index :math:`e`, this method
        extracts the vector

        .. math::

            \bigl(q_{0,e}, q_{1,e}, \dots, q_{G-1,e}\bigr)

        for each configuration in the input batch.

        Input handling:
            - If ``sigma`` is a single state of shape ``(N,)``, it is treated as a batch of size 1.
            - If ``sigma`` has three or more dimensions, the leading dimensions are collapsed so that the
              last axis is interpreted as the flattened site axis.

        Output shape:
            The returned array is flattened to 1D. For a batch size :math:`B` and gauge dimensions :math:`G`,
            the output has shape ``(B * G,)``. For a single input state, the output has shape ``(G,)``.

        :param sigma: Basis state or batch of states. The final axis must represent the flattened site
          layout of length :math:`N = G E`.
        :param edge: Edge token in graph notation (for example ``(u, v, key)``) accepted by
          :meth:`neuralqx.hilbert.AbstractHilbertInterface.edge_to_site`.
        :return: A 1D array containing the extracted charge values across gauge copies (and batches), in
          the stable order ``[state0 copy0..copyG-1 | state1 copy0..copyG-1 | ...]``.
        :raises ValueError: If ``edge`` cannot be mapped to a valid site index for this Hilbert space.
        """

        if sigma.ndim >= 3:
            sigma = jax.lax.collapse(sigma, 0, 2)

        if sigma.ndim == 1:
            sigma = sigma[None, :]

        # get the strided view of the basis element
        sigma_strided = self.core.view(sigma)

        # get the edge in index notation
        # we only need the index in the zero copy as we will index the multi-dimensional
        # basis element array
        edge_idx = self.edge_to_site(edge, gauge_copy=0)

        # return the slice of the array that corresponds to the charges of the specified edges
        return sigma_strided[:, :, edge_idx].reshape(-1)
