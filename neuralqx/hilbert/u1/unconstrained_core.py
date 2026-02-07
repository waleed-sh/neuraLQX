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

r"""
Unconstrained :math:`U(1)` Hilbert-space core.

This module implements :class:`UnconstrainedHilbertU1Core`, the concrete *core* backend for the
kinematical (unconstrained) Hilbert space of :math:`U(1)`-valued edge degrees of freedom on a given
graph.

A configuration (basis state) is represented as a 1D array ``sigma`` of length

.. math::

    N = G\,E,

where :math:`E` is the number of edge sites in a single gauge copy and :math:`G` is the number of
gauge copies (``gauge_dimensions``). The layout is block-strided:

.. math::

    \sigma = \bigl[q_{0,0}, \ldots, q_{0,E-1} \;\big|\; q_{1,0}, \ldots, q_{1,E-1} \;\big|\; \cdots \bigr],

so that ``copy g`` occupies indices ``[g*E, (g+1)*E)``.

No gauge-fixing or Gauss constraints are imposed in this core: all tensor-product basis
configurations built from the local domain are admissible. The local domain itself is defined by
the base class :class:`neuralqx.hilbert._abstract_hilbert_core.AbstractHilbertSpace` from the
parameters ``cutoff``, ``step``, ``positive_qn``, and ``qn_start``.
"""

from __future__ import annotations

import logging
from typing import Optional
from typing import Union

import netket as nk
import humanize

from neuralqx.graph.core import AbstractGraph
from ..abstract_hilbert_core import AbstractHilbertSpace
from neuralqx.debug import event


class UnconstrainedHilbertU1Core(AbstractHilbertSpace):
    r"""
    Unconstrained :math:`U(1)` Hilbert-space core.

    This core represents the full tensor-product Hilbert space for :math:`U(1)` edge variables on a
    graph, without imposing gauge invariance or gauge fixing. It is the implementation backend used by
    :class:`neuralqx.hilbert.u1.HilbertU1` when ``is_gauge_invariant=False``.

    Degrees of freedom and layout
        Let :math:`E` be the number of edge sites in a single gauge copy (in neuraLQX this is typically
        the number of dual-graph vertices, i.e. one site per *primal* edge degree of freedom), and let
        :math:`G` be ``gauge_dimensions``. A basis configuration is stored as a flat vector
        :math:`\sigma \in \mathbb{R}^{N}` (or :math:`\mathbb{Z}^{N}`) with

        .. math::

            N = G\,E.

        Conceptually, the same data may be viewed as a rank-2 array :math:`q_{g,e}` of shape
        :math:`(G, E)` (gauge copy :math:`g`, edge site :math:`e`), using the block layout

        .. math::

            \sigma_{gE + e} = q_{g,e}.

    Local basis (single-site domain)
        The allowed local values are constructed by the base class
        :class:`neuralqx.hilbert._abstract_hilbert_core.AbstractHilbertSpace` as a
        :class:`netket.utils.StaticRange`. In the common symmetric case (``positive_qn=False``), the
        domain is

        .. math::

            \mathcal{Q} = \{-c,\,-c+s,\,\ldots,\,c-s,\,c\},

        where :math:`c` is ``cutoff`` and :math:`s` is ``step``. When ``positive_qn=True``, the domain is
        restricted to a non-negative range starting at ``qn_start`` (defaulting to 0 if omitted); the
        exact endpoint is determined by the base-class construction.

    NetKet representation
        This core builds a flat NetKet :class:`netket.hilbert.HomogeneousHilbert` with:

        - ``N = E * G`` sites,
        - the local states given by ``self.allowed_basis_states``.

    Dimension bookkeeping
        If the NetKet Hilbert is indexable (``self.hilbert.is_indexable``), the total number of basis
        states is taken from ``self.hilbert.n_states``. Otherwise, the core stores the formal tensor
        size

        .. math::

            \dim(\mathcal{H}) = |\mathcal{Q}|^{N},

        which grows exponentially in :math:`N` and may be extremely large. A scientific-notation string
        is cached for readable representations.

    :param graph: Graph defining the edge degrees of freedom and edge-index mappings.
    :param cutoff: Cutoff controlling the extent of the local :math:`U(1)` basis.
    :param step: Step size between consecutive allowed local values.
    :param gauge_dimensions: Number of gauge copies :math:`G` in the flattened block layout.
    :param positive_qn: If ``True``, restrict the local domain to non-negative values.
    :param qn_start: Starting value for the local domain when ``positive_qn=True`` (defaults to 0 if
      omitted).
    :return: None.
    :raises ValueError: If ``gauge_dimensions < 1`` (validated by the base class).
    """

    __slots__ = ("_print_dims",)

    def __init__(
        self,
        graph: AbstractGraph,
        cutoff: Union[int, float],
        *,
        step: Union[int, float] = 1,
        gauge_dimensions: int = 1,
        positive_qn: bool = False,
        qn_start: Optional[int] = None,
    ):
        """
        Initialise an unconstrained :math:`U(1)` Hilbert-space core.

        This constructor performs three main steps:

        1. Delegates local-domain construction and bookkeeping to the base class
           :class:`neuralqx.hilbert._abstract_hilbert_core.AbstractHilbertSpace` (cutoff/step, dtype,
           allowed local basis states, and gauge-layout metadata).
        2. Builds the full NetKet Hilbert space as a :class:`netket.hilbert.HomogeneousHilbert` with
           ``N = tiny_size * gauge_dimensions`` sites.
        3. Computes and caches the total Hilbert-space dimension (exact if indexable, otherwise the formal
           tensor-product size).

        :param graph: Graph defining the edge degrees of freedom and index mappings.
        :param cutoff: Cutoff controlling the allowed local values.
        :param step: Step size between allowed local values.
        :param gauge_dimensions: Number of gauge copies :math:`G` in the flattened block layout.
        :param positive_qn: If ``True``, restrict the local domain to non-negative values.
        :param qn_start: Starting local value when ``positive_qn=True`` (defaults to 0 if omitted).
        :return: None.
        :raises ValueError: If ``gauge_dimensions < 1``.
        """

        super().__init__(
            graph,
            cutoff,
            step=step,
            gauge_dimensions=gauge_dimensions,
            positive_qn=positive_qn,
            qn_start=qn_start,
        )

        N = self.tiny_size * self.gauge_dimensions
        self._hilbert = nk.hilbert.HomogeneousHilbert(
            local_states=self.allowed_basis_states,
            N=N,
        )

        # dimension (may be huge...)
        if not self.hilbert.is_indexable:
            self._dimensions = int(self.local_size) ** int(self.size)
        else:
            self._dimensions = int(self.hilbert.n_states)

        self._print_dims = humanize.number.scientific(self._dimensions, precision=4)

        event(
            msg="INITIALIZING HILBERT CORE",
            tag="HILBERT:U1:UNCONSTRAINED:INIT",
            level=logging.INFO,
        )

    def __repr__(self):
        return (
            f"UnconstrainedHilbertU1Core("
            f"dimensions={self._print_dims}, cutoff={self.cutoff}, "
            f"dofs={self.allowed_basis_states}, "
            f"gauge_dimensions={self.gauge_dimensions}, "
            f"indexable={self.hilbert.is_indexable})"
        )
