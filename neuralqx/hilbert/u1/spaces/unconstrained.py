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


"""Unconstrained homogeneous U(1)^N Hilbert space.

The unconstrained U(1) Hilbert space uses the shared U(1) construction without
additional gauge-fixing relations. It is useful for free edge-charge models and
as the base class for constructive gauge-invariant spaces.
"""

from __future__ import annotations

from .abstract import AbstractU1Hilbert


class U1Hilbert(AbstractU1Hilbert):
    """Unconstrained homogeneous U(1)^N edge Hilbert space.

    ``U1Hilbert`` places one or more compact U(1) charge components on every
    edge of a graph. A flat state stores all components in component-major
    order with trailing shape ``gauge_dimensions * graph.n_edges``. Each scalar
    entry is drawn from the same finite charge range produced by ``cutoff``,
    ``step``, ``positive_qn``, and ``qn_start``.

    The space is unconstrained in the sense that no gauge-fixing relations are
    imposed. Every locally valid charge assignment is accepted, random states
    are sampled directly from the local range, generic basis ranking is
    available while the rectangular dimension fits the indexing limit, and
    proposal moves may update free edges or plaquettes without a projection
    step.

    This class is the public entry point for free edge-charge models and the
    parent of :class:`U1GaugeInvariantHilbert`. It inherits graph-layout views,
    local-index conversion, random sampling, free-edge proposals, and plaquette
    proposals from :class:`AbstractU1Hilbert` and its mixins.
    """

    def __repr__(self) -> str:
        """Returns a compact representation of U(1) Hilbert metadata.

        Returns:
            String containing the dimension summary, local charge range
            parameters, graph edge count, gauge-component count, constrained
            flag, and generic indexability status.
        """
        return (
            f"U1Hilbert("
            f"dimensions={self.dimensions_pretty}, "
            f"cutoff={self.cutoff}, "
            f"step={self.step}, "
            f"local_size={self.local_size}, "
            f"n_edges={self.tiny_size}, "
            f"gauge_dimensions={self.gauge_dimensions}, "
            f"constrained={self.constrained}, "
            f"indexable={self.is_indexable})"
        )


UnconstrainedU1Hilbert = U1Hilbert
