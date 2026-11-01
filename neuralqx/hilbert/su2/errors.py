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


"""SU(2) spin-network Hilbert-space errors.

The exceptions in this module describe construction, layout, recoupling, and
state-completion failures specific to the SU(2) Hilbert API.
"""

# TODO: move to the errors in neuralqx.utils?

from __future__ import annotations


class SU2HilbertError(ValueError):
    """Base class for SU(2) Hilbert-space construction and state errors.

    All SU(2)-specific public exceptions inherit from this type so callers can
    catch configuration, layout, recoupling, and exact-enumeration failures
    without also catching unrelated :class:`ValueError` instances from lower
    level numerical code. The class is intentionally broad and carries no
    additional fields.
    """


class InvalidSU2CutoffError(SU2HilbertError):
    """Invalid maximum doubled-spin cutoff for an SU(2) Hilbert space.

    SU(2) labels are represented as non-negative doubled spins. The cutoff must
    therefore be an integer ``two_j_max >= 0``. This error is raised for missing
    cutoffs, booleans, non-integer values, negative values, or conflicting
    ``two_j_max`` and compatibility ``cutoff`` arguments.
    """

    def __init__(self, value: object) -> None:
        """Initializes the cutoff error message.

        Args:
            value: Invalid cutoff value supplied by the caller.

        The resulting message explains the non-negative integer requirement for
        doubled-spin cutoffs and echoes the offending value.
        """
        super().__init__(
            "SU(2) Hilbert spaces require an integer two_j_max >= 0. "
            f"Received {value!r}."
        )


class SU2UnconstrainedNotImplementedError(SU2HilbertError):
    """The spin-network constructor cannot select the magnetic-index basis.

    The SU2Hilbert alias intentionally represents the gauge-invariant
    spin-network basis. The unconstrained magnetic-index link basis has
    different state variables, validation rules, sampling behavior, and
    proposal kernels, so construction fails explicitly instead of silently
    producing the wrong representation. This avoids exposing misleading state
    layout or proposal semantics.
    """

    def __init__(self) -> None:
        """Initializes the unsupported-basis error message.

        The constructor takes no arguments because the error always describes
        the separate unconstrained SU(2) representation and points callers
        to its explicit constructor.
        """
        super().__init__(
            "The new SU(2) implementation represents the gauge-invariant "
            "spin-network basis. The unconstrained magnetic-index link basis "
            "is a different Hilbert space, use SU2PeterWeylHilbert explicitly."
        )


class UnsupportedRecouplingSchemeError(SU2HilbertError):
    """Requested SU(2) vertex recoupling scheme is unavailable.

    Spin-network vertices currently use the ``"left_comb"`` recoupling scheme.
    This error is raised when a caller asks for another scheme so the boundary
    of the implemented API is clear at construction time and downstream layout
    assumptions stay explicit for every SU(2) Hilbert-space method.
    """

    def __init__(self, recoupling: object) -> None:
        """Initializes the recoupling-scheme error message.

        Args:
            recoupling: Unsupported recoupling scheme requested by the caller.

        The resulting message names the currently supported scheme and echoes
        the requested value for configuration diagnostics.
        """
        super().__init__(
            "SU(2) currently supports recoupling='left_comb' only. "
            f"Received {recoupling!r}."
        )


class InvalidSU2StateLayoutError(SU2HilbertError):
    """Flat SU(2) spin-network state has the wrong trailing shape.

    SU(2) helpers expect the final axis of each state array to match the
    spin-network layout size: edge doubled-spin labels followed by vertex
    channel labels. This error reports arrays whose trailing shape cannot be
    interpreted as such a flat state.
    """

    def __init__(self, expected_size: int, got: object) -> None:
        """Initializes the state-layout error message.

        Args:
            expected_size: Required trailing state-axis length.
            got: Observed trailing shape or shape-like object.

        The resulting message is intended for user-facing validation failures
        where a flat state cannot be interpreted by the SU(2) layout.
        """
        super().__init__(
            "Invalid SU(2) spin-network state shape: expected final axis "
            f"of length {expected_size}, got {got!r}."
        )


class SU2ChannelConstructionFailureError(SU2HilbertError):
    """Admissible recoupling channels cannot be built for fixed edge labels.

    Some proposed edge-spin sectors cannot satisfy the local SU(2)
    admissibility rules at a vertex. This error is raised by helpers that are
    asked to produce a completed physical state and cannot find compatible
    recoupling channels for the supplied edge labels.
    """

    def __init__(self, vertex: object, edge_spins: object) -> None:
        """Initializes the channel-construction error message.

        Args:
            vertex: Vertex identifier associated with the failure.
            edge_spins: Edge doubled-spin labels that could not be completed.

        The resulting message preserves the local edge sector so callers can
        report or debug the inadmissible spin configuration.
        """
        super().__init__(
            "Could not construct SU(2)-admissible recoupling channels at "
            f"vertex {vertex!r} for incident doubled-spins {edge_spins!r}."
        )


class SU2ExactEnumerationTooLargeError(SU2HilbertError):
    """Exact SU(2) basis enumeration exceeded an explicit guard.

    Exact enumeration is intended for small validation systems, deterministic
    tests, and cases where ranking or unranking is required. The guard prevents
    accidental materialization of very large spin-network bases that would be
    expensive in memory and time. Increase the explicit limit only when the
    basis size is known to be acceptable.
    """

    def __init__(self, max_states: int) -> None:
        """Initializes the exact-enumeration guard error message.

        Args:
            max_states: Explicit enumeration limit that was exceeded.

        The resulting message explains that exact enumeration is intentionally
        guarded and reports the active limit.
        """
        super().__init__(
            "Exact SU(2) enumeration is restricted to small validation "
            f"systems. The valid basis exceeded max_states={max_states}."
        )


__all__ = [
    "InvalidSU2CutoffError",
    "InvalidSU2StateLayoutError",
    "SU2ChannelConstructionFailureError",
    "SU2ExactEnumerationTooLargeError",
    "SU2HilbertError",
    "SU2UnconstrainedNotImplementedError",
    "UnsupportedRecouplingSchemeError",
]
