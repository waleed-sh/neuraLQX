#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from neuralqx.utils.errors import neuralqxError


class InvalidSU2CutoffError(neuralqxError):
    """Raised when ``two_j_max`` is not a valid non-negative integer cutoff."""

    def __init__(self, value: object):
        super().__init__(
            "SU(2) Hilbert spaces require an integer two_j_max >= 0. "
            f"Received {value!r}."
        )


class SU2UnconstrainedNotImplementedError(neuralqxError):
    """Raised when users request the magnetic-index SU(2) link space."""

    def __init__(self):
        super().__init__(
            "HilbertSU2 currently implements the gauge-invariant spin-network "
            "basis only. The unconstrained magnetic-index link basis is a "
            "different Hilbert space and is not implemented in this version."
        )


class UnsupportedRecouplingSchemeError(neuralqxError):
    """Raised when a recoupling tree kind is not supported yet."""

    def __init__(self, recoupling: object):
        super().__init__(
            "The first SU(2) implementation supports recoupling='left_comb' "
            f"only. Received {recoupling!r}."
        )


class InvalidSU2StateLayoutError(neuralqxError):
    """Raised when a state has the wrong shape for a compiled SU(2) layout."""

    def __init__(self, expected_size: int, got: object):
        super().__init__(
            "Invalid SU(2) spin-network state shape: expected final axis "
            f"of length {expected_size}, got {got!r}."
        )


class SU2ChannelConstructionFailureError(neuralqxError):
    """Raised when local channels cannot be completed for supplied edge spins."""

    def __init__(self, vertex: object, edge_spins: object):
        super().__init__(
            "Could not construct admissible SU(2) recoupling channels at "
            f"vertex {vertex!r} for incident doubled-spins {edge_spins!r}."
        )


class SU2ExactEnumerationTooLargeError(neuralqxError):
    """Raised when exact-mode enumeration exceeds the configured guard."""

    def __init__(self, max_states: int):
        super().__init__(
            "Exact SU(2) enumeration is intentionally restricted to tiny "
            f"systems. The valid basis exceeded max_exact_states={max_states}."
        )


__all__ = [
    "InvalidSU2CutoffError",
    "InvalidSU2StateLayoutError",
    "SU2ChannelConstructionFailureError",
    "SU2ExactEnumerationTooLargeError",
    "neuralqxError",
    "SU2UnconstrainedNotImplementedError",
    "UnsupportedRecouplingSchemeError",
]
