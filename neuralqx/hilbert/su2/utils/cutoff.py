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


"""Cutoff validation helpers for SU(2) Hilbert spaces."""

from __future__ import annotations

from neuralqx.hilbert.su2.errors import InvalidSU2CutoffError


def validate_two_j_max(value: object) -> int:
    """Validates and normalizes a doubled-spin cutoff.

    Args:
        value: Candidate maximum doubled-spin label.

    Returns:
        Normalized integer cutoff.

    Raises:
        InvalidSU2CutoffError: If ``value`` is not a non-negative integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidSU2CutoffError(value)
    return int(value)


__all__ = ["validate_two_j_max"]
