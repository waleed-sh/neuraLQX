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


"""Construction and validation for finite U(1) local ranges.

U(1) local domains are represented as arithmetic ranges. These helpers build
the scalar range from physical cutoff parameters and validate the number of
independent gauge components.
"""

from __future__ import annotations

import numpy as np

from neuralqx.hilbert.local import LocalRange


def u1_local_range(
    cutoff: int | float,
    step: int | float,
    *,
    positive_qn: bool,
    qn_start: int | float | None,
) -> LocalRange:
    """Builds the scalar finite U(1) quantum-number range.

    Args:
        cutoff: Symmetric magnitude cutoff before optional positive shift.
        step: Charge spacing.
        positive_qn: Whether to start the range at a non-negative value.
        qn_start: Optional explicit start value when ``positive_qn`` is true.

    Returns:
        Local arithmetic range of allowed U(1) charges.
    """
    if step == 0:
        raise ValueError("U(1) step must be non-zero.")
    raw_size = (2 * cutoff) / step
    nearest = round(raw_size)
    if not np.isclose(raw_size, nearest):
        raise ValueError("2 * cutoff must be an integer multiple of step.")
    start = (
        qn_start
        if positive_qn and qn_start is not None
        else (0 if positive_qn else -cutoff)
    )
    return LocalRange(start, step, int(nearest) + 1)


def validate_gauge_dimensions(gauge_dimensions: int) -> None:
    """Validates the number of U(1) gauge copies or components.

    Args:
        gauge_dimensions: Number of independent U(1) components per edge.

    Raises:
        TypeError: If ``gauge_dimensions`` is not an integer.
        ValueError: If ``gauge_dimensions`` is not positive.
    """
    if isinstance(gauge_dimensions, bool) or not isinstance(gauge_dimensions, int):
        raise TypeError("gauge_dimensions must be an integer.")
    if gauge_dimensions <= 0:
        raise ValueError("gauge_dimensions must be positive.")
