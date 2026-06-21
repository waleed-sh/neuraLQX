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


"""Immutable normalization for user-facing gauge-fixing specs.

Gauge-fixing declarations may be supplied as nested lists. Freezing converts
those lists into tuples so the specification can be stored in static struct
fields and participate in hashing.
"""

from __future__ import annotations

from typing import Any

from neuralqx.hilbert.u1.constraints.types import GaugeFixingSpec


def freeze_gauge_fixing(spec: GaugeFixingSpec | None) -> GaugeFixingSpec | None:
    """Converts nested user gauge-fixing lists into hashable tuples.

    Args:
        spec: Gauge-fixing specification or ``None``.

    Returns:
        Immutable gauge-fixing specification preserving token values and nested
        structure, or ``None`` when no specification was supplied.
    """
    if spec is None:
        return None
    return tuple(
        tuple(tuple(_freeze_token(token) for token in side) for side in condition)
        for condition in spec
    )


def _freeze_token(token: Any) -> Any:
    """Recursively freezes a gauge-fixing token."""
    if isinstance(token, list):
        return tuple(_freeze_token(item) for item in token)
    if isinstance(token, tuple):
        return tuple(_freeze_token(item) for item in token)
    return token
