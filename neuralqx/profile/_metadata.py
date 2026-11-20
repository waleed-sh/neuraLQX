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


"""Small scalar metadata that cannot materialize a device array."""

from __future__ import annotations

import math
from collections.abc import Mapping

from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError

Scalar = str | int | float | bool | None
"""Accepted metadata values. Array scalars are deliberately excluded."""


def label(value: str, field: str = "name") -> str:
    """Validate a stable trace label without converting arbitrary objects.

    Args:
        value: Nonempty Python string of at most 256 characters.
        field: Field name included in an invalid-label diagnostic.

    Returns:
        The original validated string.

    Raises:
        NeuralqxValueError: The value is not a bounded nonempty string.
    """
    if type(value) is not str or not value or len(value) > 256:
        raise NeuralqxValueError(f"{field} must contain between 1 and 256 characters.")
    return value


def metadata(values: Mapping[str, Scalar] | None) -> dict[str, Scalar]:
    """Copy bounded scalar attributes without reading arrays or object representations.

    Args:
        values: Up to 32 string-keyed Python scalar fields, or None.

    Returns:
        Independent attributes with finite numbers and a bounded text budget.

    Raises:
        NeuralqxTypeError: A value is not a Python scalar.
        NeuralqxValueError: A key, value, or aggregate size exceeds the trace limits.
    """
    if values is None:
        return {}
    if len(values) > 32:
        raise NeuralqxValueError(
            "Trace metadata is limited to 32 scalar fields per event."
        )
    result = {}
    budget = 0
    for key, value in values.items():
        label(key, "metadata key")
        if type(value) not in (str, int, float, bool, type(None)):
            raise NeuralqxTypeError(
                "Trace metadata accepts Python scalars only. Device arrays are not converted."
            )
        if type(value) is float and not math.isfinite(value):
            raise NeuralqxValueError("Trace metadata must contain finite numbers.")
        if type(value) is int and value.bit_length() > 64:
            raise NeuralqxValueError("Trace integers must fit in 64 bits.")
        if type(value) is str and len(value) > 1024:
            raise NeuralqxValueError(
                "Trace string values are limited to 1024 characters."
            )
        budget += len(key) + (len(value) if type(value) is str else 24)
        if budget > 4096:
            raise NeuralqxValueError(
                "Trace metadata exceeds the 4096 character budget."
            )
        result[key] = value
    return result
