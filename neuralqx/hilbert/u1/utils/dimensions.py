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


"""Dimension formatting helpers for large Hilbert spaces."""

from __future__ import annotations


def scientific_int(value: int, precision: int = 4) -> str:
    """Returns a compact scientific string for an exact integer dimension.

    Args:
        value: Exact dimension to format.
        precision: Number of digits after the leading digit.

    Returns:
        Decimal string for small values or scientific notation for large ones.
    """
    text = str(int(value))
    if len(text) <= precision + 2:
        return text
    decimals = text[1 : 1 + precision].ljust(precision, "0")
    return f"{text[0]}.{decimals}e+{len(text) - 1}"
