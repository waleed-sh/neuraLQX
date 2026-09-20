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


"""Formatting helpers for Monte Carlo statistics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def scalar_item(value: Any) -> Any:
    """Return a Python scalar for scalar arrays, preserving non-scalars."""
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def format_stat_scalar(value: Any, *, precision: int = 3) -> str:
    """Format a scalar diagnostic compactly and robustly."""
    value = scalar_item(value)
    if isinstance(value, complex):
        if value.imag == 0:
            return format_stat_scalar(value.real, precision=precision)
        real = format_stat_scalar(value.real, precision=precision)
        imag = format_stat_scalar(abs(value.imag), precision=precision)
        sign = "+" if value.imag >= 0 else "-"
        return f"{real}{sign}{imag}j"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if number == 0:
        return f"{number:.{precision}f}"
    if abs(number) < 1e-3 or abs(number) >= 1e4:
        return f"{number:.{precision}e}"
    return f"{number:.{precision}f}"


def stats_repr(
    mean: Any,
    error_of_mean: Any,
    variance: Any,
    r_hat: Any,
) -> str:
    """Return the public scalar ``Stats`` representation."""
    text = (
        f"{format_stat_scalar(mean)} ± {format_stat_scalar(error_of_mean)} "
        f"[σ² = {format_stat_scalar(variance)}"
    )
    try:
        r_hat_value = float(scalar_item(r_hat))
    except (TypeError, ValueError):
        r_hat_value = math.nan
    if not math.isnan(r_hat_value):
        text += f", R̂ = {format_stat_scalar(r_hat_value)}"
    return text + "]"


__all__ = ["format_stat_scalar", "scalar_item", "stats_repr"]
