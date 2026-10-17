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


"""Conversion of driver log payloads into host-side logging values."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Number
from typing import Any

import numpy as np

import jax


def normalize_log_tree(item: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a driver log mapping into nested Python/NumPy values."""
    return {str(key): normalize_log_value(value) for key, value in item.items()}


def normalize_log_value(value: Any) -> Any:
    """Normalize one value for host-side retention and plotting."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return normalize_log_value(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): normalize_log_value(child) for key, child in value.items()}
    if isinstance(value, (str, bytes, bool)) or value is None:
        return value
    if isinstance(value, Number):
        return value
    if _is_jax_array(value):
        return _array_to_host_value(jax.device_get(value))
    if isinstance(value, np.ndarray):
        return _array_to_host_value(value)
    if hasattr(value, "tolist"):
        try:
            listed = value.tolist()
        except TypeError:
            return value
        return normalize_log_value(listed)
    if isinstance(value, tuple):
        return tuple(normalize_log_value(item) for item in value)
    if isinstance(value, list):
        return [normalize_log_value(item) for item in value]
    return value


def serializable_log_tree(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-compatible representation of a normalized log tree."""
    return {str(key): serializable_log_value(value) for key, value in item.items()}


def serializable_log_value(value: Any) -> Any:
    """Convert normalized values into JSON-compatible containers."""
    if isinstance(value, Mapping):
        return {str(key): serializable_log_value(child) for key, child in value.items()}
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return {
                "real": np.asarray(value.real).tolist(),
                "imag": np.asarray(value.imag).tolist(),
            }
        return value.tolist()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.generic):
        return serializable_log_value(value.item())
    if isinstance(value, tuple):
        return [serializable_log_value(item) for item in value]
    if isinstance(value, list):
        return [serializable_log_value(item) for item in value]
    return value


def _is_jax_array(value: Any) -> bool:
    return isinstance(value, jax.Array)


def _array_to_host_value(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return arr


__all__ = [
    "normalize_log_tree",
    "normalize_log_value",
    "serializable_log_tree",
    "serializable_log_value",
]
