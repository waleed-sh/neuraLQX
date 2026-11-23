# Copyright (c) 2026 The neuraLQX and nkDSL Authors - All rights reserved.
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


"""Shared naming rules for fluent DSL clause registration."""

from __future__ import annotations

from neuralqx.utils.errors import NeuralqxValueError

_FORBIDDEN_CLAUSE_METHOD_NAMES: frozenset[str] = frozenset(
    {
        "build",
        "compile",
        "emit",
        "named",
        "max_conn_size",
        "fanout",
        "__getattr__",
        "__dir__",
    }
)
"""Names reserved by the builder and therefore not allowed for clauses."""


def normalize_clause_method_name(name: str) -> str:
    """
    Normalizes and validates a fluent clause method name.

    Args:
        name: Raw method name candidate.

    Returns:
        str: Normalized method name.

    Raises:
        ValueError: If the method name is empty, not a valid identifier,
            prefixed with ``_``, or conflicts with reserved builder names.
    """
    normalized = str(name).strip()
    if not normalized:
        raise NeuralqxValueError("Clause method name must be a non-empty string.")
    if not normalized.isidentifier():
        raise NeuralqxValueError(
            f"Clause method name {normalized!r} is not a valid Python identifier."
        )
    if normalized.startswith("_"):
        raise NeuralqxValueError("Clause method names starting with '_' are reserved.")
    if normalized in _FORBIDDEN_CLAUSE_METHOD_NAMES:
        raise NeuralqxValueError(
            f"Clause method name {normalized!r} is reserved and cannot be registered."
        )
    return normalized


__all__ = ["normalize_clause_method_name"]
