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


"""Typed emission-clause specification values."""

from __future__ import annotations

import dataclasses
from typing import Any

from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError

_EMISSION_CLAUSE_MODES: frozenset[str] = frozenset(
    {
        "emit",
        "emit_if",
        "emit_elseif",
        "emit_else",
    }
)


@dataclasses.dataclass(frozen=True, repr=False)
class EmissionClauseSpec:
    """
    One normalized emission-clause action.

    Attributes:
        mode: Emission action mode.
        predicate: Optional branch predicate (used by conditional modes).
        update: Rewrite program payload.
        matrix_element: Matrix-element payload.
        amplitude: Deprecated alias of *matrix_element*.
        tag: Optional branch tag.
    """

    mode: str = "emit"
    predicate: Any = True
    update: Any = None
    matrix_element: Any = 1.0
    amplitude: Any | None = None
    tag: Any = None

    def __post_init__(self) -> None:
        if self.mode not in _EMISSION_CLAUSE_MODES:
            raise NeuralqxValueError(
                f"Unsupported emission clause mode: {self.mode!r}. "
                f"Allowed: {sorted(_EMISSION_CLAUSE_MODES)}."
            )


def coerce_emission_clause_spec(value: Any) -> EmissionClauseSpec:
    """
    Coerces one value to an :class:`EmissionClauseSpec`.

    Args:
        value: Clause output value.

    Returns:
        EmissionClauseSpec: Normalized spec.

    Raises:
        TypeError: If *value* is unsupported.
    """
    if isinstance(value, EmissionClauseSpec):
        return value
    raise NeuralqxTypeError(
        f"Cannot coerce {type(value)!r} into an EmissionClauseSpec. "
        "Emission clauses must return EmissionClauseSpec values."
    )


__all__ = [
    "EmissionClauseSpec",
    "coerce_emission_clause_spec",
]
