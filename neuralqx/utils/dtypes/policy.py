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


"""
DType policy abstraction for neuraLQX runtime defaults.

The :class:`DTypePolicy` class is the single source of truth for effective
dtype defaults derived from package configuration (which may be driven by
environment variables).
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any


def _validate_choice(value: Any, *, field_name: str, allowed: Collection[str]) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    if normalized not in allowed:
        raise ValueError(
            f"{field_name} must be one of {tuple(sorted(allowed))!r}; "
            f"received {value!r}."
        )
    return normalized


def _promote_for_x64(name: str) -> str:
    """Promotes dtype names to x64 variants when x64 mode is enabled."""
    promoted = {
        "float32": "float64",
        "complex64": "complex128",
        "int32": "int64",
    }
    return promoted.get(name, name)


@dataclass(frozen=True, slots=True)
class DTypePolicy:
    """
    Canonical dtype policy for neuraLQX arrays.

    Attributes:
        real: Default real-valued array dtype.
        complex: Default complex-valued array dtype.
        index: Integer dtype for index and topology arrays.
    """

    real: str
    complex: str
    index: str

    def __post_init__(self) -> None:
        from neuralqx.utils.dtypes import names

        object.__setattr__(
            self,
            "real",
            _validate_choice(
                self.real, field_name="real", allowed=names.REAL_DTYPE_NAMES
            ),
        )
        object.__setattr__(
            self,
            "complex",
            _validate_choice(
                self.complex, field_name="complex", allowed=names.COMPLEX_DTYPE_NAMES
            ),
        )
        object.__setattr__(
            self,
            "index",
            _validate_choice(
                self.index, field_name="index", allowed=names.INDEX_DTYPE_NAMES
            ),
        )

    @classmethod
    def from_config(cls, config_manager: Any | None = None) -> DTypePolicy:
        """
        Builds a dtype policy from neuraLQX configuration values.

        Args:
            config_manager: Optional config manager instance. If omitted, the
                global :data:`neuralqx.configs.cfg` singleton is used.

        Returns:
            Policy populated from effective configuration values.
        """
        cfg = config_manager
        if cfg is None:
            from neuralqx.configs import cfg as global_cfg

            cfg = global_cfg

        real = str(cfg.DTYPE_REAL)
        complex_name = str(cfg.DTYPE_COMPLEX)
        index = str(cfg.DTYPE_INDEX)

        if bool(cfg.ENABLE_X64):
            real = _promote_for_x64(real)
            complex_name = _promote_for_x64(complex_name)
            index = _promote_for_x64(index)

        return cls(
            real=real,
            complex=complex_name,
            index=index,
        )

    def as_dict(self) -> dict[str, str]:
        """Serializes the policy into a plain dictionary."""
        return {
            "real": self.real,
            "complex": self.complex,
            "index": self.index,
        }


__all__ = ["DTypePolicy"]
