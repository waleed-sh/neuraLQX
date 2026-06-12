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


"""Custom neuraLQX errors."""

from __future__ import annotations

import warnings

from textwrap import dedent

from neuralqx import config

_ERROR_SEPARATOR = "=" * 113


def _error_anchor_url(error_name: str) -> str:
    base = config.get_static("Errors Directory")
    return f"{base.rstrip('/')}#{error_name.lower()}"


def new_error_content(message: str, *, error_name: str | None = None) -> str:
    """Formats an error or warning message with documentation links."""
    base = config.get_static("Errors Directory")
    specific = _error_anchor_url(error_name) if error_name else None
    specific_block = f"\n\nThis specific error:\n\t{specific}" if specific else ""

    return (
        f"{dedent(message)}"
        f"\n"
        f"\n{_ERROR_SEPARATOR}"
        f"\n"
        f"You can find a list of all neuraLQX errors and warnings including their "
        f"\ndetailed explanations at:"
        f"\n\t {base}{specific_block}"
        f"\n{_ERROR_SEPARATOR}"
        f"\n"
    )


def _nqx_formatwarning(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"


warnings.formatwarning = _nqx_formatwarning


class neuralqxError(Exception):
    """Base class for all neuraLQX package errors."""

    def __init__(self, msg: str):
        super().__init__(new_error_content(msg, error_name=self.__class__.__name__))


class neuralqxWarning(Warning):
    """Base class for all neuraLQX package warnings."""

    def __init__(self, msg: str, stack_level: int = 2):
        self.msg = new_error_content(msg, error_name=self.__class__.__name__)
        super().__init__(self.msg)
        warnings.warn(self, stacklevel=stack_level)

    def __str__(self) -> str:
        return self.msg


class StructError(neuralqxError):
    """Base exception for the struct and pytree subsystem."""


class FrozenStructError(StructError, AttributeError):
    """Raised when mutation is attempted on a frozen ``Struct`` instance."""


class SerializationError(StructError):
    """Raised when struct or registered-pytree I/O operations fail."""


class ValidationError(StructError, TypeError):
    """Raised when struct declarations or runtime values are invalid."""


__all__ = [
    "neuralqxError",
    "neuralqxWarning",
    "StructError",
    "FrozenStructError",
    "SerializationError",
    "ValidationError",
]
