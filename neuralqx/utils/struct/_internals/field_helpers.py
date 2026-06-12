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


import typing

from collections.abc import Callable
from collections.abc import Sequence

from typing import Any

from ._types import ValidatorLike


class MissingType:
    """Sentinel marking the absence of a default/default_factory."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"


MISSING = MissingType()


class DefaultSentinel:
    """Signature sentinel shown for default constructor arguments."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<default>"


FACTORY_DEFAULT = DefaultSentinel()


def normalize_validators(validator: ValidatorLike) -> tuple[Callable[..., Any], ...]:
    """Normalise validator input to a tuple of callables."""
    if validator is None:
        return ()
    if isinstance(validator, Sequence) and not isinstance(validator, (str, bytes)):
        return tuple(typing.cast(Sequence[Callable[..., Any]], validator))
    return (typing.cast(Callable[..., Any], validator),)
