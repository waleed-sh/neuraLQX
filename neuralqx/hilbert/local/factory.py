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


from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .abstract import AbstractLocalSpace
from .explicit import ExplicitLocalSpace


def as_local_space(value: AbstractLocalSpace | Sequence[Any]) -> AbstractLocalSpace:
    """Coerces user input to a local-space object.

    Args:
        value: Existing local-space instance or sequence of explicit values.

    Returns:
        ``value`` unchanged when it already implements ``AbstractLocalSpace``.
        Otherwise an ``ExplicitLocalSpace`` wrapping the supplied values.
    """
    if isinstance(value, AbstractLocalSpace):
        return value
    return ExplicitLocalSpace(tuple(value))


__all__ = ["as_local_space"]
