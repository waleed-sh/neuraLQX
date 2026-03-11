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
Abstract plum-dispatch entry points for state enumeration.

Dispatch model
--------------
1. Public calls use the core object:

       states_to_numbers(core, states, ...)
       numbers_to_states(core, numbers, ...)

2. The core overload forwards to ``core.index``.

3. Concrete implementations dispatch on ``(index_class, core_class)``.
"""

from typing import Any

from plum import dispatch

from ._abstract import HilbertStateEnumerator


@dispatch
def states_to_numbers(
    space: object,
    states: Any,
    **kwargs: Any,
) -> Any:
    """
    Public wrapper: dispatch via ``space.index``.

    This overload intentionally avoids importing any concrete/abstract core
    classes to keep the dispatch layer independent of core module import order.
    """
    index = getattr(space, "index", None)
    if index is None:
        raise TypeError(
            "states_to_numbers(space, ...) expects `space` to expose an `index` "
            "enumerator attribute."
        )
    return states_to_numbers(index, space, states, **kwargs)


@dispatch
def states_to_numbers(
    index: HilbertStateEnumerator,
    space: object,
    states: Any,
    *args,
    **kwargs: Any,
) -> Any:
    """Abstract fallback for ``(index, core, states)`` dispatch."""
    raise NotImplementedError(
        "No states_to_numbers dispatch registered for "
        f"index={type(index).__name__}, core={type(space).__name__}."
    )


@dispatch
def numbers_to_states(
    space: object,
    numbers: Any,
    **kwargs: Any,
) -> Any:
    """
    Public wrapper: dispatch via ``space.index``.

    This overload intentionally avoids importing any concrete/abstract core
    classes to keep the dispatch layer independent of core module import order.
    """
    index = getattr(space, "index", None)
    if index is None:
        raise TypeError(
            "numbers_to_states(space, ...) expects `space` to expose an `index` "
            "enumerator attribute."
        )
    return numbers_to_states(index, space, numbers, **kwargs)


@dispatch
def numbers_to_states(
    index: HilbertStateEnumerator,
    space: object,
    numbers: Any,
    *args,
    **kwargs: Any,
) -> Any:
    """Abstract fallback for ``(index, core, numbers)`` dispatch."""
    raise NotImplementedError(
        "No numbers_to_states dispatch registered for "
        f"index={type(index).__name__}, core={type(space).__name__}."
    )
