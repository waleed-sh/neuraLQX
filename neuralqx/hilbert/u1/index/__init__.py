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
U(1) indexing subpackage.

Provides concrete U(1) ``states_to_numbers`` / ``numbers_to_states``
implementations and registers plum dispatch overloads for
:class:`neuralqx.hilbert.u1.unconstrained_core.UnconstrainedHilbertU1Core`
and
:class:`neuralqx.hilbert.u1.constrained_core.ConstrainedHilbertU1Core`.
"""

from typing import Any

from plum import dispatch

from neuralqx.hilbert.u1.constrained_core import ConstrainedHilbertU1Core
from neuralqx.hilbert.u1.unconstrained_core import UnconstrainedHilbertU1Core
from neuralqx.hilbert.utils.index.dispatch import states_to_numbers
from neuralqx.hilbert.utils.index.dispatch import numbers_to_states

from .enumerator import U1ConstrainedStateEnumerator
from .enumerator import U1UnconstrainedStateEnumerator
from .mapping import states_to_numbers as _u1_states_to_numbers
from .mapping import numbers_to_states as _u1_numbers_to_states


@dispatch
def states_to_numbers(
    index: U1UnconstrainedStateEnumerator,
    space: UnconstrainedHilbertU1Core,
    states: Any,
    *args,
    **kwargs: Any,
) -> Any:
    """Dispatch overload: unconstrained U(1) states -> numbers."""
    return _u1_states_to_numbers(space, states, *args, **kwargs)


@dispatch
def numbers_to_states(
    index: U1UnconstrainedStateEnumerator,
    space: UnconstrainedHilbertU1Core,
    numbers: Any,
    *args,
    **kwargs: Any,
) -> Any:
    """Dispatch overload: unconstrained U(1) numbers -> states."""
    return _u1_numbers_to_states(space, numbers, *args, **kwargs)


@dispatch
def states_to_numbers(
    index: U1ConstrainedStateEnumerator,
    space: ConstrainedHilbertU1Core,
    states: Any,
    *args,
    **kwargs: Any,
) -> Any:
    """Dispatch overload: constrained U(1) states -> numbers."""
    return _u1_states_to_numbers(space, states, *args, **kwargs)


@dispatch
def numbers_to_states(
    index: U1ConstrainedStateEnumerator,
    space: ConstrainedHilbertU1Core,
    numbers: Any,
    *args,
    **kwargs: Any,
) -> Any:
    """Dispatch overload: constrained U(1) numbers -> states."""
    return _u1_numbers_to_states(space, numbers, *args, **kwargs)


__all__ = [
    "U1UnconstrainedStateEnumerator",
    "U1ConstrainedStateEnumerator",
    "states_to_numbers",
    "numbers_to_states",
]
