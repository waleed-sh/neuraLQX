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


"""Process-wide capture ownership, with no environment reads on the hot path."""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

from .errors import ProfileStateError

if TYPE_CHECKING:
    from .session import Profiler

active: Profiler | None = None
"""Current process-owned session, or None when annotations should do no work."""
_lock = threading.Lock()


def claim(session: Profiler) -> None:
    """Acquire exclusive process-wide session ownership.

    Args:
        session: Session to expose to annotations in every thread.

    Raises:
        ProfileStateError: Another session already owns the process.
    """
    global active
    with _lock:
        if active is not None:
            raise ProfileStateError(
                "A profiler already owns this process. Nested sessions are unsupported."
            )
        active = session


def release(session: Profiler) -> None:
    """Clear ownership only if the supplied session is still the owner.

    Args:
        session: Session completing normal shutdown or startup rollback.
    """
    global active
    with _lock:
        if active is session:
            active = None


def _after_fork() -> None:
    global active, _lock
    active = None
    _lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
