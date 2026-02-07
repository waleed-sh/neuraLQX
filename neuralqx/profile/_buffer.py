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

import threading
from collections import deque

from typing import Any
from typing import Deque
from typing import Dict
from typing import List


class TraceBuffer:
    """
    Bounded buffer for trace events in Chrome/Perfetto format.

    Note: This is only used when trace export is enabled. When disabled,
    the profiler avoids emitting any trace events to keep overhead minimal.
    """

    def __init__(self, max_events: int) -> None:
        self._buf: Deque[Dict[str, Any]] = deque(maxlen=max(1, int(max_events)))
        self._lock = threading.Lock()

    def append(self, event: Dict[str, Any]) -> None:
        # keep critical path short
        with self._lock:
            self._buf.append(event)

    def extend(self, events: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._buf.extend(events)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._buf)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
