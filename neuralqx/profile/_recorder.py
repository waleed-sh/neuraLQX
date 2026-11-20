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


"""Bounded per-thread event storage with batched budget reservations."""

from __future__ import annotations

import asyncio
import threading
import weakref
from dataclasses import dataclass
from dataclasses import field


@dataclass(slots=True)
class _Buffer:
    """One writer thread's event storage and budget accounting."""

    events: list[dict] = field(default_factory=list)
    """Completed events owned by this buffer's writer thread."""

    remaining: int = 0
    """Reserved event slots that this thread has not used."""

    dropped: int = 0
    """Events rejected by the recorder or the event budget."""

    open_spans: int = 0
    """Entered spans that have not completed."""


class Recorder:
    """Store bounded application events without a shared lock for each append.

    Args:
        max_events: Process-wide event reservation limit.
        max_lanes: Maximum named thread and coroutine lanes.

    A thread reserves up to 64 slots at once. Export must occur after producers
    have stopped and accepting has been cleared.
    """

    def __init__(self, max_events: int, max_lanes: int):
        """Allocate accounting structures without reserving event storage."""
        self.max_events = max_events
        """Maximum event slots reserved across all writer threads."""

        self.max_lanes = max_lanes
        """Maximum named thread and coroutine lanes."""

        self._reserved = 0
        self._lock = threading.Lock()
        self._tls = threading.local()
        self._buffers: list[_Buffer] = []
        self._overflow = _Buffer()
        self._dropped_lanes = 0
        self._tasks: weakref.WeakKeyDictionary[asyncio.Task, int] = (
            weakref.WeakKeyDictionary()
        )
        self.lanes: dict[int, str] = {}
        """Trace lane IDs mapped to bounded human-readable names."""

        self.accepting = True
        """Whether new events may be appended. Cleared before export."""

    def buffer(self) -> _Buffer:
        """Return the calling thread's buffer, allocating it on first use."""
        buffer = getattr(self._tls, "buffer", None)
        if buffer is None:
            buffer = _Buffer()
            with self._lock:
                if len(self._buffers) >= self.max_lanes + 1:
                    buffer = self._overflow
                else:
                    self._buffers.append(buffer)
            self._tls.buffer = buffer
        return buffer

    def lane(self) -> tuple[int | None, bool]:
        """Resolve the current coroutine or thread to a stable trace lane.

        Returns:
            Lane ID and whether it belongs to a coroutine. The ID is None when the
            lane budget is exhausted. Coroutine identity is retained weakly.
        """
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is not None:
            with self._lock:
                lane = self._tasks.get(task)
                if lane is None:
                    lane = self._new_lane(
                        f"{threading.current_thread().name} / {task.get_name()}"
                    )
                    if lane is not None:
                        self._tasks[task] = lane
            return lane, True
        if not hasattr(self._tls, "lane"):
            with self._lock:
                self._tls.lane = self._new_lane(threading.current_thread().name)
        return self._tls.lane, False

    def _new_lane(self, name: str) -> int | None:
        if len(self.lanes) >= self.max_lanes:
            return None
        lane = len(self.lanes) + 1
        self.lanes[lane] = name[:256]
        return lane

    def append(self, event: dict, buffer: _Buffer | None = None) -> bool:
        """Append one event if recording and the reservation budget permit it.

        Args:
            event: Complete Chrome Trace Event record owned by the caller.
            buffer: Calling thread's buffer. None resolves it lazily.

        Returns:
            True when stored. False increments the observable drop count.
        """
        buffer = self.buffer() if buffer is None else buffer
        if buffer is self._overflow:
            with self._lock:
                buffer.dropped += 1
            return False
        if not self.accepting:
            buffer.dropped += 1
            return False
        if not buffer.remaining:
            with self._lock:
                count = min(64, self.max_events - self._reserved)
                self._reserved += count
            buffer.remaining = count
        if not buffer.remaining:
            buffer.dropped += 1
            return False
        buffer.remaining -= 1
        buffer.events.append(event)
        return True

    def drop_lane(self) -> None:
        """Count an event omitted because no application lane was available."""
        with self._lock:
            self._dropped_lanes += 1

    def events(self):
        """Yield stored records after producers stop, without a global sort or copy."""
        for buffer in self._buffers:
            yield from buffer.events

    def stats(self) -> dict:
        """Return event, reservation, lane, and unfinished-span counts for the manifest."""
        return {
            "recorded_events": sum(len(buffer.events) for buffer in self._buffers),
            "dropped_events": sum(buffer.dropped for buffer in self._buffers)
            + self._overflow.dropped
            + self._dropped_lanes,
            "events_without_lane": self._dropped_lanes,
            "unfinished_spans": sum(buffer.open_spans for buffer in self._buffers),
            "reserved_slots": self._reserved,
            "max_events": self.max_events,
            "lanes": len(self.lanes),
        }
