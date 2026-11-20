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


"""Contract between session orchestration and native activity collectors."""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol


class ActivityCollector(Protocol):
    """Native capture contract independent of session scheduling and transport."""

    path: Path | None
    """Raw native trace location, available only after successful export."""

    devices: list[dict]
    """Local device identity and hardware metadata for the capture manifest."""

    def annotation(self, name: str, *, attributes=None, step_num=None):
        """Return a native host annotation paired with one application region.

        Args:
            name: Stable operation label.
            attributes: Bounded scalar metadata describing this operation.
            step_num: Logical step identifier, or None for an ordinary region.

        Returns:
            Context manager that records native host context without device waits.
        """
        ...

    def start(self) -> None:
        """Acquire and start native collection, raising if ownership is unavailable."""
        ...

    def stop(self) -> None:
        """Stop an owned collection and export its raw native trace."""
        ...

    def calibrate(self) -> dict:
        """Return the native-to-host clock anchor after collection stops.

        Returns:
            Native microseconds, local monotonic nanoseconds, and bracket uncertainty.
        """
        ...

    def events(
        self, origin_ns: int, begin_ns: int, end_ns: int, window: int
    ) -> Iterator[dict]:
        """Yield retained events in rank-relative microseconds.

        Args:
            origin_ns: Rank-local monotonic timestamp represented by zero in the trace.
            begin_ns: First retained local monotonic timestamp.
            end_ns: Last retained local monotonic timestamp.
            window: Capture identifier used to separate repeated correlation IDs.

        Yields:
            Native metadata, activity, and flow records in the retained interval.
        """
        ...
