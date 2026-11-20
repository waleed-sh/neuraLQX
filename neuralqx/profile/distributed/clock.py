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


"""Round-trip clock estimates and affine drift correction."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

from ..errors import ProfileError


@dataclass(frozen=True, slots=True)
class ClockSample:
    """One round-trip estimate mapping a worker monotonic clock to rank zero."""

    local_ns: int
    """Worker-clock midpoint of the round trip in monotonic nanoseconds."""

    offset_ns: int
    """Nanoseconds added to worker time to estimate rank-zero time."""

    uncertainty_ns: int
    """Half the transport round-trip delay, rounded upward to nanoseconds."""

    @classmethod
    def round_trip(cls, t0: int, t1: int, t2: int, t3: int) -> ClockSample:
        """Estimate offset and a transport-delay uncertainty from four timestamps.

        Args:
            t0: Worker monotonic send time in nanoseconds.
            t1: Rank-zero monotonic receive time in nanoseconds.
            t2: Rank-zero monotonic reply time in nanoseconds.
            t3: Worker monotonic receive time in nanoseconds.

        Returns:
            Worker midpoint, offset to add to worker time, and half the round-trip
            delay after subtracting rank-zero service time. Asymmetric transport
            delay remains within that uncertainty under the round-trip clock model.

        Raises:
            ProfileError: Timestamps imply a negative elapsed or transport interval.
        """
        elapsed = (t3 - t0) - (t2 - t1)
        if t3 < t0 or t2 < t1 or elapsed < 0:
            raise ProfileError("Invalid profiling clock exchange.")
        return cls((t0 + t3) // 2, ((t1 - t0) + (t2 - t3)) // 2, (elapsed + 1) // 2)

    def to_dict(self) -> dict:
        """Return JSON-serializable nanosecond fields for the rank manifest."""
        return asdict(self)


class ClockMap:
    """Map local monotonic time to rank zero using two bounded clock estimates.

    Offset drift is fitted only when its change exceeds both measurement
    uncertainties combined. Otherwise a constant offset avoids interpreting
    transport jitter as clock-rate drift. Nonlinear drift is not measured.
    """

    def __init__(self, start: dict, end: dict):
        """Validate entry and exit samples and resolve any measurable linear drift.

        Args:
            start: ClockSample fields measured at session entry.
            end: ClockSample fields measured at session exit.

        Raises:
            ProfileError: Samples are reversed or imply more than one percent drift.
        """
        self.start = ClockSample(**start)
        """Best entry clock sample selected by minimum uncertainty."""

        self.end = ClockSample(**end)
        """Best exit clock sample selected by minimum uncertainty."""
        elapsed = self.end.local_ns - self.start.local_ns
        if elapsed < 0:
            raise ProfileError("The final clock sample precedes the initial sample.")
        difference = self.end.offset_ns - self.start.offset_ns
        resolved = abs(difference) > self.start.uncertainty_ns + self.end.uncertainty_ns
        self.slope = difference / elapsed if elapsed > 0 and resolved else 0.0
        """Resolved offset change per local nanosecond, or zero when hidden by jitter."""
        if abs(self.slope) > 0.01:
            raise ProfileError(
                "Clock calibration implies more than one percent drift. Refusing misleading alignment."
            )

    def relative_us(
        self, local_origin: int, timestamp_us: float, global_origin: int
    ) -> float:
        """Transform rank-relative microseconds without subtracting large floats.

        Args:
            local_origin: Rank-local monotonic trace origin in nanoseconds.
            timestamp_us: Event time relative to local_origin in microseconds.
            global_origin: Rank-zero monotonic origin of the merged trace in nanoseconds.

        Returns:
            Event timestamp in merged-trace microseconds.
        """
        delta = (local_origin - self.start.local_ns) + timestamp_us * 1000
        baseline = local_origin + self.start.offset_ns - global_origin
        return (baseline + delta * self.slope) / 1000 + timestamp_us

    def root_ns(self, timestamp_ns: int) -> int:
        """Map a local absolute monotonic timestamp to rank-zero nanoseconds.

        Args:
            timestamp_ns: Local monotonic timestamp in nanoseconds.

        Returns:
            Offset-corrected timestamp, including resolved linear drift.
        """
        return (
            timestamp_ns
            + self.start.offset_ns
            + round((timestamp_ns - self.start.local_ns) * self.slope)
        )

    @property
    def uncertainty_ns(self) -> int:
        """Return the endpoint calibration uncertainty, including unresolved offset change."""
        uncertainty = max(self.start.uncertainty_ns, self.end.uncertainty_ns)
        return uncertainty + (
            abs(self.end.offset_ns - self.start.offset_ns) if not self.slope else 0
        )
