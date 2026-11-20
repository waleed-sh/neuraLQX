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


"""Rank identity, deadlines, and transport contracts."""

from __future__ import annotations

import os
import socket
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from neuralqx.utils.errors import NeuralqxValueError

from ..errors import ProfileError
from ..errors import ProfileTimeoutError
from ..trace.io import read_json
from ..trace.io import write_json
from .clock import ClockSample


@dataclass(frozen=True, slots=True)
class Rank:
    """Process identity used for trace ownership, lane names, and host telemetry election."""

    index: int
    """Zero-based process index in the distributed job."""

    count: int
    """Total number of processes expected in the merged capture."""

    hostname: str
    """Host identity used to label ranks and elect telemetry samplers."""

    pid: int
    """Operating-system process ID, retained for diagnostics."""

    @classmethod
    def detect(cls, index: int | None, count: int | None) -> Rank:
        """Resolve an explicit rank pair or query the initialized JAX runtime.

        Args:
            index: Process index, or None for automatic detection.
            count: Process count, supplied together with index.

        Returns:
            Validated process identity including hostname and operating-system PID.

        Raises:
            NeuralqxValueError: Only one override is supplied or the rank is out of range.
        """
        if (index is None) != (count is None):
            raise NeuralqxValueError("rank and world_size must be supplied together.")
        if index is None:
            import jax

            index, count = jax.process_index(), jax.process_count()
        if type(index) is not int or type(count) is not int or not 0 <= index < count:
            raise NeuralqxValueError("Expected 0 <= rank < world_size.")
        return cls(index, count, socket.gethostname(), os.getpid())

    def to_dict(self) -> dict:
        """Return identity fields suitable for a registration or manifest."""
        return asdict(self)


def wait_until(predicate, timeout_s: float, description: str):
    """Poll a coordination predicate with a bounded monotonic deadline.

    Args:
        predicate: Callable returning a truthy completion value or a false value.
        timeout_s: Positive deadline in seconds.
        description: Operation description included in timeout diagnostics.

    Returns:
        The first truthy predicate result.

    Raises:
        ProfileTimeoutError: The deadline expires before completion.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        result = predicate()
        if result:
            return result
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProfileTimeoutError(f"Timed out waiting for {description}.")
        time.sleep(min(0.01, remaining))


def rank_directory(root: Path, index: int) -> Path:
    """Return the fixed rank directory beneath an output root.

    Args:
        root: Run or collection directory.
        index: Nonnegative process index.

    Returns:
        Path named rank- followed by a zero-padded process index.
    """
    return root / f"rank-{index:05d}"


def validate_peer(peer: dict, run_id: str, count: int) -> None:
    """Reject registrations from a different run or invalid rank topology.

    Args:
        peer: Received rank registration.
        run_id: Required shared run identifier.
        count: Expected number of participating processes.

    Raises:
        ProfileError: Run identity, process count, or rank index is invalid.
    """
    if peer.get("run_id") != run_id or peer.get("count") != count:
        raise ProfileError("Profiling peers disagree on run_id or world_size.")
    index = peer.get("index")
    if type(index) is not int or not 0 <= index < count:
        raise ProfileError("Invalid profiling rank identity.")


class LocalCoordinator:
    """Publish single-rank or offline artifacts without interprocess traffic.

    Distributed transports extend this lifecycle. Offline multi-rank manifests
    are marked unmapped and require measured offsets during a later merge.
    """

    def __init__(self, root: Path, rank: Rank, options, hello: dict):
        """Store transport context and initialize an identity clock sample.

        Args:
            root: Local run directory.
            rank: Process identity.
            options: Immutable transport and collection options.
            hello: Registration payload shared with participating ranks.
        """
        self.root = root
        """Local or shared run directory used by this rank."""

        self.rank = rank
        """Identity of the process owning this coordinator."""

        self.options = options
        """Immutable coordination and capture policy."""

        self.hello = hello
        """Registration payload including run identity and clock sample count."""

        self.peers = [hello]
        """Known registrations used for topology validation and telemetry election."""

        self.start_clock = ClockSample(time.perf_counter_ns(), 0, 0)
        """Entry sample mapping local monotonic time to rank zero."""

    def prepare(self) -> None:
        """Prepare local capture without rendezvous or external communication."""
        pass

    def collect(self, manifest: dict) -> list[Path] | None:
        """Commit a rank manifest and describe which artifacts can be merged.

        Args:
            manifest: Completed local trace metadata, updated with clock calibration.

        Returns:
            One manifest for a single process. None for offline multi-rank capture.
        """
        now = time.perf_counter_ns()
        manifest["clock"] = {
            "method": "local" if self.rank.count == 1 else "unmapped",
            "start": self.start_clock.to_dict(),
            "end": ClockSample(now, 0, 0).to_dict(),
        }
        path = rank_directory(self.root, self.rank.index) / "manifest.json"
        write_json(path, manifest)
        return [path] if self.rank.count == 1 else None

    def close(self) -> None:
        """Release transport resources. Local coordination owns none."""
        pass

    def telemetry_leader(self) -> bool:
        """Return whether this rank is the lowest registered rank on its hostname."""
        return self.rank.index == min(
            peer["index"]
            for peer in self.peers
            if peer["hostname"] == self.rank.hostname
        )


def completed_manifests(root: Path, count: int) -> list[Path] | None:
    """Find every committed rank manifest without accepting pending manifests.

    Args:
        root: Shared run directory.
        count: Expected number of participating ranks.

    Returns:
        Rank-ordered paths, or None until all are present.
    """
    paths = [rank_directory(root, index) / "manifest.json" for index in range(count)]
    return paths if all(path.is_file() for path in paths) else None


def load_peers(root: Path, count: int, run_id: str) -> list[dict] | None:
    """Read and validate all shared-storage registrations when available.

    Args:
        root: Shared run directory.
        count: Expected number of ranks.
        run_id: Required common run identifier.

    Returns:
        Rank registrations, or None while any registration is missing.

    Raises:
        ProfileError: Registrations disagree or contain duplicate rank identities.
    """
    paths = [rank_directory(root, index) / "hello.json" for index in range(count)]
    if not all(path.is_file() for path in paths):
        return None
    peers = [read_json(path) for path in paths]
    for peer in peers:
        validate_peer(peer, run_id, count)
    if {peer["index"] for peer in peers} != set(range(count)):
        raise ProfileError("Profiling ranks are missing or duplicated.")
    return peers
