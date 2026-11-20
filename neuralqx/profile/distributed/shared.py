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


"""Shared-filesystem rendezvous with no polling during application capture."""

from __future__ import annotations

import time

from ..errors import ProfileError
from ..trace.io import read_json
from ..trace.io import write_json
from .clock import ClockSample
from .common import LocalCoordinator
from .common import completed_manifests
from .common import load_peers
from .common import rank_directory
from .common import wait_until


class SharedCoordinator(LocalCoordinator):
    """Exchange boundary clock samples and committed manifests through shared storage."""

    def prepare(self) -> None:
        """Rendezvous with all ranks, validate registrations, and calibrate entry clocks."""
        write_json(
            rank_directory(self.root, self.rank.index) / "hello.json", self.hello
        )
        self.peers = wait_until(
            lambda: load_peers(self.root, self.rank.count, self.hello["run_id"]),
            self.options.timeout_s,
            "all profiling rank registrations on shared storage",
        )
        if any(
            peer["clock_samples"] != self.options.clock_samples for peer in self.peers
        ):
            raise ProfileError(
                "Shared-storage peers must use the same profile_clock_samples."
            )
        self.start_clock = self._calibrate("start")

    def _calibrate(self, phase: str) -> ClockSample:
        directory = self.root / "coordination" / phase
        directory.mkdir(parents=True, exist_ok=True)
        if self.rank.index == 0:
            pending = {
                (rank, sample)
                for rank in range(1, self.rank.count)
                for sample in range(self.options.clock_samples)
            }

            def respond():
                for rank, sample in tuple(pending):
                    stem = directory / f"{rank}-{sample}"
                    if stem.with_suffix(".request").is_file():
                        t1 = time.perf_counter_ns()
                        write_json(
                            stem.with_suffix(".response"),
                            {"t1": t1, "t2": time.perf_counter_ns()},
                        )
                        pending.remove((rank, sample))
                return not pending

            wait_until(
                respond,
                self.options.timeout_s,
                f"{phase} clock requests from all profiling ranks",
            )
            sample = ClockSample(time.perf_counter_ns(), 0, 0)
            write_json(directory / "ready.json", {"ready": True})
            return sample
        samples = []
        for index in range(self.options.clock_samples):
            stem = directory / f"{self.rank.index}-{index}"
            t0 = time.perf_counter_ns()
            write_json(stem.with_suffix(".request"), {"rank": self.rank.index})
            response = stem.with_suffix(".response")
            wait_until(
                response.is_file,
                self.options.timeout_s,
                f"rank-zero {phase} clock response",
            )
            payload = read_json(response)
            t3 = time.perf_counter_ns()
            samples.append(ClockSample.round_trip(t0, payload["t1"], payload["t2"], t3))
        wait_until(
            (directory / "ready.json").is_file,
            self.options.timeout_s,
            "clock rendezvous completion",
        )
        return min(samples, key=lambda sample: sample.uncertainty_ns)

    def collect(self, manifest: dict):
        """Calibrate exit clocks and publish this rank's completed manifest.

        Args:
            manifest: Completed local trace metadata, updated with clock samples.

        Returns:
            All committed manifests on rank zero. None on workers.

        Raises:
            ProfileTimeoutError: A rank fails to complete its clock exchange or manifest.
        """
        end = self._calibrate("end")
        manifest["clock"] = {
            "method": "shared_round_trip",
            "start": self.start_clock.to_dict(),
            "end": end.to_dict(),
        }
        path = rank_directory(self.root, self.rank.index) / "manifest.json"
        write_json(path, manifest)
        if self.rank.index:
            return None
        return wait_until(
            lambda: completed_manifests(self.root, self.rank.count),
            self.options.timeout_s,
            "completed rank traces on shared storage",
        )
