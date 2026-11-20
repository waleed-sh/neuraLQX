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


"""A cancellable background sampler with observable provider failures."""

from __future__ import annotations

import threading
import time

from ..errors import ProfileError
from .providers import NvidiaSmiProvider
from .providers import NvmlProvider


class TelemetrySampler:
    """Run cancellable management queries independently of application dispatch."""

    def __init__(self, options, emit):
        """Store sampling policy and the event sink without starting a thread.

        Args:
            options: Immutable telemetry policy and device selections.
            emit: Callback accepting a Reading, query midpoint nanoseconds, and query
                duration nanoseconds. It must avoid blocking application execution.
        """
        self.options = options
        """Immutable provider selection, interval, device, and timeout policy."""

        self.emit = emit
        """Sink receiving each GPU reading and its query timing."""

        self.provider = None
        """Selected management provider, or None if automatic selection failed."""

        self.failures: list[str] = []
        """Initialization, sampling, or shutdown failures preserved for diagnostics."""

        self.samples = 0
        """Number of completed management queries, across all selected devices."""

        self.thread = None
        """Background sampler thread, absent until provider startup succeeds."""

        self.stop_event = threading.Event()
        """Cancellation event interrupting the delay between queries."""

    def start(self) -> None:
        """Select a provider and launch its sampling thread.

        Auto selection tries NVML then nvidia-smi and records fallback failures.
        An explicitly requested provider must initialize successfully.

        Raises:
            ProfileError: The explicit provider cannot initialize.
        """
        choices = (
            ("nvml", "nvidia-smi")
            if self.options.telemetry == "auto"
            else (self.options.telemetry,)
        )
        for choice in choices:
            try:
                self.provider = (
                    NvmlProvider(self.options.telemetry_devices)
                    if choice == "nvml"
                    else NvidiaSmiProvider(
                        self.options.telemetry_devices,
                        self.options.nvidia_smi_timeout_s,
                    )
                )
                break
            except Exception as error:
                self.failures.append(f"{choice}: {type(error).__name__}: {error}")
        if self.provider is None:
            if self.options.telemetry != "auto":
                raise ProfileError(
                    "Requested telemetry provider failed: " + " | ".join(self.failures)
                )
            return
        self.thread = threading.Thread(
            target=self._run, name="neuralqx-profile-telemetry", daemon=True
        )
        self.thread.start()

    def _run(self) -> None:
        assert self.provider is not None
        try:
            while not self.stop_event.is_set():
                start = time.perf_counter_ns()
                readings = self.provider.sample()
                end = time.perf_counter_ns()
                self.samples += 1
                if not self.stop_event.is_set():
                    for reading in readings:
                        self.emit(reading, (start + end) // 2, end - start)
                if self.stop_event.wait(self.options.telemetry_interval_s):
                    break
        except Exception as error:
            self.failures.append(f"sampling: {type(error).__name__}: {error}")
        finally:
            try:
                self.provider.close()
            except Exception as error:
                self.failures.append(f"shutdown: {type(error).__name__}: {error}")

    def request_stop(self) -> None:
        """Request cancellation without waiting for an in-flight management query."""
        self.stop_event.set()

    def stop(self) -> None:
        """Request cancellation and join with a bounded wait, recording shutdown failures."""
        self.request_stop()
        if self.thread is not None:
            self.thread.join(timeout=self.options.nvidia_smi_timeout_s + 1)
            if self.thread.is_alive():
                self.failures.append(
                    "Telemetry provider did not stop before the shutdown deadline."
                )

    def status(self) -> dict:
        """Return provider, sample, timing, device, and failure metadata for the manifest."""
        return {
            "requested": self.options.telemetry,
            "provider": None if self.provider is None else self.provider.name,
            "samples": self.samples,
            "failures": list(self.failures),
            "scope": self.options.telemetry_scope,
            "devices": list(self.options.telemetry_devices),
            "timestamps": "query_midpoint",
        }
