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


"""Read-only NVIDIA management providers, independent of JAX device indices."""

from __future__ import annotations

import csv
import io
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol

from ..errors import ProfileError


@dataclass(frozen=True, slots=True)
class Reading:
    """One GPU management sample identified by its physical UUID."""

    uuid: str
    """Physical GPU UUID, retained across management and trace artifacts."""

    values: dict[str, float]
    """Available counters keyed by metric names that include their units."""


class Provider(Protocol):
    """Read-only GPU management interface, independent of kernel timing."""

    name: str
    """Provider identifier recorded in the capture manifest."""

    def sample(self) -> list[Reading]:
        """Return current readings for selected physical GPUs without JAX synchronization."""
        ...

    def close(self) -> None:
        """Release owned management resources without changing GPU configuration."""
        ...


class NvidiaSmiProvider:
    """Query GPU management counters with a bounded read-only nvidia-smi subprocess.

    Commands request CSV output only. They never change clock, power, or device
    settings. GPU UUIDs and management indices are unrelated to JAX logical IDs.
    """

    name = "nvidia-smi"
    """Manifest identifier for subprocess-based NVIDIA management queries."""

    _fields = (
        "uuid",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "clocks.sm",
        "clocks.mem",
    )
    _metrics = (
        ("utilization_gpu_percent", 1),
        ("utilization_memory_percent", 1),
        ("memory_used_bytes", 1024 * 1024),
        ("memory_total_bytes", 1024 * 1024),
        ("temperature_celsius", 1),
        ("power_watts", 1),
        ("clock_sm_mhz", 1),
        ("clock_memory_mhz", 1),
    )

    def __init__(self, devices: tuple[str, ...], timeout_s: float):
        """Resolve the executable and construct a read-only query.

        Args:
            devices: Physical GPU UUIDs or management indices. Empty selects all.
            timeout_s: Maximum duration of one subprocess query.
        """
        executable = shutil.which("nvidia-smi")
        if executable is None:
            raise ProfileError("nvidia-smi is not available on PATH.")
        self.command = [
            executable,
            f"--query-gpu={','.join(self._fields)}",
            "--format=csv,noheader,nounits",
        ]
        """Fixed read-only query, with optional physical-device selection."""
        if devices:
            self.command.append(f"--id={','.join(devices)}")
        self.timeout_s = timeout_s
        """Maximum time in seconds allowed for one management subprocess."""

    def sample(self) -> list[Reading]:
        """Run one bounded query and return available metrics in documented units."""
        process = subprocess.run(
            self.command,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
        return self.parse(process.stdout)

    @classmethod
    def parse(cls, output: str) -> list[Reading]:
        """Convert unit-free CSV rows into physical-GPU readings.

        Args:
            output: stdout from the provider's fixed nvidia-smi query.

        Returns:
            GPU readings with bytes, percent, Celsius, watts, and MHz. Unsupported
            fields are omitted rather than represented as zero.

        Raises:
            ProfileError: Rows have invalid shape, UUIDs, or numeric fields.
        """
        result = []
        for row in csv.reader(io.StringIO(output)):
            if not row:
                continue
            if len(row) != len(cls._fields):
                raise ProfileError("Unexpected nvidia-smi CSV field count.")
            uuid = row[0].strip()
            if not uuid or uuid in ("N/A", "[N/A]"):
                raise ProfileError("nvidia-smi did not supply a GPU UUID.")
            values = {}
            for raw, (name, scale) in zip(row[1:], cls._metrics, strict=True):
                raw = raw.strip()
                if raw in (
                    "N/A",
                    "[N/A]",
                    "[Not Supported]",
                    "[Not Found]",
                    "Not Supported",
                    "",
                ):
                    continue
                try:
                    value = float(raw) * scale
                except ValueError as error:
                    raise ProfileError(
                        f"Invalid nvidia-smi value for {name}: {raw!r}."
                    ) from error
                if math.isfinite(value):
                    values[name] = value
            result.append(Reading(uuid, values))
        if not result:
            raise ProfileError("nvidia-smi returned no GPU readings.")
        return result

    def close(self) -> None:
        """Complete shutdown. Each subprocess already closes after its sample."""
        pass


class NvmlProvider:
    """Read GPU management counters through an owned NVML initialization."""

    name = "nvml"
    """Manifest identifier for direct NVIDIA management-library queries."""

    def __init__(self, devices: tuple[str, ...]):
        """Initialize NVML and resolve physical device selections.

        Args:
            devices: GPU UUIDs or management indices. Empty selects every visible GPU.

        Raises:
            ProfileError: The optional package is absent or no GPU is selected.
        """
        try:
            pynvml = import_module("pynvml")
        except ImportError as error:
            raise ProfileError(
                "NVML telemetry requires the nvidia-ml-py package."
            ) from error
        self.nvml = pynvml
        """Imported management module providing NVML functions and exceptions."""

        self.initialized = False
        """Whether this instance owns an NVML initialization requiring shutdown."""
        try:
            pynvml.nvmlInit()
            self.initialized = True
            selections = devices or tuple(
                str(index) for index in range(pynvml.nvmlDeviceGetCount())
            )
            self.devices = []
            """Selected physical GPU UUIDs paired with their opaque NVML handles."""
            for selection in selections:
                handle = (
                    pynvml.nvmlDeviceGetHandleByIndex(int(selection))
                    if selection.isdecimal()
                    else pynvml.nvmlDeviceGetHandleByUUID(selection)
                )
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                self.devices.append(
                    (uuid.decode() if isinstance(uuid, bytes) else uuid, handle)
                )
            if not self.devices:
                raise ProfileError("NVML found no selected GPUs.")
        except Exception:
            self.close()
            raise

    def sample(self) -> list[Reading]:
        """Return available per-GPU counters, omitting unsupported or restricted fields."""
        n = self.nvml
        result = []
        for uuid, handle in self.devices:
            values = {}
            queries = (
                (
                    lambda handle=handle: n.nvmlDeviceGetUtilizationRates(handle),
                    lambda value: {
                        "utilization_gpu_percent": value.gpu,
                        "utilization_memory_percent": value.memory,
                    },
                ),
                (
                    lambda handle=handle: n.nvmlDeviceGetMemoryInfo(handle),
                    lambda value: {
                        "memory_used_bytes": value.used,
                        "memory_total_bytes": value.total,
                    },
                ),
                (
                    lambda handle=handle: n.nvmlDeviceGetTemperature(
                        handle, n.NVML_TEMPERATURE_GPU
                    ),
                    lambda value: {"temperature_celsius": value},
                ),
                (
                    lambda handle=handle: n.nvmlDeviceGetPowerUsage(handle),
                    lambda value: {"power_watts": value / 1000},
                ),
                (
                    lambda handle=handle: n.nvmlDeviceGetClockInfo(
                        handle, n.NVML_CLOCK_SM
                    ),
                    lambda value: {"clock_sm_mhz": value},
                ),
                (
                    lambda handle=handle: n.nvmlDeviceGetClockInfo(
                        handle, n.NVML_CLOCK_MEM
                    ),
                    lambda value: {"clock_memory_mhz": value},
                ),
            )
            for query, unpack in queries:
                try:
                    values.update(unpack(query()))
                except (n.NVMLError_NotSupported, n.NVMLError_NoPermission):
                    continue
            result.append(Reading(uuid, values))
        return result

    def close(self) -> None:
        """Release this provider's NVML initialization exactly once."""
        if self.initialized:
            self.initialized = False
            self.nvml.nvmlShutdown()
