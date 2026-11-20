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


"""Immutable settings sampled once before a profiling session starts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from dataclasses import fields
from typing import TYPE_CHECKING
from typing import Literal

from neuralqx.utils.errors import NeuralqxValueError

from .schedule import Schedule

if TYPE_CHECKING:
    from neuralqx.config import ConfigManager


@dataclass(frozen=True, slots=True)
class ProfileOptions:
    """Capture policy independent of any running collector.

    Every field has a package option named profile_<field> and an environment
    default named NEURALQX_PROFILE_<FIELD>. ProfileOptions.from_config snapshots those
    values. Constructing this dataclass directly uses the defaults below.

    Attributes:
        level: off creates no artifacts, basic records application events,
            trace also captures native CPU and accelerator activity, and full
            additionally enables native Python call tracing by default.
        detail: summary retains driver phases. detailed adds named operator
            terms and estimator stages. developer also records internal dispatch
            boundaries. This does not turn on global Python call tracing.
        output_dir: Root directory for rank artifacts and the merged trace.
        run_id: Unique run directory name shared by all ranks. A single-process
            session generates one when omitted. Existing rank directories are
            never reused.
        transport: shared uses a shared filesystem, tcp transfers completed
            rank traces to rank zero, and offline writes independent artifacts.
        coordinator_address: Reachable rank-zero host:port for TCP transport.
        timeout_s: Deadline for each coordination or transfer phase.
        clock_samples: Round trips per boundary clock calibration.
        max_events: Total application and telemetry event budget per process.
            Native XLA buffers are controlled by JAX and are not covered.
        max_lanes: Maximum application thread and asyncio task lanes.
        jax_host_tracer_level: Native host tracing detail from 1 to 3. Level zero
            cannot supply the clock markers needed for native trace alignment.
        jax_python_tracer_level: Native Python call tracing, 0 or 1. None selects
            1 for full and 0 otherwise.
        nvtx: Emit process ranges to NVIDIA tools through the optional nvtx package.
        telemetry: off, auto, nvml, or nvidia-smi. Explicit providers must work.
            Auto reports unavailable providers in the capture metadata.
        telemetry_interval_s: Delay between background management queries.
        telemetry_scope: host elects the lowest rank on each hostname. rank
            samples independently, which also supports isolated GPU containers.
        telemetry_devices: GPU UUIDs or management indices. Empty selects all
            GPUs visible to the management provider, independently of JAX's
            logical device numbering and CUDA_VISIBLE_DEVICES.
        nvidia_smi_timeout_s: Timeout for a read-only subprocess query.
        max_trace_bytes: Limit on each compressed rank trace transferred over TCP.
        synchronize: boundary waits only on explicitly supplied results or a
            wait_for callback. none introduces no application completion fences.
        wait_steps: Unprofiled steps in each scheduled cycle.
        warmup_steps: Native warmup steps discarded during export.
        active_steps: Recorded steps per cycle. Zero records the whole context.
        repeat: Number of capture cycles. Zero repeats until context exit.
        skip_first: Unprofiled steps before the first scheduled cycle.

    Raises:
        NeuralqxValueError: Options contain an invalid count, choice, path component,
            nonfinite interval, or an incomplete schedule.
    """

    level: Literal["off", "basic", "trace", "full"] = "basic"
    """Capture detail. basic records application events by default, off creates no artifacts, trace captures
    native CPU and accelerator activity, and full additionally enables native Python
    call tracing by default.
    """

    detail: Literal["summary", "detailed", "developer"] = "summary"
    """Depth of package annotations, independent of native collection level.

    summary records major phases, detailed adds operator and estimator stages,
    and developer includes internal dispatch boundaries. User spans are retained
    in every mode. Compiled-operation names remain attached across mode changes.
    """

    output_dir: str = ".neuralqx/profiles"
    """Root directory for rank artifacts and the merged trace."""

    run_id: str | None = None
    """Unique run directory name shared by all ranks. A single-process session generates
    one when omitted. Existing rank directories are never reused.
    """

    transport: Literal["shared", "tcp", "offline"] = "shared"
    """Collection transport. shared reads shared storage, tcp transfers traces to rank zero,
    and offline writes independent artifacts.
    """

    coordinator_address: str | None = None
    """Reachable rank-zero host:port for TCP transport."""

    timeout_s: float = 60.0
    """Deadline in seconds for each coordination or transfer phase."""

    clock_samples: int = 5
    """Round trips per boundary clock calibration."""

    max_events: int = 100_000
    """Total application and telemetry event budget per process. Native XLA buffers are
    controlled by JAX and are not covered.
    """

    max_lanes: int = 1024
    """Maximum application thread and asyncio task lanes."""

    jax_host_tracer_level: int = 1
    """Native host tracing detail from 1 to 3. Level zero cannot supply the clock markers
    needed for native trace alignment.
    """

    jax_python_tracer_level: int | None = None
    """Native Python call tracing, 0 or 1. None selects 1 for full and 0 otherwise."""

    nvtx: bool = False
    """Emit process ranges to NVIDIA tools through the optional nvtx package."""

    telemetry: Literal["off", "auto", "nvml", "nvidia-smi"] = "off"
    """GPU provider: off, auto, nvml, or nvidia-smi. Explicit providers must work. Auto reports
    unavailable providers in the capture metadata.
    """

    telemetry_interval_s: float = 1.0
    """Delay in seconds between background management queries."""

    telemetry_scope: Literal["host", "rank"] = "host"
    """Sampling ownership. host elects the lowest rank per hostname. rank samples separately and
    supports isolated GPU containers.
    """

    telemetry_devices: tuple[str, ...] = ()
    """GPU UUIDs or management indices. Empty selects all GPUs visible to the management
    provider, independently of JAX's logical device numbering and CUDA_VISIBLE_DEVICES.
    """

    nvidia_smi_timeout_s: float = 2.0
    """Timeout in seconds for one read-only management subprocess query."""

    max_trace_bytes: int = 4_294_967_296
    """Limit on each compressed rank trace transferred over TCP."""

    synchronize: Literal["none", "boundary"] = "boundary"
    """Completion policy. boundary waits on supplied results or a wait_for callback. none
    introduces no application completion fences.
    """

    wait_steps: int = 0
    """Unprofiled steps in each scheduled cycle."""

    warmup_steps: int = 0
    """Native warmup steps discarded during export."""

    active_steps: int = 0
    """Recorded steps per cycle. Zero records the whole context."""

    repeat: int = 1
    """Number of capture cycles. Zero repeats until context exit."""

    skip_first: int = 0
    """Unprofiled steps before the first scheduled cycle."""

    def __post_init__(self) -> None:
        for name, choices in (
            ("level", ("off", "basic", "trace", "full")),
            ("detail", ("summary", "detailed", "developer")),
            ("transport", ("shared", "tcp", "offline")),
            ("telemetry", ("off", "auto", "nvml", "nvidia-smi")),
            ("telemetry_scope", ("host", "rank")),
            ("synchronize", ("none", "boundary")),
        ):
            if getattr(self, name) not in choices:
                raise NeuralqxValueError(f"{name} must be one of {choices}.")
        for name in ("clock_samples", "max_events", "max_lanes", "max_trace_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise NeuralqxValueError(f"{name} must be a positive integer.")
        for name in (
            "wait_steps",
            "warmup_steps",
            "active_steps",
            "repeat",
            "skip_first",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise NeuralqxValueError(f"{name} must be a nonnegative integer.")
        for name in ("timeout_s", "telemetry_interval_s", "nvidia_smi_timeout_s"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise NeuralqxValueError(f"{name} must be positive and finite.")
        if type(
            self.jax_host_tracer_level
        ) is not int or self.jax_host_tracer_level not in (
            1,
            2,
            3,
        ):
            raise NeuralqxValueError("jax_host_tracer_level must be 1, 2, or 3.")
        if self.jax_python_tracer_level is not None and (
            type(self.jax_python_tracer_level) is not int
            or self.jax_python_tracer_level not in (0, 1)
        ):
            raise NeuralqxValueError("jax_python_tracer_level must be None, 0, or 1.")
        if type(self.nvtx) is not bool:
            raise NeuralqxValueError("nvtx must be a boolean.")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise NeuralqxValueError("output_dir must be a nonempty string.")
        if self.run_id is not None and (
            type(self.run_id) is not str
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.run_id)
        ):
            raise NeuralqxValueError(
                "run_id must be a path component of at most 128 letters, digits, dots, underscores, or dashes."
            )
        if not isinstance(self.telemetry_devices, tuple) or any(
            type(device) is not str or not device or len(device) > 256
            for device in self.telemetry_devices
        ):
            raise NeuralqxValueError(
                "telemetry_devices must be a tuple of nonempty identifiers."
            )
        if not self.active_steps and (
            self.wait_steps or self.warmup_steps or self.skip_first
        ):
            raise NeuralqxValueError("A step schedule requires positive active_steps.")

    @classmethod
    def from_config(cls, manager: ConfigManager | None = None) -> ProfileOptions:
        """Snapshot effective package options, including thread-local overrides.

        Args:
            manager: Configuration manager. None uses neuralqx.config.

        Returns:
            Validated immutable options. Later config changes affect new sessions.
        """
        if manager is None:
            from neuralqx.config import config

            manager = config
        return cls(
            **{
                field.name: manager.get(f"profile_{field.name}")
                for field in fields(cls)
            }
        )

    def make_schedule(self) -> Schedule | None:
        """Build the configured schedule, or return None for continuous capture.

        Returns:
            A schedule with the configured step counts.
        """
        if not self.active_steps:
            return None
        return Schedule(
            self.wait_steps,
            self.warmup_steps,
            self.active_steps,
            self.repeat,
            self.skip_first,
        )
