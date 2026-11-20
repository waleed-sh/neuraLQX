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


"""Explicit capture sessions with optional native and distributed collection."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from uuid import uuid4

from neuralqx.utils.errors import NeuralqxValueError

from . import _state
from ._capture import Window
from ._capture import export_rank
from ._recorder import Recorder
from .distributed.common import LocalCoordinator
from .distributed.common import Rank
from .distributed.common import rank_directory
from .errors import ProfileError
from .errors import ProfileStateError
from .options import ProfileOptions
from .schedule import Action
from .schedule import Schedule
from .trace.io import write_json
from .trace.merge import merge_traces

_UNSET = object()

if TYPE_CHECKING:
    from .backends.nvtx import NvtxEmitter
    from .telemetry.sampler import TelemetrySampler


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Artifacts produced when a session closes.

    Attributes:
        run_dir: Local run directory, or None for a disabled session.
        rank_trace: Completed local trace containing application and native events.
        manifest_path: Rank metadata and clock calibration used for merging.
        trace_path: Merged trace on rank zero. None on workers, for offline
            multi-process sessions, and for disabled sessions.
        rank: Process index in the profiled workload.
        dropped_events: Application and telemetry records omitted by buffer limits.
    """

    run_dir: Path | None
    """Local run directory, or None when profiling was disabled."""

    rank_trace: Path | None
    """Completed local event trace, or None when profiling was disabled."""

    manifest_path: Path | None
    """Committed rank metadata and clock calibration, or None when disabled."""

    trace_path: Path | None
    """Merged trace on rank zero. None on workers, offline multi-rank runs, or when off."""

    rank: int
    """Process index in the captured workload."""

    dropped_events: int = 0
    """Application and telemetry records omitted by configured buffer limits."""


class Profiler:
    """Own one explicit profiling session in the current process.

    No collectors, output directories, or hooks are created at construction.
    Settings are frozen on entry. Application spans are process-wide, so worker
    threads participate without copying context variables. Join workers and
    await tasks before leaving the context. Sessions cannot overlap or nest.

    In distributed jobs all ranks must enter the same run_id with their actual
    world size. shared transport requires shared output storage. tcp requires
    a reachable rank-zero coordinator_address and supports node-local output.
    Only rank zero writes the combined trace. Collection uses no JAX or MPI
    collectives and never adds communication to individual application spans.

    Args:
        output_dir: Optional override of profile_output_dir.
        level: Optional override of profile_level.
        run_id: Optional unique shared run identifier.
        options: Complete immutable settings. None snapshots package config
            on entry, including active thread-local overrides.
        schedule: Step schedule overriding configured step counts. None uses
            the configured schedule or records the whole context.
        wait_for: Callback returning a pytree of current result arrays at each
            capture boundary. With synchronize='boundary', those results are
            blocked once at a boundary. Supply every independent result whose
            device activity must finish inside the trace. This is not a global
            device fence and does not wait for unrelated computations or JAX
            effects. The callback is never called for off or synchronize='none'.
        rank: Explicit process index for non-JAX workers. Normally detected
            from the initialized JAX runtime.
        world_size: Process count, required together with an explicit rank.

    Raises:
        ProfileStateError: A session is reused, nested, or controlled from a
            different thread or forked child.
        ProfileError: A requested collector or distributed collection fails.
        NeuralqxValueError: Settings or rank identity are invalid.

    Examples:
        Capture device work without waiting after every function::

            with Profiler(level="trace", wait_for=lambda: result) as profiler:
                result = compiled_update(parameters)

        Record selected iterations and supply their completion values::

            with Profiler(level="trace", schedule=Schedule(warmup=2, active=4)) as profiler:
                for iteration in range(20):
                    result = compiled_update(parameters)
                    profiler.step(wait_for=result)
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        level: str | None = None,
        run_id: str | None = None,
        options: ProfileOptions | None = None,
        schedule: Schedule | None = None,
        wait_for: Callable[[], Any] | None = None,
        rank: int | None = None,
        world_size: int | None = None,
    ):
        """Store explicit overrides and initialize a single-use, inactive controller."""
        self._overrides: dict[str, Any] = {
            key: value
            for key, value in (
                ("output_dir", str(output_dir) if output_dir is not None else None),
                ("level", level),
                ("run_id", run_id),
            )
            if value is not None
        }
        self._supplied_options = options
        self._supplied_schedule = schedule
        self._wait_for = wait_for
        self._rank_override, self._world_override = rank, world_size
        self._state = "new"
        self._owner_pid = os.getpid()
        self._owner_thread: int | None = None
        self._recording_window: Window | None = None
        self._window: Window | None = None
        self._windows: list[Window] = []
        self._recorder: Recorder
        self._coordinator: LocalCoordinator | None = None
        self._sampler: TelemetrySampler | None = None
        self._telemetry_status: list[dict] = []
        self._telemetry_devices: set[str] = set()
        self._nvtx: NvtxEmitter | None = None
        self._annotation_failures: list[str] = []
        self._last_result = _UNSET
        self._step = 0
        self._step_begin_ns: int | None = None
        self._action = Action.WAIT
        self._cycle = -1
        self._rank_dir: Path
        self.options: ProfileOptions
        """Validated settings snapshot, available after start freezes configuration."""

        self.result: ProfileResult | None = None
        """Completed artifact paths and counts, or None before the session finishes."""

    def __enter__(self) -> Profiler:
        """Start collection and return this controller."""
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> Literal[False]:
        """Close collection while preserving an exception raised by the application."""
        try:
            self._finish(exception=exc)
        except Exception as error:
            if exc is None:
                raise
            exc.add_note(f"Profiler cleanup also failed: {error}")
        return False

    def _check_owner(self) -> None:
        if (
            os.getpid() != self._owner_pid
            or threading.get_ident() != self._owner_thread
        ):
            raise ProfileStateError(
                "Control a profiler only from the thread and process that started it."
            )

    def start(self) -> Profiler:
        """Start the configured session and return its controller.

        Returns:
            This profiler. A disabled session starts no runtime resources.

        Raises:
            ProfileStateError: The session has already been started.
            ProfileError: Collector startup or distributed rendezvous fails.
        """
        if self._state != "new" or os.getpid() != self._owner_pid:
            raise ProfileStateError(
                "Profiler instances are single-use and cannot cross a fork."
            )
        self._owner_thread = threading.get_ident()
        self.options = replace(
            self._supplied_options or ProfileOptions.from_config(), **self._overrides
        )
        self._detail_level = {"summary": 0, "detailed": 1, "developer": 2}[
            self.options.detail
        ]
        self._schedule = self._supplied_schedule or self.options.make_schedule()
        self._state = "running"
        if self.options.level == "off":
            return self
        try:
            _state.claim(self)
            self._rank = Rank.detect(self._rank_override, self._world_override)
            if self._rank.count > 1 and self.options.run_id is None:
                raise NeuralqxValueError(
                    "Distributed profiling requires the same explicit run_id on every rank."
                )
            if (
                self._rank.count > 1
                and self.options.transport == "offline"
                and self.options.telemetry != "off"
                and self.options.telemetry_scope == "host"
            ):
                raise NeuralqxValueError(
                    "Offline multi-process telemetry requires telemetry_scope='rank'."
                )
            self._run_id = self.options.run_id or uuid4().hex
            self._run_dir = (
                Path(self.options.output_dir).expanduser().resolve() / self._run_id
            )
            self._rank_dir = rank_directory(self._run_dir, self._rank.index)
            self._rank_dir.mkdir(parents=True, exist_ok=False)
            self._origin_ns = time.perf_counter_ns()
            hello = {
                **self._rank.to_dict(),
                "run_id": self._run_id,
                "origin_ns": self._origin_ns,
                "wall_time_ns": time.time_ns(),
                "clock_samples": self.options.clock_samples,
            }
            coordinator_type = LocalCoordinator
            if self._rank.count > 1 and self.options.transport == "shared":
                from .distributed.shared import SharedCoordinator

                coordinator_type = SharedCoordinator
            elif self._rank.count > 1 and self.options.transport == "tcp":
                from .distributed.tcp import TcpCoordinator

                coordinator_type = TcpCoordinator
            self._coordinator = coordinator_type(
                self._run_dir, self._rank, self.options, hello
            )
            self._coordinator.prepare()
            self._recorder = Recorder(self.options.max_events, self.options.max_lanes)
            if self.options.nvtx:
                from .backends.nvtx import NvtxEmitter

                self._nvtx = NvtxEmitter()
            action = (
                Action.RECORD if self._schedule is None else self._schedule.action(0)
            )
            cycle = 0 if self._schedule is None else self._schedule.cycle(0)
            self._transition(action, cycle)
        except BaseException as error:
            self._recording_window = None
            try:
                self._close_collectors()
            except BaseException as cleanup:
                error.add_note(f"Collector startup cleanup failed: {cleanup}")
            try:
                if self._coordinator is not None:
                    self._coordinator.close()
            except BaseException as cleanup:
                error.add_note(f"Coordinator startup cleanup failed: {cleanup}")
            finally:
                _state.release(self)
                self._state = "failed"
            raise
        return self

    def _open_window(self) -> None:
        assert self._coordinator is not None
        window = Window(len(self._windows))
        self._windows.append(window)
        self._window = window
        if self.options.level in ("trace", "full"):
            from .backends.jax import JaxCollector

            window.collector = JaxCollector(
                self._rank_dir / "native" / f"capture-{window.index:05d}", self.options
            )
            window.collector.start()
        if self.options.telemetry != "off" and (
            self.options.telemetry_scope == "rank"
            or self._coordinator.telemetry_leader()
        ):
            from .telemetry.sampler import TelemetrySampler

            self._sampler = TelemetrySampler(self.options, self._emit_telemetry)
            self._sampler.start()

    def _fence(self) -> bool:
        if self.options.synchronize == "none":
            return False
        value = self._wait_for() if self._wait_for is not None else self._last_result
        if value is _UNSET:
            return False
        import jax

        before = time.perf_counter_ns()
        jax.block_until_ready(value)
        after = time.perf_counter_ns()
        if self._recording_window is not None:
            lane, _ = self._recorder.lane()
            if lane is not None:
                self._recorder.append(
                    {
                        "name": "wait_for_completion",
                        "cat": "profiler.synchronization",
                        "ph": "X",
                        "pid": 1,
                        "tid": lane,
                        "ts": (before - self._origin_ns) / 1000,
                        "dur": (after - before) / 1000,
                        "args": {"explicit_results": True},
                    }
                )
        return True

    def _close_collectors(self) -> None:
        if self._sampler is not None:
            self._sampler.request_stop()
        try:
            if self._window is not None and self._window.collector is not None:
                self._window.collector.stop()
        finally:
            if self._sampler is not None:
                self._sampler.stop()
                self._telemetry_status.append(self._sampler.status())
                self._sampler = None

    def _close_window(self) -> None:
        if self._window is None:
            return
        try:
            self._window.completion_waited = self._fence()
        finally:
            self._window.end_ns = time.perf_counter_ns()
            self._recording_window = None
            try:
                self._close_collectors()
            finally:
                self._window = None
                self._last_result = _UNSET

    def _transition(self, action: Action, cycle: int) -> None:
        if self._window is not None and (action is Action.WAIT or cycle != self._cycle):
            self._close_window()
        if action is not Action.WAIT and self._window is None:
            if self._step > 0 and self._action is Action.WAIT:
                self._fence()
            self._open_window()
        if action is Action.RECORD and self._recording_window is None:
            assert self._window is not None
            if self._action is Action.WARMUP:
                self._fence()
            self._window.begin_ns = time.perf_counter_ns()
            self._recording_window = self._window
            self._step_begin_ns = self._window.begin_ns
        self._action, self._cycle = action, cycle

    def _record_step(self, end_ns: int) -> None:
        if self._recording_window is None or self._step_begin_ns is None:
            return
        lane, _ = self._recorder.lane()
        if lane is not None:
            self._recorder.append(
                {
                    "name": "step",
                    "cat": "profiler.step",
                    "ph": "X",
                    "pid": 1,
                    "tid": lane,
                    "ts": (self._step_begin_ns - self._origin_ns) / 1000,
                    "dur": (end_ns - self._step_begin_ns) / 1000,
                    "args": {"step": self._step},
                }
            )

    def step(self, *, wait_for: Any = _UNSET) -> None:
        """Finish one iteration and advance the capture schedule.

        Args:
            wait_for: Current result pytree. It replaces the previous reference
                and is blocked only if this step ends or begins a capture
                window, or at final exit. A constructor callback takes priority.
                Prefer that callback when results are later donated to JAX.

        Raises:
            ProfileStateError: The session is not running in its owner thread.
        """
        self._check_owner()
        if self._state != "running":
            raise ProfileStateError("step requires a running profiler.")
        if self.options.level == "off":
            return
        if wait_for is not _UNSET:
            self._last_result = wait_for
        self._record_step(time.perf_counter_ns())
        self._step += 1
        action = (
            Action.RECORD
            if self._schedule is None
            else self._schedule.action(self._step)
        )
        cycle = 0 if self._schedule is None else self._schedule.cycle(self._step)
        self._transition(action, cycle)
        self._step_begin_ns = (
            time.perf_counter_ns() if self._recording_window is not None else None
        )

    def _instant(self, name, category, attributes, phase="i") -> None:
        lane, _ = self._recorder.lane()
        if lane is None:
            self._recorder.drop_lane()
            return
        record = {
            "name": name,
            "cat": category,
            "ph": phase,
            "pid": 1,
            "tid": lane,
            "ts": (time.perf_counter_ns() - self._origin_ns) / 1000,
            "args": attributes,
        }
        if phase == "i":
            record["s"] = "t"
        self._recorder.append(record)

    def _emit_telemetry(self, reading, timestamp_ns, query_ns) -> None:
        window = self._recording_window
        if window is None or timestamp_ns < window.begin_ns:
            return
        self._telemetry_devices.add(reading.uuid)
        values = dict(reading.values, query_duration_ns=query_ns)
        for name, value in values.items():
            self._recorder.append(
                {
                    "name": name,
                    "cat": "gpu.telemetry",
                    "ph": "C",
                    "pid": f"telemetry:{reading.uuid}",
                    "tid": 0,
                    "ts": (timestamp_ns - self._origin_ns) / 1000,
                    "args": {"value": value},
                }
            )

    def stop(self, *, wait_for: Any = _UNSET) -> ProfileResult:
        """Stop collectors, commit rank artifacts, and merge on rank zero.

        Repeated calls return the same result after successful completion.
        Completed rank artifacts are kept if later collection or merging fails.
        No atexit handler or finalizer performs distributed communication.

        Args:
            wait_for: Final result pytree to complete before stopping native
                capture when boundary synchronization is enabled.

        Returns:
            Paths to local artifacts and, on rank zero, the merged trace.

        Raises:
            ProfileStateError: The controller is not running or its owner differs.
            ProfileError: Collection failed. Local artifacts remain for inspection.
        """
        return self._finish(wait_for=wait_for)

    def _finish(self, *, wait_for=_UNSET, exception=None) -> ProfileResult:
        self._check_owner()
        if self._state == "stopped":
            assert self.result is not None
            return self.result
        if self._state != "running":
            raise ProfileStateError("stop requires a running profiler.")
        if self.options.level == "off":
            self.result = ProfileResult(None, None, None, None, 0)
            self._state = "stopped"
            return self.result
        if wait_for is not _UNSET:
            self._last_result = wait_for
        assert self._coordinator is not None
        failures = list(self._annotation_failures)
        try:
            try:
                self._close_window()
            except Exception as error:
                failures.append(
                    f"Collector shutdown failed: {type(error).__name__}: {error}"
                )
            self._recorder.accepting = False
            stats = self._recorder.stats()
            if stats["unfinished_spans"]:
                failures.append(
                    "Application spans are still open. Join threads and await tasks before closing the profiler."
                )
            local_trace = self._rank_dir / "trace.json.gz"
            windows, native_failures = export_rank(self, local_trace)
            failures.extend(native_failures)
            for telemetry in self._telemetry_status:
                if self.options.telemetry != "auto":
                    failures.extend(telemetry["failures"])
            manifest = {
                "schema": "neuralqx.profile.rank.v1",
                **self._coordinator.hello,
                "status": "error" if failures or exception is not None else "ok",
                "workload_exception": (
                    None if exception is None else type(exception).__name__
                ),
                "options": asdict(self.options),
                "recording": stats,
                "windows": windows,
                "telemetry": self._telemetry_status,
                "errors": failures,
            }
            write_json(self._rank_dir / "manifest.pending.json", manifest)
            manifests = self._coordinator.collect(manifest)
            (self._rank_dir / "manifest.pending.json").unlink(missing_ok=True)
            merged = (
                None
                if manifests is None
                else merge_traces(manifests, self._run_dir / "trace.json.gz")
            )
            self.result = ProfileResult(
                self._run_dir,
                local_trace,
                self._rank_dir / "manifest.json",
                merged,
                self._rank.index,
                stats["dropped_events"],
            )
            self._state = "stopped"
            if failures:
                raise ProfileError(
                    " | ".join(failures) + f" Artifacts: {self._rank_dir}"
                )
            return self.result
        except BaseException:
            if self._state != "stopped":
                self._state = "failed"
            raise
        finally:
            self._recording_window = None
            self._last_result = _UNSET
            try:
                self._coordinator.close()
            finally:
                _state.release(self)
