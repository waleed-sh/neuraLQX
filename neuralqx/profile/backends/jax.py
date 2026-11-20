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


"""Native CPU and accelerator collection through JAX's public profiler API."""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from ..errors import ProfileError
from ..trace.io import iter_events


class JaxCollector:
    """Collect runtime CPU and accelerator activity through JAX's public API.

    Device execution timestamps come from the native runtime. No application
    call is timed by inserting a completion fence. Clock markers link the
    native trace clock to the process monotonic clock.
    """

    def __init__(self, directory: Path, options):
        """Resolve the local device inventory without starting a collector.

        Args:
            directory: Exclusive directory for one native capture window.
            options: Immutable collection settings selected by the session.
        """
        import jax

        self.jax = jax
        """Imported JAX module providing the public profiler API."""

        self.devices = [
            {
                "id": device.id,
                "process_index": device.process_index,
                "platform": device.platform,
                "device_kind": device.device_kind,
                "local_hardware_id": getattr(device, "local_hardware_id", None),
            }
            for device in jax.local_devices()
        ]
        """Local device inventory including physical IDs when the backend supplies them."""

        self.directory = directory
        """Exclusive output directory for this native capture."""

        self.options = options
        """Immutable settings controlling native tracing detail."""

        self.started = False
        """Whether this instance currently owns a running native collector."""

        self.anchors: dict[str, dict] = {}
        """Marker names mapped to monotonic midpoints and bracket uncertainties."""

        self.path: Path | None = None
        """Exported native JSON trace, or None before a successful stop."""

        self.calibration: dict | None = None
        """Selected clock anchor, cached after the first trace scan."""

    def annotation(self, name: str, *, attributes=None, step_num=None):
        """Construct native context for dispatch and correlation events.

        Args:
            name: Stable application operation name.
            attributes: Bounded Python scalars attached to the native region.
            step_num: Optional optimizer step number for native step grouping.

        Returns:
            A JAX TraceAnnotation or StepTraceAnnotation. Neither waits on arrays.
        """
        values = {} if attributes is None else dict(attributes)
        if step_num is None:
            return self.jax.profiler.TraceAnnotation(name, **values)
        return self.jax.profiler.StepTraceAnnotation(name, step_num=step_num, **values)

    def start(self) -> None:
        """Start native collection and emit its first host clock anchor."""
        options = self.jax.profiler.ProfileOptions()
        options.host_tracer_level = self.options.jax_host_tracer_level
        python_level = self.options.jax_python_tracer_level
        options.python_tracer_level = (
            int(self.options.level == "full") if python_level is None else python_level
        )
        if hasattr(options, "raise_error_on_start_failure"):
            options.raise_error_on_start_failure = True
        self.directory.mkdir(parents=True, exist_ok=False)
        self.jax.profiler.start_trace(
            str(self.directory),
            create_perfetto_link=False,
            create_perfetto_trace=False,
            profiler_options=options,
        )
        self.started = True
        self.anchor()

    def anchor(self) -> None:
        """Bracket a native marker with monotonic timestamps and retain its uncertainty."""
        name = f"__neuralqx_clock_{uuid4().hex}"
        annotation = self.jax.profiler.TraceAnnotation(name)
        before = time.perf_counter_ns()
        annotation.__enter__()
        after = time.perf_counter_ns()
        annotation.__exit__(None, None, None)
        self.anchors[name] = {
            "local_ns": (before + after) // 2,
            "uncertainty_ns": (after - before + 1) // 2,
        }

    def stop(self) -> None:
        """Stop only a collector owned by this instance and locate its exported trace."""
        if not self.started:
            return
        try:
            self.anchor()
        finally:
            try:
                self.jax.profiler.stop_trace()
            finally:
                self.started = False
        paths = sorted(self.directory.rglob("*.trace.json.gz"))
        if len(paths) != 1:
            raise ProfileError(
                f"Expected one JAX trace in {self.directory}, found {len(paths)}."
            )
        self.path = paths[0]

    def calibrate(self) -> dict:
        """Select the narrowest observed native-to-host clock bracket.

        Returns:
            Matching native timestamp in microseconds and the local midpoint and
            uncertainty in nanoseconds.

        Raises:
            ProfileError: Collection has not exported a trace or no marker survived.
        """
        if self.calibration is not None:
            return self.calibration
        if self.path is None:
            raise ProfileError("Stop the native collector before reading its trace.")
        matches = []
        for event in iter_events(self.path):
            anchor = self.anchors.get(event.get("name", ""))
            if anchor is not None and "ts" in event:
                matches.append(
                    {
                        "native_us": event["ts"],
                        **anchor,
                    }
                )
        if not matches:
            raise ProfileError(
                "JAX trace contains no clock anchor. Native activity cannot be aligned reliably."
            )
        self.calibration = min(matches, key=lambda anchor: anchor["uncertainty_ns"])
        return self.calibration

    def events(self, origin_ns: int, begin_ns: int, end_ns: int, window: int):
        """Stream retained native events into the rank's monotonic timeline.

        Args:
            origin_ns: Local monotonic origin of the rank trace.
            begin_ns: Retained interval start, after warmup.
            end_ns: Retained interval end, after any requested boundary wait.
            window: Identifier prefixed onto native correlation and flow IDs.

        Yields:
            Metadata and retained events in microseconds. Complete-duration events
            crossing a boundary are clipped and carry an explicit clipping flag.
        """
        anchor = self.calibrate()
        assert self.path is not None
        anchor_us = (anchor["local_ns"] - origin_ns) / 1000
        begin_us = (begin_ns - origin_ns) / 1000
        end_us = (end_ns - origin_ns) / 1000
        for event in iter_events(self.path):
            if event.get("name") in self.anchors:
                continue
            phase = event.get("ph")
            if phase == "M":
                yield event
                continue
            if "ts" not in event:
                continue
            timestamp = (event["ts"] - anchor["native_us"]) + anchor_us
            if phase == "X":
                finish = timestamp + event.get("dur", 0)
                if finish < begin_us or timestamp > end_us:
                    continue
                event["dur"] = max(0.0, min(finish, end_us) - max(timestamp, begin_us))
                event["ts"] = max(timestamp, begin_us)
                if timestamp < begin_us or finish > end_us:
                    event.setdefault("args", {})["neuralqx_clipped_at_boundary"] = True
            else:
                if not begin_us <= timestamp <= end_us:
                    continue
                event["ts"] = timestamp
            # XLA may restart correlation numbering on every collector start.
            for field in ("id", "bind_id"):
                if field in event:
                    event[field] = f"w{window}:{event[field]}"
            if "id2" in event:
                event["id2"] = {
                    key: f"w{window}:{value}" for key, value in event["id2"].items()
                }
            yield event
