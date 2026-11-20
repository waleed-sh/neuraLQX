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


"""Capture-window state and streaming normalization of rank artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends.base import ActivityCollector
from .trace.io import write_trace


@dataclass(slots=True)
class Window:
    """One native capture cycle and its retained application-time interval."""

    index: int
    """Zero-based capture identifier used to namespace native correlation IDs."""

    collector: ActivityCollector | None = None
    """Native collector for this cycle, or None for application-only capture."""

    begin_ns: int | None = None
    """Monotonic start of retained recording, excluding native warmup."""

    end_ns: int | None = None
    """Monotonic end of retained recording, set when the window closes."""

    completion_waited: bool = False
    """Whether explicitly supplied output arrays were waited on at the boundary."""


def export_rank(session, path: Path) -> tuple[list[dict], list[str]]:
    """Stream application, telemetry, and native events into one rank trace.

    Args:
        session: Stopped session whose recorder is no longer accepting events.
        path: Destination trace path, with .gz selecting compression.

    Returns:
        Per-window metadata and native-alignment failure descriptions. Failed
        windows remain in metadata even when their native events cannot be aligned.
    """
    windows = []
    failures = []
    ready = []
    for window in session._windows:
        detail = {
            "index": window.index,
            "begin_ns": window.begin_ns,
            "end_ns": window.end_ns,
            "completion_waited": window.completion_waited,
            "native_events": 0,
        }
        collector = window.collector
        if collector is not None:
            detail["devices"] = collector.devices
        if (
            collector is not None
            and window.begin_ns is not None
            and collector.path is not None
        ):
            try:
                detail["native_clock_anchor"] = collector.calibrate()
                detail["native_trace"] = str(
                    collector.path.relative_to(session._rank_dir)
                )
                ready.append((window, detail))
            except Exception as error:
                message = (
                    f"Native trace alignment failed: {type(error).__name__}: {error}"
                )
                failures.append(message)
                detail["error"] = message
        windows.append(detail)

    def raw_events():
        yield {
            "ph": "M",
            "name": "process_name",
            "pid": 1,
            "args": {"name": "application"},
        }
        for tid, name in session._recorder.lanes.items():
            yield {
                "ph": "M",
                "name": "thread_name",
                "pid": 1,
                "tid": tid,
                "args": {"name": name},
            }
        for uuid in session._telemetry_devices:
            yield {
                "ph": "M",
                "name": "process_name",
                "pid": f"telemetry:{uuid}",
                "args": {"name": f"GPU {uuid} management counters"},
            }
        yield from session._recorder.events()
        stats = session._recorder.stats()
        if stats["dropped_events"]:
            end_ns = max(
                (window.end_ns or session._origin_ns for window in session._windows),
                default=session._origin_ns,
            )
            yield {
                "ph": "C",
                "name": "profiler_dropped_events",
                "cat": "profiler.diagnostics",
                "pid": 1,
                "tid": 0,
                "ts": (end_ns - session._origin_ns) / 1000,
                "args": {"count": stats["dropped_events"]},
            }
        for window, detail in ready:
            for source in window.collector.events(
                session._origin_ns, window.begin_ns, window.end_ns, window.index
            ):
                event = dict(source)
                event["pid"] = ("jax", event.get("pid", 0))
                if event.get("ph") != "M":
                    detail["native_events"] += 1
                yield event

    def normalized_events():
        pids: dict[Any, int] = {}
        seen = set()
        for event in raw_events():
            key = event.get("pid", 0)
            event["pid"] = pids.setdefault(key, len(pids) + 1)
            if event.get("ph") == "M":
                identity = (event.get("name"), event["pid"], event.get("tid", 0))
                if identity in seen:
                    continue
                seen.add(identity)
            yield event

    write_trace(
        path,
        normalized_events(),
        {"schema": "neuralqx.profile.local.v1", "origin_ns": session._origin_ns},
    )
    return windows, failures
