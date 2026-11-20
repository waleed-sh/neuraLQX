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


"""Merge rank traces while preserving device lanes, flow links, and clock evidence."""

from __future__ import annotations

import math
from collections.abc import Iterable
from collections.abc import Mapping
from pathlib import Path

from ..distributed.clock import ClockMap
from ..errors import ProfileError
from .io import iter_events
from .io import read_json
from .io import write_trace


def _load_manifest(path: Path) -> dict:
    try:
        entry = read_json(path)
    except (ValueError, UnicodeError) as error:
        raise ProfileError(f"Malformed profiling manifest: {path}.") from error
    required = {"run_id", "count", "index", "hostname", "origin_ns", "clock", "schema"}
    if not isinstance(entry, dict) or not required <= entry.keys():
        raise ProfileError(f"Profiling manifest is missing required fields: {path}.")
    if not isinstance(entry["run_id"], str) or not isinstance(entry["hostname"], str):
        raise ProfileError("Manifest run and host names must be strings.")
    for field in ("count", "index", "origin_ns"):
        if type(entry[field]) is not int:
            raise ProfileError(f"Manifest {field} must be an integer.")
    clock = entry["clock"]
    if not isinstance(clock, dict) or clock.get("method") not in (
        "local",
        "unmapped",
        "manual",
        "shared_round_trip",
        "tcp_round_trip",
    ):
        raise ProfileError("Unsupported profiling clock metadata.")
    for phase in ("start", "end"):
        sample = clock.get(phase)
        if not isinstance(sample, dict) or set(sample) != {
            "local_ns",
            "offset_ns",
            "uncertainty_ns",
        }:
            raise ProfileError("Incomplete profiling clock sample.")
        if (
            any(type(value) is not int for value in sample.values())
            or sample["uncertainty_ns"] < 0
        ):
            raise ProfileError("Invalid profiling clock sample.")
    return entry


def merge_traces(
    manifests: Iterable[str | Path],
    output_path: str | Path,
    *,
    allow_partial: bool = False,
    clock_offsets_ns: Mapping[int, int] | None = None,
) -> Path:
    """Stream completed rank captures into one Perfetto-compatible trace.

    Process and thread identifiers are remapped without collisions. Native
    flow and async identifiers are namespaced by rank while keeping links
    between host dispatch and device streams. Boundary clock samples supply
    offset and linear drift correction. Their round-trip uncertainty is stored
    in the output, so the merged timeline does not imply exact cross-host order.

    Args:
        manifests: Paths to rank manifest.json files. Each associated
            trace.json.gz must be in the same directory. Files can be copied
            from node-local storage before merging.
        output_path: Final .json or .json.gz path, written atomically.
        allow_partial: Permit absent ranks and label the output incomplete.
        clock_offsets_ns: Optional externally measured offsets to add to each
            rank's monotonic clock to obtain one common clock. Required for
            ranks captured with offline transport, which performs no exchange.
            Explicit offsets override captured clock samples for those ranks.

    Returns:
        Absolute path of the merged trace.

    Raises:
        ProfileError: Ranks are duplicated, sessions differ, clocks cannot be
            aligned, a trace is malformed, or a required rank is missing.
        OSError: An input cannot be read or the output cannot be committed.
    """
    paths = [Path(path).resolve() for path in manifests]
    if not paths:
        raise ProfileError("No profiling manifests were supplied.")
    entries = [(path, _load_manifest(path)) for path in paths]
    run_ids = {entry["run_id"] for _, entry in entries}
    counts = {entry["count"] for _, entry in entries}
    indices = [entry["index"] for _, entry in entries]
    if len(run_ids) != 1 or len(counts) != 1:
        raise ProfileError("Cannot merge captures from different runs or world sizes.")
    count = next(iter(counts))
    if (
        type(count) is not int
        or count < 1
        or any(type(index) is not int or not 0 <= index < count for index in indices)
    ):
        raise ProfileError("Invalid rank index or world size in a profiling manifest.")
    if len(set(indices)) != len(indices):
        raise ProfileError("Duplicate profiling rank manifests.")
    missing = sorted(set(range(count)) - set(indices))
    if missing and not allow_partial:
        raise ProfileError(f"Missing profiling ranks: {missing}.")
    output = Path(output_path).resolve()
    if output in paths or any(
        output == path.parent / "trace.json.gz" for path in paths
    ):
        raise ProfileError("The merged output cannot overwrite a rank artifact.")
    clocks = {}
    for _, entry in entries:
        rank = entry["index"]
        if entry.get("schema") != "neuralqx.profile.rank.v1":
            raise ProfileError("Unsupported rank manifest schema.")
        clock = entry["clock"]
        if clock_offsets_ns is not None and rank in clock_offsets_ns:
            offset = clock_offsets_ns[rank]
            if type(offset) is not int:
                raise ProfileError("Manual clock offsets must be integer nanoseconds.")
            sample = {
                "local_ns": entry["origin_ns"],
                "offset_ns": offset,
                "uncertainty_ns": 0,
            }
            clock = {
                "method": "manual",
                "start": sample,
                "end": sample,
                "uncertainty_known": False,
            }
        elif clock["method"] == "unmapped":
            raise ProfileError(
                f"Rank {rank} has no common clock. Supply clock_offsets_ns for offline merging."
            )
        clocks[rank] = ClockMap(clock["start"], clock["end"])
        entry["clock"] = clock
    origin = min(
        clocks[entry["index"]].root_ns(entry["origin_ns"]) for _, entry in entries
    )
    entries.sort(key=lambda pair: pair[1]["index"])
    metadata = {
        "schema": "neuralqx.profile.trace.v1",
        "run_id": next(iter(run_ids)),
        "world_size": count,
        "missing_ranks": missing,
        "complete": not missing
        and all(entry.get("status") == "ok" for _, entry in entries),
        "data_loss": any(
            entry.get("recording", {}).get("dropped_events", 0) for _, entry in entries
        ),
        "clock_origin_ns": origin,
        "clock_uncertainty_ns": {
            str(entry["index"]): (
                None
                if entry["clock"]["method"] == "manual"
                else clocks[entry["index"]].uncertainty_ns
            )
            for _, entry in entries
        },
        "ranks": [entry for _, entry in entries],
    }

    def merged_events():
        next_pid = 1
        next_tid = 1
        next_flow = 1
        for path, entry in entries:
            rank = entry["index"]
            clock = clocks[rank]
            pids = {}
            tids = {}
            flows: dict[tuple, int] = {}
            for source in iter_events(path.parent / "trace.json.gz"):
                event = dict(source)
                old_pid, old_tid = source.get("pid", 0), source.get("tid", 0)
                if old_pid not in pids:
                    pids[old_pid] = next_pid
                    next_pid += 1
                key = (old_pid, old_tid)
                if key not in tids:
                    tids[key] = next_tid
                    next_tid += 1
                event["pid"], event["tid"] = pids[old_pid], tids[key]
                if "ts" in event:
                    timestamp = event["ts"]
                    if not isinstance(timestamp, (int, float)) or not math.isfinite(
                        timestamp
                    ):
                        raise ProfileError("A trace timestamp is not finite.")
                    event["ts"] = clock.relative_us(
                        entry["origin_ns"], timestamp, origin
                    )
                if "dur" in event:
                    duration = event["dur"]
                    if (
                        not isinstance(duration, (int, float))
                        or not math.isfinite(duration)
                        or duration < 0
                    ):
                        raise ProfileError("A trace duration is invalid.")
                    event["dur"] = duration * (1 + clock.slope)
                for field in ("id", "bind_id"):
                    if field in event:
                        flow_key = ("global", event.get("cat", ""), str(event[field]))
                        if flow_key not in flows:
                            flows[flow_key] = next_flow
                            next_flow += 1
                        event[field] = flows[flow_key]
                if "id2" in event:
                    mapped = {}
                    for scope, identifier in event["id2"].items():
                        scoped_key = (
                            scope,
                            old_pid if scope == "local" else None,
                            event.get("cat", ""),
                            str(identifier),
                        )
                        if scoped_key not in flows:
                            flows[scoped_key] = next_flow
                            next_flow += 1
                        mapped[scope] = hex(flows[scoped_key])
                    event["id2"] = mapped
                if event.get("ph") == "M":
                    args = dict(event.get("args", {}))
                    if event.get("name") == "process_name":
                        args["name"] = (
                            f"rank {rank} @ {entry['hostname']} / {args.get('name', old_pid)}"
                        )
                    elif event.get("name") == "process_sort_index":
                        args["sort_index"] = pids[old_pid]
                    event["args"] = args
                yield event

    write_trace(output, merged_events(), metadata)
    return output
