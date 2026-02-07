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


from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from pathlib import Path

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ._rank import RankInfo

# generate a stable-ish run id without requiring MPI communication
# users can override with NQX_PROFILE_RUN_ID to force a single directory across ranks

_START_EPOCH_S = int(time.time())
_JOB_ID = (
    os.environ.get("SLURM_JOB_ID")
    or os.environ.get("PBS_JOBID")
    or os.environ.get("LSB_JOBID")
    or ""
)
_RUN_ID = (
    os.environ.get("NQX_PROFILE_RUN_ID") or f"{_JOB_ID or 'local'}_{_START_EPOCH_S}"
)


def run_id() -> str:
    return _RUN_ID


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), encoding="utf-8"
    ) as f:
        tmp = f.name
        f.write(text)
    os.replace(tmp, str(path))


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=False))


def _safe_path_component(s: str) -> str:
    # keep filenames safe on shared filesystems
    return "".join(c if c.isalnum() or c in "._-=" else "_" for c in s)[:120]


def get_run_dir(output_dir: str, rank_info: RankInfo) -> Path:
    root = Path(output_dir)
    rid = _safe_path_component(run_id())
    d = root / f"run_{rid}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def export_summary_json(
    output_dir: str, rank_info: RankInfo, summary: Dict[str, Any]
) -> Path:
    run_dir = get_run_dir(output_dir, rank_info)
    p = run_dir / f"summary_rank{rank_info.rank:06d}.json"
    _atomic_write_json(p, summary)
    return p


def export_trace_json(
    output_dir: str, rank_info: RankInfo, trace_events: List[Dict[str, Any]]
) -> Path:
    run_dir = get_run_dir(output_dir, rank_info)
    p = run_dir / f"trace_rank{rank_info.rank:06d}.json"
    payload = {
        "displayTimeUnit": "us",
        "traceEvents": trace_events,
    }
    _atomic_write_json(p, payload)
    return p


def export_metrics(
    output_dir: str, rank_info: RankInfo, samples: List[Dict[str, Any]]
) -> Optional[Path]:
    if not samples:
        return None
    run_dir = get_run_dir(output_dir, rank_info)
    p = run_dir / f"metrics_rank{rank_info.rank:06d}.csv"
    # flatten a conservative subset, keep GPU as JSON in one column
    fieldnames = [
        "t_wall_s",
        "proc_cpu_percent",
        "proc_rss",
        "proc_vms",
        "proc_read_bytes",
        "proc_write_bytes",
        "sys_cpu_percent",
        "sys_mem_total",
        "sys_mem_used",
        "sys_mem_available",
        "gpus_json",
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(run_dir), encoding="utf-8", newline=""
    ) as f:
        tmp = f.name
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            proc = s.get("process", {}) or {}
            sys = s.get("system", {}) or {}
            row = {
                "t_wall_s": s.get("t_wall_s"),
                "proc_cpu_percent": proc.get("cpu_percent"),
                "proc_rss": proc.get("rss"),
                "proc_vms": proc.get("vms"),
                "proc_read_bytes": proc.get("read_bytes"),
                "proc_write_bytes": proc.get("write_bytes"),
                "sys_cpu_percent": sys.get("cpu_percent"),
                "sys_mem_total": sys.get("mem_total"),
                "sys_mem_used": sys.get("mem_used"),
                "sys_mem_available": sys.get("mem_available"),
                "gpus_json": json.dumps(s.get("gpus", [])),
            }
            writer.writerow(row)
    os.replace(tmp, str(p))
    return p


def export_readme(output_dir: str, rank_info: RankInfo) -> Path:
    run_dir = get_run_dir(output_dir, rank_info)
    p = run_dir / "README.txt"
    if p.exists():
        return p
    text = (
        "neuralqx profiling output\n"
        "\n"
        "Files:\n"
        "  summary_rankXXXXXX.json : hierarchical section timings (ns)\n"
        "  trace_rankXXXXXX.json   : Chrome/Perfetto trace (us) - open in Perfetto UI or chrome://tracing\n"
        "  metrics_rankXXXXXX.csv  : optional low-rate telemetry (CPU/mem + NVML GPU stats)\n"
        "\n"
        "Viewing traces:\n"
        "  - Perfetto UI (recommended) can load trace_rank*.json\n"
        "  - chrome://tracing can also load trace JSON\n"
        "\n"
        "Nsight Systems/Compute:\n"
        "  If NVTX is enabled, regions will appear automatically.\n"
        "\n"
        "Notes:\n"
        "  - Per-rank output avoids shared-file contention on HPC.\n"
        "  - MPI aggregation at exit is disabled by default (to avoid blocking).\n"
        "    You can aggregate summaries offline by reading the per-rank JSON files.\n"
    )
    _atomic_write_text(p, text)
    return p


def export_all(
    output_dir: str,
    rank_info: RankInfo,
    summary: Dict[str, Any],
    trace_events: List[Dict[str, Any]],
    metrics_samples: List[Dict[str, Any]],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    export_readme(output_dir, rank_info)
    out["summary"] = str(export_summary_json(output_dir, rank_info, summary))
    if trace_events:
        out["trace"] = str(export_trace_json(output_dir, rank_info, trace_events))
    mp = export_metrics(output_dir, rank_info, metrics_samples)
    if mp is not None:
        out["metrics"] = str(mp)
    return out
