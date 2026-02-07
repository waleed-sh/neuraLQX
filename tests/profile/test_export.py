#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import csv
import importlib
import json
from pathlib import Path

import pytest

from neuralqx.profile._rank import RankInfo


def _rankinfo(rank=0, size=1, local_rank=0) -> RankInfo:
    return RankInfo(
        rank=rank,
        size=size,
        local_rank=local_rank,
        hostname="host",
        pid=1234,
        backend="serial",
    )


def test_safe_path_component_sanitizes_and_truncates():
    from neuralqx.profile import _export

    s = "abc/def:ghi jkl\t\n*?<>|🙂" + ("x" * 1000)
    out = _export._safe_path_component(s)
    assert "/" not in out
    assert ":" not in out
    assert " " not in out
    assert len(out) <= 120
    assert out


def test_get_run_dir_creates_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("NQX_PROFILE_RUN_ID", "run/unsafe:id")
    import neuralqx.profile._export as exp

    exp = importlib.reload(exp)

    outdir = tmp_path / "profiles"
    ri = _rankinfo()
    d = exp.get_run_dir(str(outdir), ri)
    assert d.exists()
    assert d.is_dir()

    assert d.name.startswith("run_")
    assert "unsafe" in d.name
    assert "/" not in d.name
    assert ":" not in d.name


def test_atomic_write_text(tmp_path):
    from neuralqx.profile import _export

    p = tmp_path / "a" / "b" / "x.txt"
    _export._atomic_write_text(p, "hello")
    assert p.exists()
    assert p.read_text(encoding="utf-8") == "hello"


def test_export_summary_and_trace_json(tmp_path, monkeypatch):
    monkeypatch.setenv("NQX_PROFILE_RUN_ID", "testrun")
    import neuralqx.profile._export as exp

    exp = importlib.reload(exp)

    ri = _rankinfo(rank=7)
    outdir = str(tmp_path / "profiles")

    summary = {"schema": "s", "root": {"name": "root", "stats": {"inclusive_ns": 1}}}
    sp = exp.export_summary_json(outdir, ri, summary)
    assert sp.exists()
    j = json.loads(sp.read_text(encoding="utf-8"))
    assert j["schema"] == "s"

    trace_events = [{"name": "X", "ph": "X", "ts": 0, "dur": 1, "pid": 7, "tid": 0}]
    tp = exp.export_trace_json(outdir, ri, trace_events)
    assert tp.exists()
    tj = json.loads(tp.read_text(encoding="utf-8"))
    assert tj["displayTimeUnit"] == "us"
    assert tj["traceEvents"][0]["name"] == "X"


def test_export_metrics_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("NQX_PROFILE_RUN_ID", "testrun")
    import neuralqx.profile._export as exp

    exp = importlib.reload(exp)

    ri = _rankinfo(rank=0)
    outdir = str(tmp_path / "profiles")

    samples = [
        {
            "t_wall_s": 0.5,
            "process": {
                "cpu_percent": 12.5,
                "rss": 111,
                "vms": 222,
                "read_bytes": 10,
                "write_bytes": 20,
            },
            "system": {
                "cpu_percent": 33.3,
                "mem_total": 1000,
                "mem_used": 600,
                "mem_available": 400,
            },
            "gpus": [{"index": 0, "util_gpu": 90}],
        }
    ]
    p = exp.export_metrics(outdir, ri, samples)
    assert p is not None
    assert p.exists()

    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    r0 = rows[0]
    assert float(r0["t_wall_s"]) == pytest.approx(0.5)
    assert r0["proc_rss"] == "111"
    assert r0["sys_mem_total"] == "1000"
    gj = json.loads(r0["gpus_json"])
    assert gj[0]["index"] == 0


def test_export_readme_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("NQX_PROFILE_RUN_ID", "testrun")
    import neuralqx.profile._export as exp

    exp = importlib.reload(exp)

    ri = _rankinfo(rank=0)
    outdir = str(tmp_path / "profiles")
    p1 = exp.export_readme(outdir, ri)
    assert p1.exists()
    t1 = p1.read_text(encoding="utf-8")
    p2 = exp.export_readme(outdir, ri)
    assert p2 == p1
    t2 = p2.read_text(encoding="utf-8")
    assert t1 == t2
    assert "neuralqx profiling output" in t1


def test_export_all_returns_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("NQX_PROFILE_RUN_ID", "testrun")
    import neuralqx.profile._export as exp

    exp = importlib.reload(exp)

    ri = _rankinfo(rank=0)
    outdir = str(tmp_path / "profiles")

    summary = {"schema": "s", "root": {"name": "root", "stats": {"inclusive_ns": 1}}}
    trace = [{"name": "X", "ph": "X", "ts": 0, "dur": 1, "pid": 0, "tid": 0}]
    metrics = [
        {"t_wall_s": 0.0, "process": {}, "system": {}, "gpus": []},
    ]

    out = exp.export_all(outdir, ri, summary, trace, metrics)
    assert "summary" in out and Path(out["summary"]).exists()
    assert "trace" in out and Path(out["trace"]).exists()
    assert "metrics" in out and Path(out["metrics"]).exists()

    run_dir = exp.get_run_dir(outdir, ri)
    assert (run_dir / "README.txt").exists()
