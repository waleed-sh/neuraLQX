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


import json
from pathlib import Path
import pytest


@pytest.mark.integration
def test_end_to_end_profile_flush_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("NQX_PROFILE", "1")
    monkeypatch.setenv("NQX_PROFILE_TRACE", "1")
    monkeypatch.setenv("NQX_PROFILE_NVTX", "0")
    monkeypatch.setenv("NQX_PROFILE_METRICS", "0")
    monkeypatch.setenv("NQX_PROFILE_JAX_TRACE", "0")
    monkeypatch.setenv("NQX_PROFILE_JAX_ANNOTATE", "0")
    monkeypatch.setenv("NQX_PROFILE_SYNC", "0")
    monkeypatch.setenv("NQX_PROFILE_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("NQX_PROFILE_RUN_ID", "integration_run")

    from neuralqx.profile import _cfg

    monkeypatch.setattr(_cfg, "_try_import_cfg", lambda: None, raising=True)

    from neuralqx.profile import section, step, flush

    with step(0, name="ITER"):
        with section("outer", cat="vmc", args={"alpha": 1}):
            with section("inner", cat="vmc"):
                pass

    flush()

    out_dir = Path(_cfg.get_output_dir())
    assert out_dir.exists()

    runs = [p for p in out_dir.glob("run_*") if p.is_dir()]
    assert runs, f"no run_* dir created under {out_dir}"
    run_dir = runs[0]

    assert (run_dir / "README.txt").exists()

    sj = next(run_dir.glob("summary_rank*.json"), None)
    tj = next(run_dir.glob("trace_rank*.json"), None)
    assert sj is not None and sj.exists()
    assert tj is not None and tj.exists()

    s = json.loads(sj.read_text(encoding="utf-8"))
    assert s.get("schema") == "neuralqx.profiling.summary.v1"
    assert "root" in s and "config" in s and "rank" in s

    t = json.loads(tj.read_text(encoding="utf-8"))
    assert t.get("displayTimeUnit") == "us"
    assert isinstance(t.get("traceEvents"), list)

    txts = list(run_dir.glob("summary_rank*.txt"))
    if txts:
        assert txts[0].read_text(encoding="utf-8").strip()
