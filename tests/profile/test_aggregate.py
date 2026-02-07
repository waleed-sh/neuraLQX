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

from neuralqx.profile.aggregate import aggregate_run


def _write_summary(path: Path, rank: int, root_inc: int, a_inc: int):
    payload = {
        "schema": "neuralqx.profiling.summary.v1",
        "rank": {
            "rank": rank,
            "size": 2,
            "local_rank": 0,
            "hostname": "h",
            "pid": 1,
            "backend": "serial",
        },
        "root": {
            "name": "root",
            "cat": "",
            "stats": {
                "inclusive_ns": root_inc,
                "exclusive_ns": 0,
                "calls": 1,
                "flops": 0.0,
                "bytes": 0.0,
            },
            "children": [
                {
                    "name": "A",
                    "cat": "cat",
                    "stats": {
                        "inclusive_ns": a_inc,
                        "exclusive_ns": a_inc,
                        "calls": 1,
                        "flops": 1.0,
                        "bytes": 2.0,
                    },
                    "children": [],
                }
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_run_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        aggregate_run(str(tmp_path))


def test_aggregate_run_merges_and_emits_min_mean_max(tmp_path):
    _write_summary(tmp_path / "summary_rank000000.json", rank=0, root_inc=100, a_inc=10)
    _write_summary(tmp_path / "summary_rank000001.json", rank=1, root_inc=200, a_inc=20)

    out = aggregate_run(str(tmp_path))
    assert out["schema"] == "neuralqx.profiling.aggregate.v1"
    assert out["n_ranks"] == 2
    assert len(out["ranks"]) == 2

    root = out["root"]

    assert root["stats"]["inclusive_ns_sum"] == 300
    assert root["stats"]["inclusive_ns_min"] == 100
    assert root["stats"]["inclusive_ns_max"] == 200
    assert root["stats"]["inclusive_ns_mean"] == 150

    a = [c for c in root["children"] if c["name"] == "A" and c["cat"] == "cat"][0]
    assert a["stats"]["inclusive_ns_sum"] == 30
    assert a["stats"]["inclusive_ns_min"] == 10
    assert a["stats"]["inclusive_ns_max"] == 20
    assert a["stats"]["inclusive_ns_mean"] == 15
    assert a["stats"]["calls_sum"] == 2
    assert a["stats"]["flops_sum"] == 2.0
    assert a["stats"]["bytes_sum"] == 4.0


def test_aggregate_run_writes_output_path(tmp_path):
    _write_summary(tmp_path / "summary_rank000000.json", rank=0, root_inc=100, a_inc=10)
    _write_summary(tmp_path / "summary_rank000001.json", rank=1, root_inc=100, a_inc=10)

    outp = tmp_path / "agg.json"
    out = aggregate_run(str(tmp_path), output_path=str(outp))
    assert outp.exists()
    j = json.loads(outp.read_text(encoding="utf-8"))
    assert j["schema"] == out["schema"]
