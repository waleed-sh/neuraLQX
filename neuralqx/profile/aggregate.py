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

import json
from dataclasses import dataclass
from pathlib import Path

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional

# ((cat,name),...)
PathKey = Tuple[Tuple[str, str], ...]


@dataclass
class RankStat:
    inclusive_ns: int
    exclusive_ns: int
    calls: int
    flops: float
    bytes: float


def _iter_nodes(node: Dict[str, Any], path: List[Tuple[str, str]]):
    cat = node.get("cat") or ""
    name = node.get("name") or ""
    if name != "root":
        path2 = path + [(cat, name)]
    else:
        path2 = path
    st = node.get("stats", {}) or {}
    yield tuple(path2), RankStat(
        inclusive_ns=int(st.get("inclusive_ns", 0) or 0),
        exclusive_ns=int(st.get("exclusive_ns", 0) or 0),
        calls=int(st.get("calls", 0) or 0),
        flops=float(st.get("flops", 0.0) or 0.0),
        bytes=float(st.get("bytes", 0.0) or 0.0),
    )
    for ch in node.get("children", []) or []:
        yield from _iter_nodes(ch, path2)


def _load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_run(run_dir: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Offline aggregation of per-rank summary files.

    Produces:
      - merged tree with summed inclusive/exclusive/calls/flops/bytes
      - per-node across-rank min/mean/max for inclusive_ns and exclusive_ns

    This avoids runtime MPI barriers and is recommended for large HPC runs.
    """
    d = Path(run_dir)
    files = sorted(d.glob("summary_rank*.json"))
    if not files:
        raise FileNotFoundError(f"No summary_rank*.json found in {run_dir}")

    per_path: Dict[PathKey, List[RankStat]] = {}
    metas: List[Dict[str, Any]] = []

    for f in files:
        s = _load_summary(f)
        metas.append(s.get("rank", {}))
        root = s.get("root", s)
        for pk, rs in _iter_nodes(root, []):
            per_path.setdefault(pk, []).append(rs)

    # build aggregated stats for each path
    agg: Dict[PathKey, Dict[str, Any]] = {}
    for pk, lst in per_path.items():
        n = len(lst)
        incs = [x.inclusive_ns for x in lst]
        excs = [x.exclusive_ns for x in lst]
        calls = sum(x.calls for x in lst)
        inc_sum = sum(incs)
        exc_sum = sum(excs)
        flops = sum(x.flops for x in lst)
        bytes_ = sum(x.bytes for x in lst)
        agg[pk] = {
            "calls_sum": calls,
            "inclusive_ns_sum": inc_sum,
            "exclusive_ns_sum": exc_sum,
            "inclusive_ns_min": min(incs),
            "inclusive_ns_mean": inc_sum / n,
            "inclusive_ns_max": max(incs),
            "exclusive_ns_min": min(excs),
            "exclusive_ns_mean": exc_sum / n,
            "exclusive_ns_max": max(excs),
            "flops_sum": flops,
            "bytes_sum": bytes_,
            "ranks": n,
        }

    # build a tree structure from paths
    def node_dict(name: str, cat: str) -> Dict[str, Any]:
        return {"name": name, "cat": cat, "stats": {}, "children": []}

    root = node_dict("root", "")
    index: Dict[PathKey, Dict[str, Any]] = {tuple(): root}

    # ensure parents created before children
    for pk in sorted(agg.keys(), key=lambda x: len(x)):
        parent_pk = pk[:-1]
        if parent_pk not in index:
            # create missing parents
            cur = tuple()
            node = root
            for cat, name in pk[:-1]:
                nxt = cur + ((cat, name),)
                if nxt not in index:
                    newn = node_dict(name, cat)
                    node["children"].append(newn)
                    index[nxt] = newn
                node = index[nxt]
                cur = nxt

        cat, name = pk[-1] if pk else ("", "root")
        if pk not in index:
            parent = index[parent_pk]
            newn = node_dict(name, cat)
            parent["children"].append(newn)
            index[pk] = newn

        n = index[pk]
        n["stats"] = agg[pk]

    # also set root stats from path=()
    if tuple() in agg:
        root["stats"] = agg[tuple()]

    out = {
        "schema": "neuralqx.profiling.aggregate.v1",
        "run_dir": str(d),
        "n_ranks": len(files),
        "ranks": metas,
        "root": root,
    }

    if output_path:
        Path(output_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
