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

from typing import Any
from typing import Dict
from typing import List

_NS_IN_S = 1_000_000_000


def _fmt_ns(ns: int) -> str:
    if ns < 1_000:
        return f"{ns} ns"
    if ns < 1_000_000:
        return f"{ns/1_000:.3f} µs"
    if ns < 1_000_000_000:
        return f"{ns/1_000_000:.3f} ms"
    return f"{ns/1_000_000_000:.3f} s"


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _sort_children(children: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # sort by inclusive time descending
    return sorted(
        children,
        key=lambda c: (c.get("stats", {}).get("inclusive_ns", 0), c.get("name", "")),
        reverse=True,
    )


def _line(node: Dict[str, Any], total_ns: int) -> str:
    st = node.get("stats", {}) or {}
    inc = int(st.get("inclusive_ns", 0) or 0)
    calls = int(st.get("calls", 0) or 0)
    pct = 100.0 * _safe_div(inc, float(total_ns)) if total_ns else 0.0
    cat = node.get("cat") or ""
    name = node.get("name") or "?"
    extra = ""
    flops = float(st.get("flops", 0.0) or 0.0)
    bytes_ = float(st.get("bytes", 0.0) or 0.0)
    if flops or bytes_:
        ai = _safe_div(flops, bytes_)
        # bytes per second and flops per second derived from inclusive time
        sec = inc / _NS_IN_S if inc else 0.0
        gflops = _safe_div(flops, sec) / 1e9 if sec else 0.0
        gbs = _safe_div(bytes_, sec) / 1e9 if sec else 0.0
        extra = f" | AI={ai:.3g} flop/B, {gflops:.3g} GF/s, {gbs:.3g} GB/s"
    if cat:
        label = f"{cat}:{name}"
    else:
        label = name
    return f"({pct:5.1f}%) | {label} : {_fmt_ns(inc)} | calls={calls}{extra}"


def render_summary_text(summary: Dict[str, Any], max_depth: int = 64) -> str:
    """
    Render a hierarchical summary dict (as produced by Profiler.summary_dict()) into a readable tree
    """
    root = summary.get("root", summary)
    st = root.get("stats", {}) or {}
    total_ns = int(st.get("inclusive_ns", 0) or 0)
    lines: List[str] = []
    lines.append(f"Total: {_fmt_ns(total_ns)}")
    children = _sort_children(root.get("children", []) or [])

    def rec(nodes: List[Dict[str, Any]], prefix: str, depth: int) -> None:
        if depth >= max_depth:
            return
        for idx, n in enumerate(nodes):
            last = idx == len(nodes) - 1
            branch = "└── " if last else "├── "
            lines.append(prefix + branch + _line(n, total_ns))
            sub = _sort_children(n.get("children", []) or [])
            if sub:
                ext = "    " if last else "│   "
                rec(sub, prefix + ext, depth + 1)

    rec(children, "", 0)
    return "\n".join(lines)
