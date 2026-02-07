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


import re

from neuralqx.profile.summary import _fmt_ns, render_summary_text


def test_fmt_ns_boundaries():
    assert _fmt_ns(10).endswith("ns")
    assert "µs" in _fmt_ns(1_234)
    assert "ms" in _fmt_ns(1_234_567)
    assert "s" in _fmt_ns(1_234_567_890)


def test_render_summary_text_sorting_and_tree():
    summary = {
        "root": {
            "name": "root",
            "cat": "",
            "stats": {"inclusive_ns": 1000, "calls": 1},
            "children": [
                {
                    "name": "fast",
                    "cat": "cat",
                    "stats": {"inclusive_ns": 100, "calls": 2},
                    "children": [],
                },
                {
                    "name": "slow",
                    "cat": "cat",
                    "stats": {"inclusive_ns": 700, "calls": 1},
                    "children": [
                        {
                            "name": "inner",
                            "cat": "",
                            "stats": {"inclusive_ns": 200, "calls": 3},
                            "children": [],
                        }
                    ],
                },
            ],
        }
    }

    txt = render_summary_text(summary)
    lines = txt.splitlines()
    assert lines[0].startswith("Total:")
    slow_idx = [i for i, l in enumerate(lines) if "cat:slow" in l][0]
    fast_idx = [i for i, l in enumerate(lines) if "cat:fast" in l][0]
    assert slow_idx < fast_idx
    assert any(re.search(r"slow.*\n.*inner", txt, re.S) for _ in [0])


def test_render_summary_text_ai_metrics_present_when_flops_or_bytes():
    summary = {
        "root": {
            "name": "root",
            "cat": "",
            "stats": {"inclusive_ns": 1_000_000_000, "calls": 1},
            "children": [
                {
                    "name": "kern",
                    "cat": "jax",
                    "stats": {
                        "inclusive_ns": 1_000_000_000,
                        "calls": 1,
                        "flops": 10e9,
                        "bytes": 5e9,
                    },
                    "children": [],
                }
            ],
        }
    }
    txt = render_summary_text(summary)
    assert "AI=" in txt
    assert "GF/s" in txt
    assert "GB/s" in txt


def test_render_summary_text_max_depth_limits():
    node = {
        "name": "leaf",
        "cat": "",
        "stats": {"inclusive_ns": 1, "calls": 1},
        "children": [],
    }
    for i in range(20):
        node = {
            "name": f"n{i}",
            "cat": "",
            "stats": {"inclusive_ns": 1, "calls": 1},
            "children": [node],
        }
    summary = {
        "root": {
            "name": "root",
            "cat": "",
            "stats": {"inclusive_ns": 100, "calls": 1},
            "children": [node],
        }
    }

    txt = render_summary_text(summary, max_depth=3)
    assert txt.count("└──") + txt.count("├──") <= 4
