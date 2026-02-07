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


import types
import pytest


def _fake_rankinfo():
    from neuralqx.profile._rank import RankInfo

    return RankInfo(rank=0, size=1, local_rank=0, hostname="h", pid=1, backend="serial")


class _Cfg:
    def __init__(
        self,
        enabled=True,
        trace=False,
        nvtx=False,
        jax_annot=False,
        jax_trace=False,
        metrics=False,
        sync=False,
        max_events=1000,
        sample_period_s=1.0,
        mpi_agg=False,
    ):
        self.enabled = enabled
        self.trace = trace
        self.nvtx = nvtx
        self.jax_trace_annotations = jax_annot
        self.jax_profiler_trace = jax_trace
        self.metrics = metrics
        self.sync = sync
        self.max_events = max_events
        self.sample_period_s = sample_period_s
        self.mpi_aggregate_on_exit = mpi_agg


def test_profiler_section_inclusive_exclusive_accounting(monkeypatch, tmp_path):
    import neuralqx.profile.profiler as pm

    monkeypatch.setattr(pm.atexit, "register", lambda fn: None)

    monkeypatch.setattr(
        pm,
        "ProfilingConfig",
        types.SimpleNamespace(load=lambda: _Cfg(enabled=True, trace=False)),
    )

    monkeypatch.setattr(pm, "get_rank_info", lambda: _fake_rankinfo())
    monkeypatch.setattr(pm, "get_output_dir", lambda: str(tmp_path))

    monkeypatch.setattr(pm, "profiling_enabled", lambda: True, raising=True)

    times = iter([0, 100, 110, 150, 200, 250, 300, 350, 400])
    monkeypatch.setattr(pm, "_now_ns", lambda: next(times))

    p = pm.Profiler()

    with p.section("A", cat="c"):
        with p.section("B", cat="c"):
            pass

    s = p.summary_dict()
    root = s["root"]
    children = {(c["cat"], c["name"]): c for c in (root.get("children") or [])}
    a = children[("c", "A")]
    b = {(c["cat"], c["name"]): c for c in (a.get("children") or [])}[("c", "B")]

    assert a["stats"]["inclusive_ns"] == 100
    assert b["stats"]["inclusive_ns"] == 40
    assert b["stats"]["exclusive_ns"] == 40
    assert a["stats"]["exclusive_ns"] == 60


def test_profiler_trace_events_include_metadata_and_step(monkeypatch, tmp_path):
    import neuralqx.profile.profiler as pm

    monkeypatch.setattr(pm.atexit, "register", lambda fn: None)
    monkeypatch.setattr(
        pm,
        "ProfilingConfig",
        types.SimpleNamespace(
            load=lambda: _Cfg(enabled=True, trace=True, max_events=100)
        ),
    )
    monkeypatch.setattr(pm, "profiling_enabled", lambda: True, raising=True)

    monkeypatch.setattr(pm, "get_rank_info", lambda: _fake_rankinfo())
    monkeypatch.setattr(pm, "get_output_dir", lambda: str(tmp_path))

    p = pm.Profiler()

    with p.step(7, name="ITER", cat="step"):
        with p.section("X", cat="vmc", args={"k": 1}):
            pass

    evs = p.trace_events()
    assert any(e.get("name") == "process_name" and e.get("ph") == "M" for e in evs)
    assert any(e.get("name") == "thread_name" and e.get("ph") == "M" for e in evs)

    xs = [e for e in evs if e.get("ph") == "X" and e.get("name") == "X"]
    assert xs, "missing X trace event"
    args = xs[0]["args"]
    assert args["k"] == 1
    assert args["rank"] == 0
    assert args["local_rank"] == 0
    assert args.get("step") == 7


def test_profiler_maybe_sync_delegates_to_jax_block_until_ready(monkeypatch, tmp_path):
    import neuralqx.profile.profiler as pm

    monkeypatch.setattr(pm.atexit, "register", lambda fn: None)
    monkeypatch.setattr(
        pm,
        "ProfilingConfig",
        types.SimpleNamespace(load=lambda: _Cfg(enabled=True, sync=True)),
    )
    monkeypatch.setattr(pm, "profiling_enabled", lambda: True, raising=True)

    monkeypatch.setattr(pm, "get_rank_info", lambda: _fake_rankinfo())
    monkeypatch.setattr(pm, "get_output_dir", lambda: str(tmp_path))

    called = {"n": 0}
    monkeypatch.setattr(
        pm,
        "jax_block_until_ready",
        lambda x: called.__setitem__("n", called["n"] + 1) or x,
    )

    p = pm.Profiler()
    out = p.maybe_sync({"x": 1})
    assert out == {"x": 1}
    assert called["n"] == 1


def test_refresh_if_needed_flushes_when_toggled_off(monkeypatch, tmp_path):
    import neuralqx.profile.profiler as pm

    monkeypatch.setattr(pm.atexit, "register", lambda fn: None)

    cfg_on = _Cfg(enabled=True)
    cfg_off = _Cfg(enabled=False)

    loads = {"n": 0}

    def load():
        loads["n"] += 1
        return cfg_on if loads["n"] == 1 else cfg_off

    monkeypatch.setattr(pm, "ProfilingConfig", types.SimpleNamespace(load=load))
    monkeypatch.setattr(pm, "get_rank_info", lambda: _fake_rankinfo())
    monkeypatch.setattr(pm, "get_output_dir", lambda: str(tmp_path))

    monkeypatch.setattr(pm, "profiling_enabled", lambda: False, raising=True)

    p = pm.Profiler()

    flushed = {"n": 0}
    monkeypatch.setattr(p, "flush", lambda: flushed.__setitem__("n", flushed["n"] + 1))

    p.refresh_if_needed()
    assert flushed["n"] == 1
    assert p.config.enabled is False
