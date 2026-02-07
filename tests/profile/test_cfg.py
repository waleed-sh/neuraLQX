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


import os
from neuralqx.profile import _cfg


class _FakeCfgModule:
    def __init__(self, mapping=None, static=None):
        self._m = dict(mapping or {})
        self._s = dict(static or {})

    def get(self, key):
        return self._m.get(key)

    def get_static(self, key):
        return self._s.get(key)


def test_as_bool_variants():
    f = _cfg._as_bool
    assert f(True) is True
    assert f(False) is False
    assert f(1) is True
    assert f(0) is False
    assert f("1") is True
    assert f("true") is True
    assert f("YES") is True
    assert f("on") is True
    assert f("0") is False
    assert f("false") is False
    assert f("Off") is False
    assert f("") is False
    assert f("maybe", default=True) is True
    assert f("maybe", default=False) is False
    assert f(None, default=True) is True


def test_as_int_and_float():
    assert _cfg._as_int(3, default=7) == 3
    assert _cfg._as_int(3.9, default=7) == 3
    assert _cfg._as_int("12", default=7) == 12
    assert _cfg._as_int(" 12 ", default=7) == 12
    assert _cfg._as_int("x", default=7) == 7

    assert _cfg._as_float(1.5, default=2.5) == 1.5
    assert _cfg._as_float(4, default=2.5) == 4.0
    assert _cfg._as_float("3.25", default=2.5) == 3.25
    assert _cfg._as_float("x", default=2.5) == 2.5


def test_profiling_enabled_env_takes_precedence(monkeypatch):
    monkeypatch.setenv("NQX_PROFILE", "1")
    assert _cfg.profiling_enabled() is True
    monkeypatch.setenv("NQX_PROFILE", "0")
    assert _cfg.profiling_enabled() is False


def test_profiling_enabled_falls_back_to_cfg(monkeypatch):
    monkeypatch.delenv("NQX_PROFILE", raising=False)

    fake = _FakeCfgModule(mapping={"PROFILE": "1"})
    monkeypatch.setattr(_cfg, "_try_import_cfg", lambda: fake)
    assert _cfg.profiling_enabled() is True

    fake2 = _FakeCfgModule(mapping={"PROFILE": "0"})
    monkeypatch.setattr(_cfg, "_try_import_cfg", lambda: fake2)
    assert _cfg.profiling_enabled() is False


def test_get_output_dir_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("NQX_PROFILE_DIR", str(tmp_path / "envdir"))

    fake = _FakeCfgModule(static={"Profiling Directory": str(tmp_path / "cfgdir")})
    monkeypatch.setattr(_cfg, "_try_import_cfg", lambda: fake)
    assert _cfg.get_output_dir() == str(tmp_path / "cfgdir")

    fake2 = _FakeCfgModule(static={})
    monkeypatch.setattr(_cfg, "_try_import_cfg", lambda: fake2)
    assert _cfg.get_output_dir() == str(tmp_path / "envdir")

    monkeypatch.delenv("NQX_PROFILE_DIR", raising=False)
    monkeypatch.setattr(_cfg, "_try_import_cfg", lambda: None)
    out = _cfg.get_output_dir()
    assert out.endswith("nqx_profiles")
    assert os.path.isabs(out)


def test_getters_prefer_cfg_over_env(monkeypatch):
    fake = _FakeCfgModule(mapping={"PROFILE_TRACE": "1", "PROFILE_MAX_EVENTS": "123"})
    monkeypatch.setattr(_cfg, "_try_import_cfg", lambda: fake)
    monkeypatch.setenv("NQX_PROFILE_TRACE", "0")
    monkeypatch.setenv("NQX_PROFILE_MAX_EVENTS", "999")

    assert (
        _cfg.get_bool("PROFILE_TRACE", default=False, env_key="NQX_PROFILE_TRACE")
        is True
    )
    assert (
        _cfg.get_int("PROFILE_MAX_EVENTS", default=0, env_key="NQX_PROFILE_MAX_EVENTS")
        == 123
    )


def test_profiling_config_load_defaults_and_overrides(monkeypatch):
    monkeypatch.setattr(_cfg, "_try_import_cfg", lambda: None, raising=True)

    monkeypatch.setenv("NQX_PROFILE", "0")
    cfg = _cfg.ProfilingConfig.load()
    assert cfg.enabled is False
    assert cfg.trace is True
    assert cfg.nvtx is False
    assert cfg.jax_trace_annotations is True

    monkeypatch.setenv("NQX_PROFILE", "1")
    monkeypatch.setenv("NQX_PROFILE_TRACE", "1")
    monkeypatch.setenv("NQX_PROFILE_NVTX", "1")
    monkeypatch.setenv("NQX_PROFILE_JAX_ANNOTATE", "0")
    monkeypatch.setenv("NQX_PROFILE_JAX_TRACE", "1")
    monkeypatch.setenv("NQX_PROFILE_METRICS", "1")
    monkeypatch.setenv("NQX_PROFILE_SYNC", "1")
    monkeypatch.setenv("NQX_PROFILE_MAX_EVENTS", "7")
    monkeypatch.setenv("NQX_PROFILE_SAMPLE_PERIOD_S", "0.25")
    monkeypatch.setenv("NQX_PROFILE_MPI_AGG", "1")

    cfg2 = _cfg.ProfilingConfig.load()
    assert cfg2.enabled is True
    assert cfg2.trace is True
    assert cfg2.nvtx is True
    assert cfg2.jax_trace_annotations is False
    assert cfg2.jax_profiler_trace is True
    assert cfg2.metrics is True
    assert cfg2.sync is True
    assert cfg2.max_events == 7
    assert abs(cfg2.sample_period_s - 0.25) < 1e-9
    assert cfg2.mpi_aggregate_on_exit is True
