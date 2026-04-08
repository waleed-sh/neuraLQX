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


def test_public_api_symbols_exist():
    import neuralqx.profile as p

    assert hasattr(p, "Profiler")
    assert hasattr(p, "get_profiler")
    assert hasattr(p, "profile")
    assert hasattr(p, "section")
    assert hasattr(p, "step")
    assert hasattr(p, "wrap_callable")
    assert hasattr(p, "profile_call")
    assert hasattr(p, "patch_method")
    assert hasattr(p, "patch_attr")
    assert hasattr(p, "python_call_trace")
    assert hasattr(p, "enabled")
    assert hasattr(p, "flush")


def test_public_flush_safe_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("NQX_PROFILE", "0")
    monkeypatch.setenv("NQX_PROFILE_DIR", str(tmp_path / "profiles"))

    import neuralqx.profile as p

    p.flush()


def test_get_profiler_stays_uninitialised_when_profile_disabled(monkeypatch):
    monkeypatch.setenv("NQX_PROFILE", "0")

    import neuralqx.profile.profiler as pm

    pm._PROFILER = None
    prof = pm.get_profiler()

    assert prof.enabled() is False
    assert pm._PROFILER is None


def test_get_profiler_flushes_enabled_instance_on_disable(monkeypatch):
    monkeypatch.setenv("NQX_PROFILE", "0")

    import types
    import neuralqx.profile.profiler as pm

    calls = {"flush": 0}
    fake_prof = types.SimpleNamespace(
        config=types.SimpleNamespace(enabled=True),
        flush=lambda: calls.__setitem__("flush", calls["flush"] + 1),
    )
    pm._PROFILER = fake_prof

    prof = pm.get_profiler()

    assert prof.enabled() is False
    assert calls["flush"] == 1
    assert pm._PROFILER is None
