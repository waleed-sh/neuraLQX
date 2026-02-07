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
import neuralqx.profile.decorators as dec


class _FakeCM:
    def __init__(self):
        self.entered = 0
        self.exited = []

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited.append((exc_type, exc, tb))
        return False


class _FakeProfiler:
    def __init__(self, enabled=True, sync=False):
        self._enabled = enabled
        self.config = types.SimpleNamespace(sync=sync)
        self.sections = []
        self.steps = []
        self._sync = sync

    def sync_enabled(self):
        return self._sync

    def enabled(self):
        return self._enabled

    def section(self, name, cat="", args=None, flops=0.0, bytes=0.0):
        cm = _FakeCM()
        self.sections.append((name, cat, args, flops, bytes, cm))
        return cm

    def step(self, step_num, name="step", cat="step", args=None):
        cm = _FakeCM()
        self.steps.append((step_num, name, cat, args, cm))
        return cm


def test_section_step_helpers_delegate(monkeypatch):
    fp = _FakeProfiler(enabled=True)
    monkeypatch.setattr(dec, "get_profiler", lambda: fp)

    with dec.section("S", cat="c", args={"a": 1}, flops=2.0, bytes=3.0):
        pass
    assert fp.sections[0][0] == "S"
    assert fp.sections[0][1] == "c"
    assert fp.sections[0][2] == {"a": 1}

    with dec.step(5, name="ITER", cat="step", args={"k": 9}):
        pass
    assert fp.steps[0][0] == 5
    assert fp.steps[0][1] == "ITER"


def test_profile_decorator_noop_when_disabled(monkeypatch):
    fp = _FakeProfiler(enabled=False)
    monkeypatch.setattr(dec, "get_profiler", lambda: fp)

    called = {"n": 0}

    @dec.profile(cat="x")
    def f(x):
        called["n"] += 1
        return x + 1

    assert f(2) == 3
    assert called["n"] == 1
    assert fp.sections == []


def test_profile_decorator_sync_true_blocks(monkeypatch):
    fp = _FakeProfiler(enabled=True, sync=False)
    monkeypatch.setattr(dec, "get_profiler", lambda: fp)

    seen = {"sync": 0}

    def fake_block(x):
        seen["sync"] += 1
        return ("ready", x)

    monkeypatch.setattr(dec, "block_until_ready", fake_block)

    @dec.profile(sync=True)
    def g():
        return 123

    out = g()
    assert out == ("ready", 123)
    assert seen["sync"] == 1

    assert fp.sections
    assert fp.sections[0][0] == g.__qualname__


def test_profile_decorator_sync_none_uses_global_config(monkeypatch):
    fp = _FakeProfiler(enabled=True, sync=True)
    monkeypatch.setattr(dec, "get_profiler", lambda: fp)

    seen = {"sync": 0}
    monkeypatch.setattr(
        dec,
        "block_until_ready",
        lambda x: seen.__setitem__("sync", seen["sync"] + 1) or x,
    )

    @dec.profile(sync=None)
    def h():
        return "x"

    assert h() == "x"
    assert seen["sync"] == 1


def test_profile_decorator_exception_calls_exit_and_reraises(monkeypatch):
    fp = _FakeProfiler(enabled=True)
    monkeypatch.setattr(dec, "get_profiler", lambda: fp)

    @dec.profile(name="boom", cat="c")
    def bad():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        bad()

    _name, _cat, _args, _flops, _bytes, cm = fp.sections[0]
    assert cm.entered == 1
    assert cm.exited
    et, ev, tb = cm.exited[-1]
    assert et is ValueError
    assert isinstance(ev, ValueError)
    assert tb is not None
