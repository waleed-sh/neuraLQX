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


from neuralqx.profile import _nvtx


def test_nvtx_unavailable_noop(monkeypatch):
    monkeypatch.setattr(_nvtx, "_load_nvtx", lambda: None, raising=True)
    _nvtx._NVTX = None
    _nvtx._NVTX_KIND = None

    assert _nvtx.available() is False
    with _nvtx.range("x"):
        pass


def test_nvtx_python_package_path(monkeypatch):
    events = []

    class FakeNvtx:
        def annotate(self, name, payload=None):
            class Ctx:
                def __enter__(self_):
                    events.append(("enter", name, payload))

                def __exit__(self_, exc_type, exc, tb):
                    events.append(("exit", name, payload))
                    return False

            return Ctx()

    def fake_load():
        _nvtx._NVTX = FakeNvtx()
        _nvtx._NVTX_KIND = "nvtx"

    monkeypatch.setattr(_nvtx, "_load_nvtx", fake_load, raising=True)
    _nvtx._NVTX = None
    _nvtx._NVTX_KIND = None

    assert _nvtx.available() is True
    with _nvtx.range("hello", payload=123):
        events.append(("body",))

    assert events[0] == ("enter", "hello", 123)
    assert ("body",) in events
    assert events[-1] == ("exit", "hello", 123)


def test_nvtx_cupy_path(monkeypatch):
    stack = []

    class FakeCuNvtx:
        def RangePush(self, name):
            stack.append(("push", name))

        def RangePop(self):
            stack.append(("pop",))

    def fake_load():
        _nvtx._NVTX = FakeCuNvtx()
        _nvtx._NVTX_KIND = "cupy"

    monkeypatch.setattr(_nvtx, "_load_nvtx", fake_load, raising=True)
    _nvtx._NVTX = None
    _nvtx._NVTX_KIND = None

    assert _nvtx.available() is True
    with _nvtx.range("r1"):
        pass
    with _nvtx.range("r2", payload=7):
        pass

    assert stack[0] == ("push", "r1")
    assert stack[1] == ("pop",)
    assert stack[2] == ("push", "r2")
    assert stack[3] == ("pop",)
