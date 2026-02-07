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


import sys
import types
from neuralqx.profile import _jax


def test_jax_unavailable_noops(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", None)
    _jax._JAX = None
    assert _jax.available() is False

    with _jax.trace_annotation("x"):
        pass
    with _jax.step_annotation("step", 1):
        pass

    obj = object()
    assert _jax.block_until_ready(obj) is obj


def test_jax_available_and_annotations(monkeypatch):
    class _FakeCtx:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeProfiler:
        TraceAnnotation = _FakeCtx
        StepTraceAnnotation = _FakeCtx

        def __init__(self):
            self.started = []
            self.stopped = 0

        def start_trace(self, logdir):
            self.started.append(logdir)

        def stop_trace(self):
            self.stopped += 1

    class _FakeTreeUtil:
        @staticmethod
        def tree_map(fn, x):
            if isinstance(x, dict):
                return {k: _FakeTreeUtil.tree_map(fn, v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                out = [_FakeTreeUtil.tree_map(fn, v) for v in x]
                return type(x)(out)
            return fn(x)

    fake = types.SimpleNamespace(profiler=_FakeProfiler(), tree_util=_FakeTreeUtil())
    monkeypatch.setitem(sys.modules, "jax", fake)
    _jax._JAX = None
    assert _jax.available() is True

    with _jax.trace_annotation("hello"):
        pass
    with _jax.step_annotation("iter", 3):
        pass

    class X:
        def __init__(self):
            self.ready = 0

        def block_until_ready(self):
            self.ready += 1
            return self

    x = {"a": X(), "b": [X(), X()]}
    out = _jax.block_until_ready(x)
    assert out["a"].ready == 1
    assert out["b"][0].ready == 1
    assert out["b"][1].ready == 1

    _jax.start_profiler_trace("/tmp/fake")
    _jax.stop_profiler_trace()
    assert fake.profiler.started == ["/tmp/fake"]
    assert fake.profiler.stopped == 1
