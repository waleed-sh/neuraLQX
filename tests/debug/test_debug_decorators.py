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


import logging
import time
import jax.numpy as jnp


def test_trace_emits_call_and_ret_and_indentation(debug_enabled):
    dbg = debug_enabled

    @dbg.trace(tag="TR", warn_slow_ms=None)
    def inner(x):
        return x + 1

    @dbg.trace(tag="TR", warn_slow_ms=None)
    def outer(x):
        return inner(x) + 1

    out = outer(1)
    assert out == 3

    tail = "\n".join(list(dbg._EVENT_BUFFER)[-50:])
    # must contain both CALL and RET for both functions
    assert "CALL" in tail and "RET" in tail
    assert "inner" in tail and "outer" in tail

    # indentation should show a deeper call for inner than outer
    # (the message portion includes drawing characters inserted by _emit)
    lines = [
        ln for ln in list(dbg._EVENT_BUFFER)[-50:] if "CALL" in ln and "[TR]" in ln
    ]
    assert len(lines) >= 2
    # heuristic: inner call line should contain more box-drawing separators than outer
    inner_line = [ln for ln in lines if "inner" in ln][0]
    outer_line = [ln for ln in lines if "outer" in ln][0]
    assert inner_line.count("│") >= outer_line.count("│")


def test_trace_warns_when_threshold_exceeded(debug_enabled):
    dbg = debug_enabled

    @dbg.trace(tag="SLOW", warn_slow_ms=0.0)
    def f():
        return 1

    f()
    tail = "\n".join(list(dbg._EVENT_BUFFER)[-20:])
    assert "SLOW" in tail  # warning line includes SLOW prefix


def test_errors_only_logs_exception_and_optionally_swallows(debug_enabled):
    dbg = debug_enabled

    @dbg.errors_only(tag="ERR", reraise=False)
    def boom():
        raise ValueError("nope")

    res = boom()
    assert res is None

    tail = "\n".join(list(dbg._EVENT_BUFFER)[-50:])
    assert "[ERR]" in tail
    assert "EXC" in tail
    assert "ValueError" in tail

    # stats should be recorded
    with dbg._STATS_LOCK:
        assert any(v >= 1 for v in dbg._STATS_ERRORS.values())


def test_io_trace_summarizes_arrays(debug_enabled):
    dbg = debug_enabled

    @dbg.io_trace(tag="IO", at_level=logging.DEBUG)
    def f(x):
        return x * 2

    x = jnp.ones((2, 3), dtype=jnp.float32)
    y = f(x)
    assert y.shape == (2, 3)

    tail = "\n".join(list(dbg._EVENT_BUFFER)[-30:])
    assert "[IO]" in tail
    assert "IN" in tail and "OUT" in tail
    assert "shape=(2, 3)" in tail
    assert "dtype=" in tail


def test_timeit_emits_time_and_optional_slow_warning(debug_enabled):
    dbg = debug_enabled

    @dbg.timeit(tag="TM", warn_ms=0.0, level=logging.DEBUG)
    def f():
        # sleep a tiny amount to avoid 0.000ms on some platforms
        time.sleep(0.001)
        return 5

    assert f() == 5

    tail = "\n".join(list(dbg._EVENT_BUFFER)[-20:])
    assert "[TM]" in tail
    assert "TIME" in tail
    assert "SLOW" in tail
