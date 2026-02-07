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

from contextlib import contextmanager
from typing import Any

_JAX = None


def _load_jax():
    global _JAX
    if _JAX is not None:
        return
    try:
        import jax  # type: ignore

        _JAX = jax
    except Exception:
        _JAX = False


def available() -> bool:
    _load_jax()
    return bool(_JAX)


@contextmanager
def trace_annotation(name: str):
    """
    Adds a JAX trace annotation (shows up in JAX traces / perfetto, depending on tooling).
    No-op if JAX unavailable.
    """
    _load_jax()
    if not _JAX:
        yield
        return
    try:
        with _JAX.profiler.TraceAnnotation(name):
            yield
    except Exception:
        yield


@contextmanager
def step_annotation(name: str, step_num: int):
    """
    StepTraceAnnotation helps group repeated steps (e.g. VMC iterations) in JAX profilers.
    No-op if JAX unavailable.
    """
    _load_jax()
    if not _JAX:
        yield
        return
    try:
        with _JAX.profiler.StepTraceAnnotation(name, step_num=step_num):
            yield
    except Exception:
        yield


def block_until_ready(x: Any) -> Any:
    """
    Best-effort synchronization for accurate device timing. This CAN introduce delays.
    Only used when config sync=True.
    """
    _load_jax()
    if not _JAX:
        return x
    try:
        from jax import tree_util  # type: ignore

        def _bur(v):
            try:
                return v.block_until_ready()
            except Exception:
                return v

        return tree_util.tree_map(_bur, x)
    except Exception:
        try:
            return x.block_until_ready()  # type: ignore
        except Exception:
            return x


def start_profiler_trace(logdir: str) -> None:
    _load_jax()
    if not _JAX:
        return
    try:
        _JAX.profiler.start_trace(logdir)
    except Exception:
        return


def stop_profiler_trace() -> None:
    _load_jax()
    if not _JAX:
        return
    try:
        _JAX.profiler.stop_trace()
    except Exception:
        return
