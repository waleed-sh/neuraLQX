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


"""User spans, decorators, events, and counters with an inactive fast path."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from collections.abc import Mapping
from contextlib import AbstractContextManager
from functools import wraps
from typing import Any
from typing import ParamSpec
from typing import TypeVar
from typing import overload

from neuralqx.utils.errors import NeuralqxTypeError

from . import _state
from ._metadata import Scalar
from ._metadata import label
from ._metadata import metadata

P = ParamSpec("P")
R = TypeVar("R")


class _NullSpan:
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


_NULL = _NullSpan()


class _Span:
    __slots__ = (
        "attributes",
        "buffer",
        "category",
        "lane",
        "name",
        "nvtx",
        "session",
        "start_ns",
        "window",
    )

    def __init__(self, session, name, category, attributes):
        self.session = session
        self.name = name
        self.category = category
        self.attributes = attributes
        self.window = None
        self.nvtx = None

    def __enter__(self):
        session = self.session
        window = session._recording_window
        if window is None:
            return None
        name = label(self.name)
        category = label(self.category, "category")
        attributes = metadata(self.attributes)
        lane, _ = session._recorder.lane()
        if lane is None:
            session._recorder.drop_lane()
            return None
        self.name, self.category, self.attributes = name, category, attributes
        self.lane = lane
        self.buffer = session._recorder.buffer()
        if session._nvtx is not None:
            self.nvtx = session._nvtx.begin(name, category)
        self.window = window
        self.buffer.open_spans += 1
        self.start_ns = time.perf_counter_ns()
        return None

    def __exit__(self, exc_type, exc, traceback):
        if self.window is None:
            return False
        end_ns = time.perf_counter_ns()
        self.buffer.open_spans -= 1
        attributes = self.attributes
        if exc_type is not None:
            attributes["exception_type"] = exc_type.__name__
        if self.window.end_ns is not None and end_ns > self.window.end_ns:
            end_ns = self.window.end_ns
            attributes["clipped_at_boundary"] = True
        try:
            if end_ns >= self.start_ns:
                self.session._recorder.append(
                    {
                        "name": self.name,
                        "cat": self.category,
                        "ph": "X",
                        "pid": 1,
                        "tid": self.lane,
                        "ts": (self.start_ns - self.session._origin_ns) / 1000,
                        "dur": (end_ns - self.start_ns) / 1000,
                        "args": attributes,
                    },
                    self.buffer,
                )
        finally:
            try:
                if self.nvtx is not None:
                    self.session._nvtx.end(self.nvtx)
            except Exception as error:
                if len(self.session._annotation_failures) < 16:
                    self.session._annotation_failures.append(
                        f"NVTX range cleanup failed: {error}"
                    )
                if exc is None:
                    raise
                exc.add_note(f"NVTX range cleanup also failed: {error}")
            finally:
                self.window = None
        return False


def span(
    name: str,
    *,
    category: str = "user",
    attributes: Mapping[str, Scalar] | None = None,
) -> AbstractContextManager[None]:
    """Measure host elapsed time for a block in the active recording window.

    This context never waits for a device. A span around asynchronous dispatch
    measures dispatch and Python work, while native device lanes show execution.
    Each thread and asyncio task has a separate lane. A span that crosses the
    end of a recording window is clipped and marked in its metadata.

    Args:
        name: Stable event label of at most 256 characters.
        category: Category used for trace filtering.
        attributes: Up to 32 Python scalar fields, copied on entry. Arrays and
            arbitrary objects are rejected without conversion. Metadata is
            limited to 4096 characters to bound retained memory.

    Returns:
        A context manager. Outside an active window it is a reusable no-op.

    Raises:
        NeuralqxTypeError: Active-span metadata contains a nonscalar object.
        NeuralqxValueError: An active label or metadata value exceeds its limits.

    Examples:
        Add an application region to a capture::

            with span("sample", attributes={"chains": 64}):
                samples = sampler.sample(parameters)
    """
    session = _state.active
    if session is None or session._recording_window is None:
        return _NULL
    return _Span(session, name, category, attributes)


@overload
def annotate(
    function: Callable[P, R],
    *,
    name: str | None = None,
    category: str = "user",
    attributes: Mapping[str, Scalar] | None = None,
    sync: bool = False,
) -> Callable[P, R]:
    """Wrap a supplied callable while preserving its argument and return types."""
    ...


@overload
def annotate(
    function: None = None,
    *,
    name: str | None = None,
    category: str = "user",
    attributes: Mapping[str, Scalar] | None = None,
    sync: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Configure an annotation decorator before supplying its callable."""
    ...


def annotate(
    function: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    category: str = "user",
    attributes: Mapping[str, Scalar] | None = None,
    sync: bool = False,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function or coroutine with an application span.

    The inactive wrapper reads the active-session reference and calls the
    function directly. It does not read configuration, inspect arguments,
    traverse pytrees, acquire a recording lock, or query a clock.

    Args:
        function: Function to decorate. Omit it to supply keyword options.
        name: Event label. Defaults to the function's qualified name.
        category: Trace category.
        attributes: Small Python scalar metadata copied when decorating.
        sync: When recording, explicitly wait for all returned JAX array leaves
            before ending the span. This measures completion latency and can
            include previously queued work. It serializes dispatch and is
            intended for deliberate latency experiments.

    Returns:
        A wrapped callable, or a decorator when function is omitted.

    Raises:
        NeuralqxTypeError: The target is a generator or async generator. Put spans
            inside such generators to measure execution rather than creation.

    Examples:
        Keep the annotation outside jit to measure every dispatch::

            @annotate(name="local_energy")
            @jax.jit
            def local_energy(parameters, samples):
                return evaluate(parameters, samples)

        An annotation inside jit runs during tracing, not each compiled call.
        Use compiled_scope for stable names inside compiled computations.
    """
    label(category, "category")
    saved_attributes = metadata(attributes)
    if type(sync) is not bool:
        raise NeuralqxTypeError("sync must be a boolean.")

    def decorate(fn):
        if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
            raise NeuralqxTypeError(
                "Use profiling spans inside generators rather than decorating their creation."
            )
        event_name = label(name if name is not None else fn.__qualname__)

        def wait(value):
            import jax

            return jax.block_until_ready(value)

        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def asynchronous(*args, **kwargs):
                session = _state.active
                if session is None or session._recording_window is None:
                    return await fn(*args, **kwargs)
                with _Span(session, event_name, category, saved_attributes):
                    result = await fn(*args, **kwargs)
                    return wait(result) if sync else result

            return asynchronous

        @wraps(fn)
        def synchronous(*args, **kwargs):
            session = _state.active
            if session is None or session._recording_window is None:
                return fn(*args, **kwargs)
            with _Span(session, event_name, category, saved_attributes):
                result = fn(*args, **kwargs)
                return wait(result) if sync else result

        return synchronous

    return decorate if function is None else decorate(function)


def event(
    name: str, *, category: str = "user", attributes: Mapping[str, Scalar] | None = None
) -> None:
    """Record an instantaneous host event on the current thread or task lane.

    Args:
        name: Stable event label.
        category: Trace category.
        attributes: Python scalar metadata with the same limits as span.

    Raises:
        NeuralqxTypeError: Active event metadata contains a nonscalar object.
        NeuralqxValueError: Active labels or metadata violate the span limits.
    """
    session = _state.active
    if session is None or session._recording_window is None:
        return
    session._instant(label(name), label(category, "category"), metadata(attributes))


def counter(name: str, values: Mapping[str, int | float]) -> None:
    """Record finite host numeric values as a time-series counter.

    Args:
        name: Counter track name.
        values: Mapping of series names to Python integers or floats. Convert
            device scalars deliberately outside this call if desired.

    Raises:
        NeuralqxTypeError: A value is not a Python integer or float.
        NeuralqxValueError: Metadata is too large or contains nonfinite values.
    """
    session = _state.active
    if session is None or session._recording_window is None:
        return
    attributes = metadata(values)
    if not attributes or any(
        type(value) not in (int, float) for value in attributes.values()
    ):
        raise NeuralqxTypeError("Counters require at least one Python numeric value.")
    session._instant(label(name), "counter", attributes, phase="C")


def compiled_scope(name: str) -> AbstractContextManager[Any]:
    """Name operations in compiled JAX programs without host callbacks or fences.

    This scope always contributes compiler metadata, even when capture is off.
    Its names survive cached executions and appear in native operation views.
    It does not emit a separate timed application span.

    Args:
        name: Stable compiler scope name.

    Returns:
        JAX's named_scope context manager.
    """
    import jax

    return jax.named_scope(label(name))
