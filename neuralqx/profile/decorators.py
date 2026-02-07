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

from functools import wraps

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from ._jax import block_until_ready
from .profiler import get_profiler


def section(
    name: str,
    cat: str = "",
    *,
    args: Optional[Dict[str, Any]] = None,
    flops: float = 0.0,
    bytes: float = 0.0,
):
    """
    Context manager to profile a section of code.

    :param name: Section name
    :param cat: Category (e.g. "vmc", "sampling", "jax", ...)
    :param args: Additional metadata to attach to trace events
    :param flops: Optional counters for roofline-style derived metrics
    :param bytes: Optional counters for roofline-style derived metrics
    """
    return get_profiler().section(name, cat=cat, args=args, flops=flops, bytes=bytes)


def step(
    step_num: int,
    name: str = "step",
    cat: str = "step",
    *,
    args: Optional[Dict[str, Any]] = None,
):
    """
    Context manager to mark a repeated step (e.g. VMC iteration number).
    """
    return get_profiler().step(step_num=step_num, name=name, cat=cat, args=args)


def profile(
    name: Optional[str] = None,
    cat: str = "",
    *,
    sync: Optional[bool] = None,
    args: Optional[Dict[str, Any]] = None,
    flops: float = 0.0,
    bytes: float = 0.0,
):
    """
    Decorator to profile a function.

    sync:
      If True, block_until_ready() the returned value inside the timed region.
      This provides device-time-ish measurements for JAX, but can introduce delays.
      If None, uses global config (default False).
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        disp = name or getattr(fn, "__qualname__", getattr(fn, "__name__", "fn"))

        @wraps(fn)
        def wrapped(*a, **k):
            prof = get_profiler()
            if not prof.enabled():
                return fn(*a, **k)

            do_sync = bool(sync) if sync is not None else bool(prof.sync_enabled())
            cm = prof.section(disp, cat=cat, args=args, flops=flops, bytes=bytes)
            cm.__enter__()
            try:
                out = fn(*a, **k)
                if do_sync:
                    out = block_until_ready(out)
                cm.__exit__(None, None, None)
                return out
            except BaseException as e:
                cm.__exit__(type(e), e, e.__traceback__)
                raise

        return wrapped

    return deco
