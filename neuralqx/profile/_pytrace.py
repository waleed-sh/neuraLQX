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


import sys
import threading

from contextlib import contextmanager

from types import FrameType

from typing import Any
from typing import Iterable
from typing import Optional
from typing import Tuple

from .profiler import get_profiler


def _normalize_prefixes(prefixes: Optional[Iterable[str]]) -> Optional[Tuple[str, ...]]:
    if prefixes is None:
        return None
    out = []
    for p in prefixes:
        if p is None:
            continue
        s = str(p).strip()
        if s:
            out.append(s)
    return tuple(out) if out else None


def _matches_prefix(value: str, prefixes: Optional[Tuple[str, ...]]) -> bool:
    if prefixes is None:
        return True
    return any(value.startswith(p) for p in prefixes)


def _should_skip_module(module_name: str) -> bool:
    return module_name.startswith("neuralqx.profile")


class _PythonCallTraceCtx:
    def __init__(
        self,
        *,
        cat: str,
        include_prefixes: Optional[Tuple[str, ...]],
        exclude_prefixes: Optional[Tuple[str, ...]],
        max_depth: Optional[int],
    ) -> None:
        self._cat = str(cat)
        self._include_prefixes = include_prefixes
        self._exclude_prefixes = exclude_prefixes
        self._max_depth = None if max_depth is None else int(max_depth)

        self._prev_cb = None
        self._tls = threading.local()
        self._depth_by_frame: dict[int, int] = {}
        self._cm_by_frame: dict[int, Any] = {}

    def __enter__(self):
        self._prev_cb = sys.getprofile()
        sys.setprofile(self._callback)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.setprofile(self._prev_cb)
        self._prev_cb = None

        # Best effort: close any dangling sections in reverse depth order.
        dangling = list(self._cm_by_frame.items())
        if dangling:
            dangling.sort(
                key=lambda kv: self._depth_by_frame.get(kv[0], 0), reverse=True
            )
            for _, cm in dangling:
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass

        self._depth_by_frame.clear()
        self._cm_by_frame.clear()
        return False

    def _enter_guarded(self) -> bool:
        if getattr(self._tls, "in_cb", False):
            return False
        self._tls.in_cb = True
        return True

    def _exit_guarded(self) -> None:
        self._tls.in_cb = False

    def _callback(self, frame: FrameType, event: str, arg):
        # chain previous callback first
        if self._prev_cb is not None:
            try:
                self._prev_cb(frame, event, arg)
            except Exception:
                pass

        if event not in {"call", "return", "exception"}:
            return
        if not self._enter_guarded():
            return

        try:
            fid = id(frame)

            if event == "call":
                parent = frame.f_back
                parent_depth = self._depth_by_frame.get(id(parent), 0) if parent else 0
                depth = parent_depth + 1
                self._depth_by_frame[fid] = depth

                mod = str(frame.f_globals.get("__name__", ""))
                if not mod:
                    mod = "<unknown>"
                if _should_skip_module(mod):
                    return
                if not _matches_prefix(mod, self._include_prefixes):
                    return
                if self._exclude_prefixes and _matches_prefix(
                    mod, self._exclude_prefixes
                ):
                    return
                if self._max_depth is not None and depth > self._max_depth:
                    return

                name = f"{mod}.{frame.f_code.co_name}"
                try:
                    cm = get_profiler().section(name, cat=self._cat)
                    cm.__enter__()
                    self._cm_by_frame[fid] = cm
                except Exception:
                    pass
                return

            # return / exception
            cm = self._cm_by_frame.pop(fid, None)
            self._depth_by_frame.pop(fid, None)
            if cm is None:
                return

            if event == "exception" and isinstance(arg, tuple) and len(arg) == 3:
                et, ev, tb = arg
                try:
                    cm.__exit__(et, ev, tb)
                except Exception:
                    pass
                return

            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
        finally:
            self._exit_guarded()


@contextmanager
def python_call_trace(
    *,
    cat: str = "python",
    include_prefixes: Optional[Iterable[str]] = None,
    exclude_prefixes: Optional[Iterable[str]] = None,
    max_depth: Optional[int] = None,
):
    """
    Profile nested Python function calls in the current thread.

    This is intentionally opt-in and can add noticeable overhead.
    """
    prof = get_profiler()
    if not prof.enabled():
        yield None
        return

    max_depth_i = None if max_depth is None else int(max_depth)
    if max_depth_i is not None and max_depth_i <= 0:
        max_depth_i = None

    ctx = _PythonCallTraceCtx(
        cat=cat,
        include_prefixes=_normalize_prefixes(include_prefixes),
        exclude_prefixes=_normalize_prefixes(exclude_prefixes),
        max_depth=max_depth_i,
    )
    with ctx:
        yield ctx
