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

import atexit
import os
import threading
import time
from dataclasses import dataclass, replace

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ._buffer import TraceBuffer
from ._cfg import ProfilingConfig
from ._cfg import get_output_dir
from ._cfg import profiling_enabled
from ._cfg import get_bool
from ._events import Frame
from ._events import ProfileNode
from ._export import export_all
from ._export import get_run_dir
from ._jax import trace_annotation as jax_trace_annotation
from ._jax import step_annotation as jax_step_annotation
from ._jax import start_profiler_trace as jax_start_profiler_trace
from ._jax import stop_profiler_trace as jax_stop_profiler_trace
from ._jax import block_until_ready as jax_block_until_ready
from ._metrics import start_metrics, stop_metrics
from ._nvtx import range as nvtx_range
from ._rank import get_rank_info
from ._rank import RankInfo
from .summary import render_summary_text


def _now_ns() -> int:
    return time.perf_counter_ns()


class _NullContext:
    def __enter__(self):  # noqa
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa
        return False

    def sync(self, value: Any) -> Any:
        return value


_NULL_CONTEXT = _NullContext()


class _DisabledProfiler:
    """
    Zero-overhead-ish stand-in when profiling is disabled.

    This avoids constructing a full Profiler object (threads, atexit hooks,
    output directories) while preserving the public API surface expected by
    decorators and call sites.
    """

    def __init__(self) -> None:
        self.config = ProfilingConfig(
            enabled=False,
            trace=False,
            nvtx=False,
            jax_trace_annotations=False,
            jax_profiler_trace=False,
            metrics=False,
            sync=False,
            max_events=0,
            sample_period_s=0.0,
            mpi_aggregate_on_exit=False,
        )

    def enabled(self) -> bool:
        return False

    def refresh_if_needed(self) -> None:
        return

    def maybe_sync(self, value: Any) -> Any:
        return value

    def section(
        self,
        name: str,
        cat: str = "",
        *,
        args: Optional[Dict[str, Any]] = None,
        flops: float = 0.0,
        bytes: float = 0.0,
    ):
        return _NULL_CONTEXT

    def step(
        self,
        step_num: int,
        name: str = "step",
        cat: str = "step",
        *,
        args: Optional[Dict[str, Any]] = None,
    ):
        return _NULL_CONTEXT

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "schema": "neuralqx.profiling.summary.v1",
            "config": {
                "trace": False,
                "nvtx": False,
                "jax_trace_annotations": False,
                "jax_profiler_trace": False,
                "metrics": False,
                "sync": False,
                "max_events": 0,
                "sample_period_s": 0.0,
                "mpi_aggregate_on_exit": False,
            },
            "root": {"name": "root", "cat": "", "stats": {}, "children": []},
        }

    def trace_events(self) -> List[Dict[str, Any]]:
        return []

    def flush(self) -> None:
        return

    def sync_enabled(self) -> bool:
        return False


@dataclass
class _TLSState:
    stack: List[Frame]
    step_stack: List[int]
    tid: int


class Profiler:
    """
    Process-local profiler with hierarchical section timing and optional trace/telemetry.

    Thread-safety:
      - Each thread gets an independent stack to compute exclusive time.
      - Summary aggregation merges into a shared tree. Updates are guarded by a lock.
        (Summary updates happen at section exit only; trace emission is optional.)
    """

    def __init__(self) -> None:
        self.config = ProfilingConfig.load()
        self.rank_info: RankInfo = get_rank_info()
        self.output_dir = get_output_dir()

        # create the run directory early when enabled
        # in notebooks users expect the directory to exist immediately,
        # and on HPC this is a cheap mkdir on the local/shared FS
        self._run_dir = None
        if self.config.enabled:
            try:
                self._run_dir = get_run_dir(self.output_dir, self.rank_info)
            except Exception:
                self._run_dir = None

        self._t0_ns = _now_ns()
        self._root = ProfileNode(name="root", cat="")
        self._lock = threading.Lock()

        self._trace: Optional[TraceBuffer] = (
            TraceBuffer(self.config.max_events)
            if (self.config.enabled and self.config.trace)
            else None
        )
        self._tls = threading.local()
        self._registered_threads: Dict[int, str] = {}
        self._flushed = False

        # optional telemetry sampler
        if self.config.enabled and self.config.metrics:
            start_metrics(sample_period_s=self.config.sample_period_s)

        # optional JAX profiler trace
        if self.config.enabled and self.config.jax_profiler_trace:
            try:
                run_dir = self._run_dir or get_run_dir(self.output_dir, self.rank_info)
                jdir = str(run_dir / f"jax_trace_rank{self.rank_info.rank:06d}")
                os.makedirs(jdir, exist_ok=True)
                jax_start_profiler_trace(jdir)
            except Exception:
                pass

        # ensure flush at exit
        atexit.register(self.flush)

    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def refresh_if_needed(self) -> None:
        """
        Refresh config if PROFILE toggled at runtime (common in notebooks).

        This is designed to be extremely cheap in the steady state:
        it only reads `os.environ["NQX_PROFILE"]` via profiling_enabled().
        """

        # allow toggling SYNC without recreating the profiler.
        # this is cheap (just an env read + bool parse) and avoids "stuck sync=True".
        # try:
        #     new_sync = get_bool("PROFILE_SYNC", default=False, env_key="NQX_PROFILE_SYNC")
        #     if bool(new_sync) != bool(self.config.sync):
        #         self.config.sync = bool(new_sync)
        # except Exception:
        #     pass

        # Allow toggling SYNC without recreating the profiler/config tree.
        try:
            new_sync = self.sync_enabled()
            if new_sync != bool(self.config.sync):
                self.config = replace(self.config, sync=new_sync)
        except Exception:
            pass

        try:
            en = profiling_enabled()
        except Exception:
            return
        if bool(en) == bool(self.config.enabled):
            return

        # if we are disabling, flush what we have first
        if self.config.enabled and not en:
            try:
                self.flush()
            except Exception:
                pass

        # reset core state
        self.config = ProfilingConfig.load()
        self._t0_ns = _now_ns()
        self._root = ProfileNode(name="root", cat="")
        self._flushed = False
        self._trace = (
            TraceBuffer(self.config.max_events)
            if (self.config.enabled and self.config.trace)
            else None
        )

        # (re)create run dir
        self._run_dir = None
        if self.config.enabled:
            try:
                self._run_dir = get_run_dir(self.output_dir, self.rank_info)
            except Exception:
                self._run_dir = None

        # (re)start metrics
        try:
            if self.config.enabled and self.config.metrics:
                start_metrics(sample_period_s=self.config.sample_period_s)
        except Exception:
            pass

        # (re)start JAX trace
        try:
            if self.config.enabled and self.config.jax_profiler_trace:
                run_dir = self._run_dir or get_run_dir(self.output_dir, self.rank_info)
                jdir = str(run_dir / f"jax_trace_rank{self.rank_info.rank:06d}")
                os.makedirs(jdir, exist_ok=True)
                jax_start_profiler_trace(jdir)
        except Exception:
            pass

    def _get_tls(self) -> _TLSState:
        st = getattr(self._tls, "state", None)
        if st is None:
            tid = threading.get_ident()
            st = _TLSState(stack=[], step_stack=[], tid=tid)
            self._tls.state = st
            # best-effort thread name
            try:
                self._registered_threads[tid] = threading.current_thread().name
            except Exception:
                self._registered_threads[tid] = str(tid)
        return st

    def maybe_sync(self, value: Any) -> Any:
        """
        Synchronize JAX arrays if configured. This can introduce delays.
        """
        if not (self.config.enabled and self.sync_enabled()):
            return value
        return jax_block_until_ready(value)

    def section(
        self,
        name: str,
        cat: str = "",
        *,
        args: Optional[Dict[str, Any]] = None,
        flops: float = 0.0,
        bytes: float = 0.0,
    ):
        """
        Create a profiling section context manager.
        """
        self.refresh_if_needed()
        if not self.config.enabled:
            return _NullContext()
        return _SectionCtx(
            self, name=name, cat=cat, args=args, flops=flops, bytes=bytes
        )

    def step(
        self,
        step_num: int,
        name: str = "step",
        cat: str = "step",
        *,
        args: Optional[Dict[str, Any]] = None,
    ):
        """
        Mark a repeated step. Adds metadata and (optionally) JAX step trace annotation.
        """
        self.refresh_if_needed()
        if not self.config.enabled:
            return _NullContext()
        return _StepCtx(self, step_num=step_num, name=name, cat=cat, args=args)

    def summary_dict(self) -> Dict[str, Any]:
        with self._lock:
            root = self._root.to_dict()
        # define a meaningful total for reporting: wall time since profiler init
        t_end_ns = _now_ns()
        elapsed_ns = int(t_end_ns - self._t0_ns)
        try:
            # root stats: 1 call spanning entire run, exclusive is unaccounted time
            child_sum = 0
            for ch in root.get("children", []) or []:
                st = ch.get("stats", {}) or {}
                child_sum += int(st.get("inclusive_ns", 0) or 0)
            root["stats"] = {
                "calls": 1,
                "inclusive_ns": elapsed_ns,
                "exclusive_ns": max(0, elapsed_ns - child_sum),
                "min_ns": elapsed_ns,
                "max_ns": elapsed_ns,
                "flops": float(
                    root.get("stats", {}).get("flops", 0.0)
                    if isinstance(root.get("stats", {}), dict)
                    else 0.0
                ),
                "bytes": float(
                    root.get("stats", {}).get("bytes", 0.0)
                    if isinstance(root.get("stats", {}), dict)
                    else 0.0
                ),
            }
        except Exception:
            pass
        return {
            "schema": "neuralqx.profiling.summary.v1",
            "rank": {
                "rank": self.rank_info.rank,
                "size": self.rank_info.size,
                "local_rank": self.rank_info.local_rank,
                "hostname": self.rank_info.hostname,
                "pid": self.rank_info.pid,
                "backend": self.rank_info.backend,
            },
            "config": {
                "trace": self.config.trace,
                "nvtx": self.config.nvtx,
                "jax_trace_annotations": self.config.jax_trace_annotations,
                "jax_profiler_trace": self.config.jax_profiler_trace,
                "metrics": self.config.metrics,
                "sync": self.config.sync,
                "max_events": self.config.max_events,
                "sample_period_s": self.config.sample_period_s,
                "mpi_aggregate_on_exit": self.config.mpi_aggregate_on_exit,
            },
            "timing": {
                "t0_ns": self._t0_ns,
                "t_end_ns": t_end_ns,
            },
            "root": root,
        }

    def trace_events(self) -> List[Dict[str, Any]]:
        if self._trace is None:
            return []
        # add metadata events
        events = []

        # process name
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": self.rank_info.rank,
                "tid": 0,
                "args": {
                    "name": f"rank{self.rank_info.rank}@{self.rank_info.hostname}"
                },
            }
        )
        for tid, tname in self._registered_threads.items():
            events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": self.rank_info.rank,
                    "tid": tid,
                    "args": {"name": tname},
                }
            )
        events.extend(self._trace.snapshot())
        return events

    def flush(self) -> None:
        if self._flushed:
            return
        self._flushed = True

        # stop JAX trace
        try:
            if self.config.enabled and self.config.jax_profiler_trace:
                jax_stop_profiler_trace()
        except Exception:
            pass

        if not self.config.enabled:
            return

        summary = self.summary_dict()
        trace = self.trace_events() if self.config.trace else []

        # collect metrics before export (and stop the sampler)
        try:
            metrics = (
                stop_metrics() if (self.config.enabled and self.config.metrics) else []
            )
        except Exception:
            metrics = []

        try:
            out_paths = export_all(
                self.output_dir, self.rank_info, summary, trace, metrics
            )

            # also write a human-friendly text summary
            try:
                run_dir = get_run_dir(self.output_dir, self.rank_info)
                txt = render_summary_text(summary)
                txt_path = run_dir / f"summary_rank{self.rank_info.rank:06d}.txt"
                txt_path.write_text(txt + "\n", encoding="utf-8")
            except Exception:
                pass
        except Exception:
            # never fail user job due to profiling output
            return

    def sync_enabled(self) -> bool:
        """
        Allow runtime toggling of sync via env var without recreating the Profiler.
        """
        try:
            return bool(
                get_bool(
                    "PROFILE_SYNC",
                    default=bool(self.config.sync),
                    env_key="NQX_PROFILE_SYNC",
                )
            )
        except Exception:
            return bool(self.config.sync)


class _SectionCtx:
    __slots__ = (
        "_p",
        "name",
        "cat",
        "args",
        "flops",
        "bytes",
        "_cm_stack",
        "_start_ns",
        "_sync_value",
    )

    def __init__(
        self,
        p: Profiler,
        name: str,
        cat: str,
        args: Optional[Dict[str, Any]],
        flops: float,
        bytes: float,
    ) -> None:
        self._p = p
        self.name = name
        self.cat = cat
        self.args = args or {}
        self.flops = float(flops)
        self.bytes = float(bytes)
        self._cm_stack: List[Any] = []
        self._start_ns = 0

    def __enter__(self):
        p = self._p
        tls = p._get_tls()
        parent_node = tls.stack[-1].node if tls.stack else p._root

        with p._lock:
            node = parent_node.get_child(self.name, self.cat)

        self._start_ns = _now_ns()
        frame = Frame(
            node=node, start_ns=self._start_ns, flops=self.flops, bytes=self.bytes
        )
        tls.stack.append(frame)

        # optional NVTX and JAX annotations
        if p.config.nvtx:
            try:
                cm = nvtx_range(f"{self.cat}:{self.name}" if self.cat else self.name)
                self._cm_stack.append(cm)
                cm.__enter__()
            except Exception:
                pass

        if p.config.jax_trace_annotations:
            try:
                cm = jax_trace_annotation(
                    f"{self.cat}:{self.name}" if self.cat else self.name
                )
                self._cm_stack.append(cm)
                cm.__enter__()
            except Exception:
                pass

        self._sync_value = None
        return self

    def __exit__(self, exc_type, exc, tb):
        p = self._p
        tls = p._get_tls()

        # IMPORTANT: synchronize before taking end timestamp
        # but only if configured AND user provided a dependent value
        # IMPORTANT: read env-controlled sync at exit (this is the blocking point)
        if p.sync_enabled() and self._sync_value is not None:
            try:
                jax_block_until_ready(self._sync_value)
            except Exception:
                pass

        end_ns = _now_ns()
        try:
            frame = tls.stack.pop()
        except Exception:
            # stack corruption should not crash user
            return False

        inclusive = int(end_ns - frame.start_ns)
        exclusive = int(inclusive - frame.child_inclusive_ns_accum)
        if exclusive < 0:
            exclusive = 0

        # update parent child accum
        if tls.stack:
            tls.stack[-1].child_inclusive_ns_accum += inclusive

        # summary update
        with p._lock:
            frame.node.stats.update(
                inclusive_ns=inclusive,
                exclusive_ns=exclusive,
                flops=frame.flops,
                bytes_=frame.bytes,
            )

        # trace event
        if p._trace is not None:
            try:
                ts_us = (frame.start_ns - p._t0_ns) / 1_000.0
                dur_us = inclusive / 1_000.0
                args = dict(self.args) if self.args else {}

                # attach current step if any
                if tls.step_stack:
                    args.setdefault("step", tls.step_stack[-1])

                # attach rank/device hints
                args.setdefault("rank", p.rank_info.rank)
                args.setdefault("local_rank", p.rank_info.local_rank)
                ev = {
                    "name": self.name,
                    "cat": self.cat,
                    "ph": "X",
                    "ts": ts_us,
                    "dur": dur_us,
                    "pid": p.rank_info.rank,
                    "tid": tls.tid,
                    "args": args,
                }
                p._trace.append(ev)
            except Exception:
                pass

        # exit nested contexts (LIFO)
        while self._cm_stack:
            cm = self._cm_stack.pop()
            try:
                cm.__exit__(exc_type, exc, tb)
            except Exception:
                pass

        return False

    def sync(self, value: Any) -> Any:
        # user marks which value to block on (if PROFILE_SYNC enabled)
        self._sync_value = value
        return value


class _StepCtx:
    __slots__ = ("_p", "step_num", "name", "cat", "args", "_cm_stack", "_sec")

    def __init__(
        self,
        p: Profiler,
        step_num: int,
        name: str,
        cat: str,
        args: Optional[Dict[str, Any]],
    ) -> None:
        self._p = p
        self.step_num = int(step_num)
        self.name = name
        self.cat = cat
        self.args = args or {}
        self._cm_stack: List[Any] = []
        self._sec: Optional[_SectionCtx] = None

    def __enter__(self):
        p = self._p
        tls = p._get_tls()
        tls.step_stack.append(self.step_num)

        # optional JAX step annotation
        if p.config.jax_trace_annotations:
            try:
                cm = jax_step_annotation(self.name, self.step_num)
                self._cm_stack.append(cm)
                cm.__enter__()
            except Exception:
                pass

        # also create a section around the step (aggregated in summary, step id in args)
        a = dict(self.args)
        a.setdefault("step", self.step_num)
        self._sec = _SectionCtx(
            p, name=self.name, cat=self.cat, args=a, flops=0.0, bytes=0.0
        )
        self._sec.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        p = self._p
        tls = p._get_tls()

        if self._sec is not None:
            try:
                self._sec.__exit__(exc_type, exc, tb)
            except Exception:
                pass

        while self._cm_stack:
            cm = self._cm_stack.pop()
            try:
                cm.__exit__(exc_type, exc, tb)
            except Exception:
                pass

        try:
            tls.step_stack.pop()
        except Exception:
            pass
        return False


_PROFILER: Optional[Profiler] = None
_DISABLED_PROFILER = _DisabledProfiler()
_PROFILER_LOCK = threading.Lock()


def get_profiler() -> Profiler | _DisabledProfiler:
    global _PROFILER
    enabled_now = bool(profiling_enabled())

    # profiling disabled: return a lightweight no-op profiler without creating
    # a full profiler instance.
    if not enabled_now:
        p = _PROFILER
        if p is None:
            return _DISABLED_PROFILER
        if not bool(p.config.enabled):
            _PROFILER = None
            return _DISABLED_PROFILER

        # profiling got disabled after being enabled: flush once and drop.
        with _PROFILER_LOCK:
            p = _PROFILER
            if p is not None and bool(p.config.enabled):
                try:
                    p.flush()
                except Exception:
                    pass
                _PROFILER = None
        return _DISABLED_PROFILER

    # fast path: already initialized and enabled
    p = _PROFILER
    if p is not None and bool(p.config.enabled):
        return p

    # slow path: initialize or reconfigure to enabled mode
    with _PROFILER_LOCK:
        p = _PROFILER
        if p is None:
            _PROFILER = Profiler()
            return _PROFILER

        if not bool(p.config.enabled):
            try:
                p.flush()
            except Exception:
                pass
            _PROFILER = Profiler()

        return _PROFILER
