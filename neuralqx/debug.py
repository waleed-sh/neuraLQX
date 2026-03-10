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

"""
neuralQX debugging module.

Usage (decorate)
    from neuralqx.debugging import trace, timeit, errors_only, io_trace

    @trace(tag="SAMPLER")
    def sample(...): ...

    @timeit(tag="LANCZOS", warn_ms=250)
    def lanczos_step(...): ...

    @errors_only(tag="IO")
    def load_checkpoint(...): ...

    @io_trace(tag="KERNEL")
    def _get_conn_flattened_kernel(...): ...
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import dataclasses
import datetime as _dt
import faulthandler
import functools
import logging
import os
import platform
import random
import re
import signal
import sys
import tempfile
import threading
import time
import traceback

from collections import Counter
from collections import defaultdict
from collections import deque

from pathlib import Path

from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import Iterable
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

_T = TypeVar("_T")

# Note: we're about to redefine some functionalities available somewhere else in neuraLQX to intentionally
#       keep this module self-contained

#
#
#   Context and state

_CALL_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "nqx_call_depth", default=0
)
_CALL_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "nqx_call_id", default=""
)
_CALL_FUNC: contextvars.ContextVar[str] = contextvars.ContextVar(
    "nqx_call_func", default=""
)
_CALL_TAG: contextvars.ContextVar[str] = contextvars.ContextVar(
    "nqx_call_tag", default=""
)

_ONCE_KEYS_LOCK = threading.Lock()
_ONCE_KEYS: set[str] = set()

_EVENT_BUFFER_MAX = 5000
_EVENT_BUFFER: Deque[str] = deque(maxlen=_EVENT_BUFFER_MAX)

_STATS_LOCK = threading.Lock()
_STATS_CALLS: Counter[str] = Counter()
_STATS_ERRORS: Counter[str] = Counter()
_STATS_TOTAL_NS: Dict[str, int] = defaultdict(int)
_STATS_MAX_NS: Dict[str, int] = defaultdict(int)

_STATE_LOCK = threading.RLock()


@dataclasses.dataclass(frozen=True)
class DebugSettings:
    enabled: bool
    verbosity: int  # 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL
    log_dir: Path
    session_id: str
    rank: int
    world: int
    pid: int
    logfile: Optional[Path]


@dataclasses.dataclass
class _RuntimeState:
    initialised: bool = False
    logger: Optional[logging.Logger] = None
    settings: Optional[DebugSettings] = None
    file_handler: Optional[logging.Handler] = None
    buffer_handler: Optional[logging.Handler] = None
    hooks_installed: bool = False
    warnings_installed: bool = False
    faulthandler_enabled: bool = False


_RT = _RuntimeState()

#
#
#   Helpers: cfg, parsing, MPI


def _get_cfg():
    # lazy import to avoid circular import at package init
    from neuralqx import cfg  # type: ignore

    return cfg


_TRUE_RE = re.compile(r"^(1|true|t|yes|y|on)$", re.IGNORECASE)
_FALSE_RE = re.compile(r"^(0|false|f|no|n|off)$", re.IGNORECASE)


def _parse_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip()
    if _TRUE_RE.match(s):
        return True
    if _FALSE_RE.match(s):
        return False
    return default


_LEVELS: Dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.CRITICAL,
}


def _parse_level(v: Any, default: int = logging.INFO) -> int:
    if isinstance(v, int):
        return int(v)
    if v is None:
        return default
    s = str(v).strip().upper()
    return _LEVELS.get(s, default)


def _mpi_rank_world() -> Tuple[int, int]:
    # Legacy name kept, but rank/size now come from the distributed runtime.
    try:
        from neuralqx.utils import distributed as _dist

        return int(_dist.process_index()), int(_dist.process_count())
    except Exception:
        return 0, 1


def _safe_dir(pathlike: Any) -> Path:
    try:
        p = Path(str(pathlike)).expanduser()
        return p
    except Exception:
        return Path(tempfile.gettempdir()) / "neuralqx-logs"


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def _short_session_id() -> str:
    return "".join(random.choice("0123456789abcdef") for _ in range(10))


#
#
#   Logging formatter, buffer handler, context injection


class _IsoFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = _dt.datetime.fromtimestamp(record.created).astimezone()
        return dt.isoformat(timespec="milliseconds")


class _ContextFilter(logging.Filter):
    def __init__(self, rank: int, world: int, session_id: str):
        super().__init__()
        self.rank = rank
        self.world = world
        self.session_id = session_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.nqx_rank = self.rank
        record.nqx_world = self.world
        record.nqx_session = self.session_id
        record.nqx_thread = threading.current_thread().name

        # attach call context (if any)
        record.nqx_depth = _CALL_DEPTH.get()
        record.nqx_call_id = _CALL_ID.get()
        record.nqx_call_func = _CALL_FUNC.get()
        record.nqx_call_tag = _CALL_TAG.get()
        return True


class _RecentBufferHandler(logging.Handler):
    """Keeps formatted recent log lines in memory for crash dumps."""

    def __init__(self, formatter: logging.Formatter):
        super().__init__(level=logging.DEBUG)
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            _EVENT_BUFFER.append(line)
        except Exception:
            # never break the program because of debugging
            pass


def _build_logger(settings: DebugSettings) -> logging.Logger:
    logger = logging.getLogger("neuralqx.debug")

    # we gate emission ourselves, always accept records
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # remove old handlers if re-initialising
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = (
        "%(asctime)s | %(levelname)-8s | r%(nqx_rank)d/%(nqx_world)d | "
        "pid=%(process)d | thr=%(nqx_thread)s | sess=%(nqx_session)s | "
        "depth=%(nqx_depth)d | cid=%(nqx_call_id)s | "
        "%(message)s"
    )
    formatter = _IsoFormatter(fmt=fmt)

    # buffer handler (memory)
    bufh = _RecentBufferHandler(formatter)
    bufh.addFilter(_ContextFilter(settings.rank, settings.world, settings.session_id))
    logger.addHandler(bufh)
    _RT.buffer_handler = bufh

    if settings.logfile is not None:
        fh = logging.FileHandler(
            settings.logfile, mode="a", encoding="utf-8", delay=True
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        fh.addFilter(_ContextFilter(settings.rank, settings.world, settings.session_id))
        logger.addHandler(fh)
        _RT.file_handler = fh
    else:
        # no file possible: still keep buffer handler for diagnostics
        _RT.file_handler = None

    return logger


def _emit(level: int, msg: str, *, tag: Optional[str] = None) -> None:
    # fast path if not initialised or disabled
    with _STATE_LOCK:
        st = _RT.settings
        lg = _RT.logger
    if st is None or lg is None or not st.enabled:
        return

    # ensure tag prefix and indentation (call-tree)
    depth = _CALL_DEPTH.get()
    indent = ("│  " * depth) + ("├─ " if depth > 0 else "")
    prefix = f"[{tag}] :: " if tag else ""
    lg.log(level, f"{indent}{prefix}{msg}")


#
#
#   Settings and init


def read_settings() -> DebugSettings:
    cfg = _get_cfg()

    enabled = _parse_bool(cfg.get("DEBUG"), default=False)
    level = _parse_level(cfg.get("LOG_LEVEL"), default=logging.INFO)

    # if debug is off, we still compute directory for consistency, may not write file
    try:
        log_dir_raw = cfg.get_static("Leach Directory")
    except Exception:
        log_dir_raw = None
    log_dir = (
        _safe_dir(log_dir_raw)
        if log_dir_raw
        else (Path(tempfile.gettempdir()) / "neuralqx-logs")
    )

    rank, world = _mpi_rank_world()
    pid = os.getpid()
    session_id = _short_session_id()

    logfile: Optional[Path] = None
    if enabled:
        # ensure directory exists
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        logfile = log_dir / f"neuralqx_{ts}_pid{pid}_rank{rank}_{session_id}.log"

    return DebugSettings(
        enabled=enabled,
        verbosity=level,
        log_dir=log_dir,
        session_id=session_id,
        rank=rank,
        world=world,
        pid=pid,
        logfile=logfile,
    )


def initialise(*, force: bool = False) -> Optional[Path]:
    """
    Initialise neuralQX debugging for this process.

    - Creates per-session log file under cfg.get_static("Leach Directory") when enabled.
    - Installs exception / warning / faulthandler hooks (best-effort).
    - Safe to call multiple times.

    Returns the logfile path if created, else None.
    """
    with _STATE_LOCK:
        if _RT.initialised and not force:
            return _RT.settings.logfile if _RT.settings else None

        settings = read_settings()
        _RT.settings = settings
        _RT.logger = _build_logger(settings)
        _RT.initialised = True

    if settings.enabled:
        _write_session_header()
        _install_hooks_best_effort()
        _install_warning_hook_best_effort()
        _enable_faulthandler_best_effort()
        atexit.register(_atexit_summary)

    return settings.logfile


def refresh_settings(*, reinit: bool = False) -> None:
    """
    Refresh settings from cfg. If reinit=True, rebuilds the log file/handlers.
    """
    if reinit:
        initialise(force=True)
        return
    with _STATE_LOCK:
        if not _RT.initialised:
            return
        old = _RT.settings
        new = (
            dataclasses.replace(
                old,
                enabled=read_settings().enabled,
                verbosity=read_settings().verbosity,
            )
            if old
            else read_settings()
        )
        _RT.settings = new


def is_enabled() -> bool:
    with _STATE_LOCK:
        st = _RT.settings
    return bool(st and st.enabled)


def get_logfile() -> Optional[Path]:
    with _STATE_LOCK:
        st = _RT.settings
    return st.logfile if st else None


def _write_session_header() -> None:
    with _STATE_LOCK:
        st = _RT.settings
    if not st or not st.enabled:
        return

    lines = [
        "=== NeuralQX Debug Session ===",
        f"started: {_now_iso()}",
        f"logfile: {st.logfile}",
        f"rank/world: {st.rank}/{st.world}",
        f"pid: {st.pid}",
        f"python: {sys.version.replace(os.linesep, ' ')}",
        f"platform: {platform.platform()}",
        f"cwd: {os.getcwd()}",
        "==============================",
    ]
    for ln in lines:
        _emit(logging.INFO, ln, tag="SESSION")


#
#
#   Warning and exception hooks

_ORIG_SHOWWARNING = None
_ORIG_EXCEPTHOOK = None


def _install_warning_hook_best_effort() -> None:
    global _ORIG_SHOWWARNING
    with _STATE_LOCK:
        if _RT.warnings_installed:
            return
        st = _RT.settings
    if not st or not st.enabled:
        return

    try:
        import warnings

        _ORIG_SHOWWARNING = warnings.showwarning

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            # attach current call context in the warning message
            ctx = ""
            cid = _CALL_ID.get()
            func = _CALL_FUNC.get()
            if cid or func:
                ctx = f" (in {func}, cid={cid})"
            _emit(
                logging.WARNING,
                f"WARNING{ctx}: {category.__name__}: {message} @ {filename}:{lineno}",
                tag="WARN",
            )
            # still display via original behaviour if a file is provided
            try:
                return _ORIG_SHOWWARNING(
                    message, category, filename, lineno, file=file, line=line
                )
            except Exception:
                return None

        warnings.showwarning = _showwarning

        with _STATE_LOCK:
            _RT.warnings_installed = True
        _emit(logging.INFO, "Installed warnings hook", tag="SESSION")
    except Exception as e:
        _emit(logging.WARNING, f"Failed to install warnings hook: {e}", tag="SESSION")


def _install_hooks_best_effort() -> None:
    global _ORIG_EXCEPTHOOK
    with _STATE_LOCK:
        if _RT.hooks_installed:
            return
        st = _RT.settings
    if not st or not st.enabled:
        return

    try:
        _ORIG_EXCEPTHOOK = sys.excepthook

        def _hook(exc_type, exc, tb):
            _emit(logging.CRITICAL, "UNHANDLED EXCEPTION:", tag="CRASH")
            # log traceback (bounded)
            tb_text = "".join(traceback.format_exception(exc_type, exc, tb))
            for chunk in _split_lines(tb_text, 2000):
                _emit(logging.CRITICAL, chunk, tag="CRASH")
            _dump_recent_events_best_effort(tag="CRASH")
            try:
                return _ORIG_EXCEPTHOOK(exc_type, exc, tb)
            except Exception:
                return None

        sys.excepthook = _hook

        # Python 3.8+: thread excepthook, we dont support this anyway but still
        if hasattr(threading, "excepthook"):

            orig_thr_hook = threading.excepthook

            def _thr_hook(args):
                _emit(
                    logging.CRITICAL,
                    f"UNHANDLED THREAD EXCEPTION in {args.thread.name}:",
                    tag="CRASH",
                )
                tb_text = "".join(
                    traceback.format_exception(
                        args.exc_type, args.exc_value, args.exc_traceback
                    )
                )
                for chunk in _split_lines(tb_text, 2000):
                    _emit(logging.CRITICAL, chunk, tag="CRASH")
                _dump_recent_events_best_effort(tag="CRASH")
                try:
                    return orig_thr_hook(args)
                except Exception:
                    return None

            threading.excepthook = _thr_hook

        # optional: SIGUSR1 stack dump (helpful on clusters)
        try:
            signal.signal(signal.SIGUSR1, lambda *_: dump_stacks(tag="SIGUSR1"))
        except Exception:
            pass

        with _STATE_LOCK:
            _RT.hooks_installed = True

        _emit(logging.INFO, "Installed exception hooks", tag="SESSION")
    except Exception as e:
        _emit(logging.WARNING, f"Failed to install exception hooks: {e}", tag="SESSION")


def _enable_faulthandler_best_effort() -> None:
    with _STATE_LOCK:
        if _RT.faulthandler_enabled:
            return
        st = _RT.settings
    if not st or not st.enabled:
        return

    try:
        # prefer dumping to the log file if available
        if st.logfile is not None:
            f = open(st.logfile, "a", encoding="utf-8")
            faulthandler.enable(file=f, all_threads=True)
            faulthandler.register(signal.SIGUSR2, file=f, all_threads=True)
            _emit(
                logging.INFO,
                "Enabled faulthandler (SIGUSR2 triggers stack dump)",
                tag="SESSION",
            )
        else:
            faulthandler.enable(all_threads=True)
            _emit(logging.INFO, "Enabled faulthandler (stderr)", tag="SESSION")

        with _STATE_LOCK:
            _RT.faulthandler_enabled = True
    except Exception as e:
        _emit(logging.WARNING, f"Failed to enable faulthandler: {e}", tag="SESSION")


def _split_lines(text: str, max_chars: int) -> Iterable[str]:
    # split long blocks into log-friendly pieces
    chunk = []
    n = 0
    for line in text.splitlines():
        if n + len(line) + 1 > max_chars and chunk:
            yield "\n".join(chunk)
            chunk = []
            n = 0
        chunk.append(line)
        n += len(line) + 1
    if chunk:
        yield "\n".join(chunk)


#
#
#   Diagnostics utilities


def dump_recent_events(*, n: int = 200, tag: str = "DUMP") -> None:
    """Log the last N buffered events into the log (useful at high severity)."""
    with _STATE_LOCK:
        st = _RT.settings
    if not st or not st.enabled:
        return

    _emit(
        logging.CRITICAL,
        f"--- Recent events (last {min(n, len(_EVENT_BUFFER))}) ---",
        tag=tag,
    )
    for line in list(_EVENT_BUFFER)[-n:]:
        _emit(logging.CRITICAL, line, tag=tag)
    _emit(logging.CRITICAL, "--- End recent events ---", tag=tag)


def _dump_recent_events_best_effort(*, tag: str) -> None:
    try:
        dump_recent_events(n=200, tag=tag)
    except Exception:
        pass


def dump_stacks(*, tag: str = "STACKS") -> None:
    """Dump all thread stacks to the log."""
    with _STATE_LOCK:
        st = _RT.settings
    if not st or not st.enabled:
        return

    _emit(logging.CRITICAL, "=== Thread stack dump ===", tag=tag)
    frames = sys._current_frames()
    for th in threading.enumerate():
        f = frames.get(th.ident)
        if f is None:
            continue
        _emit(logging.CRITICAL, f"--- Thread: {th.name} (id={th.ident}) ---", tag=tag)
        stack = "".join(traceback.format_stack(f))
        for chunk in _split_lines(stack, 2000):
            _emit(logging.CRITICAL, chunk, tag=tag)
    _emit(logging.CRITICAL, "=== End thread stack dump ===", tag=tag)


def log_once(level: int, key: str, msg: str, *, tag: Optional[str] = None) -> None:
    """Log a message only once per process (per key)."""
    with _ONCE_KEYS_LOCK:
        if key in _ONCE_KEYS:
            return
        _ONCE_KEYS.add(key)
    _emit(level, msg, tag=tag)


def event(
    msg: str, *, tag: Optional[str] = None, level: int = logging.INFO, **fields: Any
) -> None:
    """Structured-ish event log with key=value fields."""
    if fields:
        kv = " ".join(
            f"{k}={_summarize_value(v, hard=True)}" for k, v in fields.items()
        )
        _emit(level, f"{msg} | {kv}", tag=tag)
    else:
        _emit(level, msg, tag=tag)


#
#
#   Value summarisation (important: avoid logging huge arrays/objects)


def _is_array_like(x: Any) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _summarize_array(x: Any, *, hard: bool) -> str:
    # avoid triggering device->host transfers, don't materialise
    try:
        shape = getattr(x, "shape", None)
        dtype = getattr(x, "dtype", None)
        cls = type(x).__name__
        s = f"{cls}(shape={tuple(shape) if shape is not None else None}, dtype={dtype})"
        if hard:
            # add a very small extra hint if present and cheap
            dev = getattr(x, "device", None)
            if dev is not None:
                s += f", device={dev}"
        return s
    except Exception:
        return f"{type(x).__name__}(array-like)"


def _summarize_mapping(m: Mapping[Any, Any], *, hard: bool, max_items: int = 8) -> str:
    try:
        items = list(m.items())[:max_items]
        inner = ", ".join(
            f"{_summarize_value(k, hard=hard)}: {_summarize_value(v, hard=hard)}"
            for k, v in items
        )
        more = "" if len(m) <= max_items else f", …(+{len(m)-max_items})"
        return f"{type(m).__name__}(len={len(m)}, {{{inner}{more}}})"
    except Exception:
        return f"{type(m).__name__}(mapping)"


def _summarize_iterable(it: Any, *, hard: bool, max_items: int = 8) -> str:
    try:
        # may be expensive, only used on small iterables in practice
        lst = list(it)
        items = lst[:max_items]
        inner = ", ".join(_summarize_value(v, hard=hard) for v in items)
        more = "" if len(lst) <= max_items else f", …(+{len(lst)-max_items})"
        return f"{type(it).__name__}(len={len(lst)}, [{inner}{more}])"
    except Exception:
        return f"{type(it).__name__}(iterable)"


def _summarize_value(v: Any, *, hard: bool, max_str: int = 240) -> str:
    if v is None:
        return "None"
    if isinstance(v, (int, float, bool)):
        return repr(v)
    if isinstance(v, str):
        s = v.replace("\n", "\\n")
        return repr(s[:max_str] + ("…" if len(s) > max_str else ""))
    if _is_array_like(v):
        return _summarize_array(v, hard=hard)
    if isinstance(v, Mapping):
        return _summarize_mapping(v, hard=hard)
    if isinstance(v, (list, tuple, set, frozenset)):
        # avoid printing huge lists
        try:
            items = list(v)[:8]
            inner = ", ".join(_summarize_value(x, hard=hard) for x in items)
            more = "" if len(v) <= 8 else f", …(+{len(v)-8})"
            return f"{type(v).__name__}(len={len(v)}, [{inner}{more}])"
        except Exception:
            return f"{type(v).__name__}(sequence)"
    # fallback repr
    try:
        r = repr(v)
        r = r.replace("\n", "\\n")
        return r[:max_str] + ("…" if len(r) > max_str else "")
    except Exception:
        return f"<{type(v).__name__}>"


def _summarize_args(
    args: Tuple[Any, ...], kwargs: Dict[str, Any], *, hard: bool
) -> str:
    parts = []
    if args:
        parts.append("args=" + _summarize_value(args, hard=hard))
    if kwargs:
        parts.append("kwargs=" + _summarize_mapping(kwargs, hard=hard))
    return " ".join(parts) if parts else ""


#
#
#   Caller and call ids


def _find_external_caller() -> str:
    """
    Best-effort caller site, skipping frames inside this module.
    Returns: "file.py:line in func"
    """
    try:
        this_file = __file__
        f = sys._getframe(2)  # wrapper -> user call site (usually)
        while f:
            fn = f.f_code.co_filename
            if fn != this_file and "functools" not in fn:
                return f"{os.path.basename(fn)}:{f.f_lineno} in {f.f_code.co_name}"
            f = f.f_back
    except Exception:
        pass
    return "unknown"


_CALL_SEQ = 0
_CALL_SEQ_LOCK = threading.Lock()


def _next_call_id() -> str:
    global _CALL_SEQ
    with _CALL_SEQ_LOCK:
        _CALL_SEQ += 1
        seq = _CALL_SEQ
    # compact, readable id: rank-seq
    with _STATE_LOCK:
        st = _RT.settings
    r = st.rank if st else 0
    return f"{r}-{seq}"


#
#
#   Decorators


def _should_emit(level: int) -> bool:
    with _STATE_LOCK:
        st = _RT.settings

    # standard semantics
    return bool(st and st.enabled and st.verbosity <= level)


def _ensure_init_if_needed() -> None:
    with _STATE_LOCK:
        ok = _RT.initialised
    if not ok:
        initialise(force=False)


def trace(
    _fn: Optional[Callable[..., _T]] = None,
    *,
    tag: Optional[str] = None,
    log_args_at: int = logging.CRITICAL,
    log_return_at: int = logging.DEBUG,
    warn_slow_ms: Optional[float] = 500.0,
    reraise: bool = True,
) -> Union[Callable[..., _T], Callable[[Callable[..., _T]], Callable[..., _T]]]:
    """
    Full-spectrum instrumentation.

    Emission policy (when NQX_DEBUG=True):
    - INFO: entry log (function called, caller site, call id)
    - DEBUG: exit log (ok, duration, optional return summary)
    - WARNING: if warn_slow_ms is set, logs WARNING when duration exceeds threshold
    - ERROR: logs exception with traceback (always, regardless of configured verbosity filter)
    - CRITICAL: includes args/kwargs summary (bounded) and richer diagnostics

    The "tag" prefix is rendered as:
        [TAG] :: message
    """

    def deco(fn: Callable[..., _T]) -> Callable[..., _T]:
        qual = f"{fn.__module__}.{getattr(fn, '__qualname__', fn.__name__)}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            _ensure_init_if_needed()
            with _STATE_LOCK:
                st = _RT.settings
            if not st or not st.enabled:
                return fn(*args, **kwargs)

            call_id = _next_call_id()
            caller = _find_external_caller()

            depth = _CALL_DEPTH.get()
            t_depth = _CALL_DEPTH.set(depth + 1)
            t_id = _CALL_ID.set(call_id)
            t_func = _CALL_FUNC.set(qual)
            t_tag = _CALL_TAG.set(tag or "")

            start_ns = time.perf_counter_ns()
            ok = False
            try:
                # ENTRY (INFO)
                if _should_emit(logging.INFO):
                    extra = ""
                    if _should_emit(log_args_at):
                        a = _summarize_args(
                            args, kwargs, hard=(_should_emit(logging.CRITICAL))
                        )
                        extra = f" | {a}" if a else ""
                    _emit(
                        logging.INFO,
                        f"CALL {qual} caller={caller} cid={call_id}{extra}",
                        tag=tag,
                    )

                out = fn(*args, **kwargs)
                ok = True
                return out
            except Exception as e:
                # always log exceptions when debug is enabled (even if verbosity would filter ERROR)
                dur_ns = time.perf_counter_ns() - start_ns
                msg = f"EXC  {qual} cid={call_id} after={dur_ns/1e6:.3f}ms caller={caller} exc={type(e).__name__}: {e}"
                _emit(logging.ERROR, msg, tag=tag)

                if _should_emit(logging.ERROR) or _should_emit(logging.CRITICAL):
                    tb = traceback.format_exc()
                    for chunk in _split_lines(tb, 2000):
                        _emit(logging.ERROR, chunk, tag=tag)

                if _should_emit(logging.CRITICAL):
                    # add richer context at CRITICAL
                    a = _summarize_args(args, kwargs, hard=True)
                    if a:
                        _emit(logging.CRITICAL, f"ARGS {a}", tag=tag)
                    dump_recent_events(n=120, tag=tag or "CRITICAL")

                with _STATS_LOCK:
                    _STATS_ERRORS[qual] += 1

                if reraise:
                    raise
                # type ignore: if not reraise, return None (caller opts into that behaviour)
                return None
            finally:
                dur_ns = time.perf_counter_ns() - start_ns
                with _STATS_LOCK:
                    _STATS_CALLS[qual] += 1
                    _STATS_TOTAL_NS[qual] += dur_ns
                    if dur_ns > _STATS_MAX_NS[qual]:
                        _STATS_MAX_NS[qual] = dur_ns

                # EXIT (DEBUG)
                if ok and _should_emit(logging.DEBUG):
                    msg = f"RET  {qual} cid={call_id} ok after={dur_ns/1e6:.3f}ms"
                    if _should_emit(log_return_at):
                        # beware huge returns, summarise
                        try:
                            # out only defined if ok
                            rv = locals().get("out", None)
                            msg += f" | return={_summarize_value(rv, hard=_should_emit(logging.CRITICAL))}"
                        except Exception:
                            pass
                    _emit(logging.DEBUG, msg, tag=tag)

                # SLOW (WARNING)
                if (
                    ok
                    and warn_slow_ms is not None
                    and dur_ns / 1e6 >= warn_slow_ms
                    and _should_emit(logging.WARNING)
                ):
                    _emit(
                        logging.WARNING,
                        f"SLOW {qual} cid={call_id} took={dur_ns/1e6:.3f}ms threshold={warn_slow_ms:.1f}ms caller={caller}",
                        tag=tag,
                    )

                # restore context vars
                _CALL_TAG.reset(t_tag)
                _CALL_FUNC.reset(t_func)
                _CALL_ID.reset(t_id)
                _CALL_DEPTH.reset(t_depth)

        return wrapper

    return deco if _fn is None else deco(_fn)


def timeit(
    _fn: Optional[Callable[..., _T]] = None,
    *,
    tag: Optional[str] = None,
    warn_ms: Optional[float] = 250.0,
    level: int = logging.DEBUG,
) -> Union[Callable[..., _T], Callable[[Callable[..., _T]], Callable[..., _T]]]:
    """
    Lightweight timing decorator.
    - logs one line with duration at `level` (default DEBUG)
    - logs WARNING if exceeds warn_ms (when warn_ms is not None)
    """

    def deco(fn: Callable[..., _T]) -> Callable[..., _T]:
        qual = f"{fn.__module__}.{getattr(fn, '__qualname__', fn.__name__)}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            _ensure_init_if_needed()
            with _STATE_LOCK:
                st = _RT.settings
            if not st or not st.enabled:
                return fn(*args, **kwargs)

            start_ns = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                dur_ms = (time.perf_counter_ns() - start_ns) / 1e6
                if _should_emit(level):
                    _emit(level, f"TIME {qual} took={dur_ms:.3f}ms", tag=tag)
                if (
                    warn_ms is not None
                    and dur_ms >= warn_ms
                    and _should_emit(logging.WARNING)
                ):
                    _emit(
                        logging.WARNING,
                        f"SLOW {qual} took={dur_ms:.3f}ms threshold={warn_ms:.1f}ms",
                        tag=tag,
                    )

        return wrapper

    return deco if _fn is None else deco(_fn)


def errors_only(
    _fn: Optional[Callable[..., _T]] = None,
    *,
    tag: Optional[str] = None,
    reraise: bool = True,
) -> Union[Callable[..., _T], Callable[[Callable[..., _T]], Callable[..., _T]]]:
    """
    Ultra-light decorator that only logs exceptions (plus traceback when verbosity allows).
    """

    def deco(fn: Callable[..., _T]) -> Callable[..., _T]:
        qual = f"{fn.__module__}.{getattr(fn, '__qualname__', fn.__name__)}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            _ensure_init_if_needed()
            with _STATE_LOCK:
                st = _RT.settings
            if not st or not st.enabled:
                return fn(*args, **kwargs)

            try:
                return fn(*args, **kwargs)
            except Exception as e:
                _emit(logging.ERROR, f"EXC {qual}: {type(e).__name__}: {e}", tag=tag)
                if _should_emit(logging.ERROR):
                    tb = traceback.format_exc()
                    for chunk in _split_lines(tb, 2000):
                        _emit(logging.ERROR, chunk, tag=tag)
                with _STATS_LOCK:
                    _STATS_ERRORS[qual] += 1
                if reraise:
                    raise
                return None

        return wrapper

    return deco if _fn is None else deco(_fn)


def io_trace(
    _fn: Optional[Callable[..., _T]] = None,
    *,
    tag: Optional[str] = None,
    at_level: int = logging.DEBUG,
) -> Union[Callable[..., _T], Callable[[Callable[..., _T]], Callable[..., _T]]]:
    """
    IO-focused decorator: logs summarised input/output (shapes/dtypes for arrays) at `at_level`.

    Typical use is for kernels / NetKet operator connections where we need shapes but not full tensors.
    """

    def deco(fn: Callable[..., _T]) -> Callable[..., _T]:
        qual = f"{fn.__module__}.{getattr(fn, '__qualname__', fn.__name__)}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            _ensure_init_if_needed()
            with _STATE_LOCK:
                st = _RT.settings
            if not st or not st.enabled:
                return fn(*args, **kwargs)

            hard = _should_emit(logging.CRITICAL)
            if _should_emit(at_level):
                _emit(
                    at_level,
                    f"IN  {qual} {_summarize_args(args, kwargs, hard=hard)}",
                    tag=tag,
                )

            out = fn(*args, **kwargs)

            if _should_emit(at_level):
                _emit(
                    at_level,
                    f"OUT {qual} return={_summarize_value(out, hard=hard)}",
                    tag=tag,
                )

            return out

        return wrapper

    return deco if _fn is None else deco(_fn)


#
#
#   Summary on exit


def _atexit_summary() -> None:
    # Best-effort never fail at exit
    try:
        with _STATE_LOCK:
            st = _RT.settings
        if not st or not st.enabled:
            return

        # only print summary if at least INFO verbosity (otherwise likely intentional silence)
        if st.verbosity > logging.INFO:
            return

        with _STATS_LOCK:
            calls = dict(_STATS_CALLS)
            errs = dict(_STATS_ERRORS)
            total_ns = dict(_STATS_TOTAL_NS)
            max_ns = dict(_STATS_MAX_NS)

        if not calls:
            return

        _emit(logging.INFO, "=== Debug summary ===", tag="SUMMARY")

        # top by total time
        top_total = sorted(
            calls.keys(), key=lambda k: total_ns.get(k, 0), reverse=True
        )[:10]
        _emit(logging.INFO, "Top total time:", tag="SUMMARY")
        for k in top_total:
            _emit(
                logging.INFO,
                f"{k} calls={calls[k]} total_ms={total_ns[k]/1e6:.1f} max_ms={max_ns[k]/1e6:.1f} errors={errs.get(k,0)}",
                tag="SUMMARY",
            )

        # top by errors
        if errs:
            top_err = sorted(errs.items(), key=lambda kv: kv[1], reverse=True)[:10]
            _emit(logging.INFO, "Top errors:", tag="SUMMARY")
            for k, n in top_err:
                _emit(logging.INFO, f"{k} errors={n}", tag="SUMMARY")

        _emit(logging.INFO, "=== End debug summary ===", tag="SUMMARY")
    except Exception:
        pass


#
#
#   Optional small context manager for adding a tag in a block


@contextlib.contextmanager
def tagged(tag: str):
    """
    Temporarily set a default tag for warnings/context (does not override decorator tag).
    """
    t = _CALL_TAG.set(tag)
    try:
        yield
    finally:
        _CALL_TAG.reset(t)


#
#
#   Public exports

__all__ = [
    "initialise",
    "refresh_settings",
    "is_enabled",
    "get_logfile",
    "trace",
    "timeit",
    "errors_only",
    "io_trace",
    "event",
    "log_once",
    "dump_recent_events",
    "dump_stacks",
    "tagged",
]
