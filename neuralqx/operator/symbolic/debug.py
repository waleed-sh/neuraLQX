# Copyright (c) 2026 The neuraLQX and nkDSL Authors - All rights reserved.
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

"""Debug and instrumentation utilities for neuralqx.operator.symbolic.

The debug runtime is controlled by :mod:`neuralqx.operator.symbolic.configs` options:

- ``DEBUG``: master on/off switch.
- ``DEBUG_VERBOSITY``: log level (``debug``, ``info``, ``warning``, ``error``, ``critical``).
- ``DEBUG_SCOPES``: comma-separated scopes to emit (``all``, ``dsl``, ``compile``, ``ir``,
  ``cache``, ``passes``, ``lowering``, ``runtime``).
- ``DEBUG_PASSES``: optional pass-name filter when ``passes`` scope is enabled.
- ``DEBUG_LOG_TO_FILE`` / ``DEBUG_LOG_DIR``: file logging destination.

Example::

    from neuralqx.operator.symbolic import cfg
    from neuralqx.operator.symbolic.debug import event

    with cfg.patch(DEBUG=True, DEBUG_SCOPES="compile,passes", DEBUG_PASSES="symbolic_validation"):
        event("compile started", scope="compile")
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import dataclasses
import datetime as _dt
import functools
import inspect
import logging
import os
import platform
import random
import re
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from collections import defaultdict
from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import TypeVar

_T = TypeVar("_T")

_CALL_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "neuralqx.operator.symbolic_call_depth", default=0
)
_CALL_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "neuralqx.operator.symbolic_call_id", default=""
)
_CALL_FUNC: contextvars.ContextVar[str] = contextvars.ContextVar(
    "neuralqx.operator.symbolic_call_func", default=""
)
_CALL_TAG: contextvars.ContextVar[str] = contextvars.ContextVar(
    "neuralqx.operator.symbolic_call_tag", default=""
)

_ONCE_KEYS_LOCK = threading.Lock()
_ONCE_KEYS: set[str] = set()

_EVENT_BUFFER_MAX = 5000
_EVENT_BUFFER: deque[str] = deque(maxlen=_EVENT_BUFFER_MAX)

_STATS_LOCK = threading.Lock()
_STATS_CALLS: Counter[str] = Counter()
_STATS_ERRORS: Counter[str] = Counter()
_STATS_TOTAL_NS: dict[str, int] = defaultdict(int)
_STATS_MAX_NS: dict[str, int] = defaultdict(int)

_STATE_LOCK = threading.RLock()


@dataclasses.dataclass(frozen=True)
class DebugSettings:
    enabled: bool
    verbosity: int
    scopes: tuple[str, ...]
    pass_filter: tuple[str, ...]
    log_to_file: bool
    log_dir: Path
    session_id: str
    rank: int
    world: int
    pid: int
    logfile: Path | None


@dataclasses.dataclass
class _RuntimeState:
    initialised: bool = False
    logger: logging.Logger | None = None
    settings: DebugSettings | None = None
    file_handler: logging.Handler | None = None
    buffer_handler: logging.Handler | None = None
    atexit_registered: bool = False


_RT = _RuntimeState()


def _get_cfg():
    # Lazy import to avoid hard dependency cycles at import time.
    from neuralqx.operator.symbolic.configs import cfg

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


_LEVELS: dict[str, int] = {
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
    return _LEVELS.get(str(v).strip().upper(), default)


def _normalize_name_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        pieces = value.split(",")
    elif isinstance(value, Iterable):
        pieces = [str(v) for v in value]
    else:
        pieces = [str(value)]
    normalized = [p.strip().lower() for p in pieces if str(p).strip()]
    return tuple(normalized)


_SCOPE_ALIASES: dict[str, str] = {
    "compiler": "compile",
    "pipeline": "passes",
    "pass": "passes",
    "lower": "lowering",
}


def _normalize_scope(scope: str) -> str:
    normalized = scope.strip().lower()
    return _SCOPE_ALIASES.get(normalized, normalized)


def _normalized_scopes(raw_scopes: Any) -> tuple[str, ...]:
    scopes = tuple(_normalize_scope(s) for s in _normalize_name_tuple(raw_scopes))
    return scopes if scopes else ("all",)


def _extract_scoped_passes(scopes: tuple[str, ...]) -> tuple[str, ...]:
    explicit: list[str] = []
    for scope in scopes:
        if scope.startswith("pass:"):
            name = scope[5:].strip().lower()
            if name:
                explicit.append(name)
    return tuple(explicit)


def _mpi_rank_world() -> tuple[int, int]:
    """Best-effort rank/world detection without external runtime dependencies."""
    rank_candidates = (
        os.environ.get("RANK"),
        os.environ.get("SLURM_PROCID"),
        os.environ.get("OMPI_COMM_WORLD_RANK"),
        os.environ.get("PMI_RANK"),
    )
    world_candidates = (
        os.environ.get("WORLD_SIZE"),
        os.environ.get("SLURM_NTASKS"),
        os.environ.get("OMPI_COMM_WORLD_SIZE"),
        os.environ.get("PMI_SIZE"),
    )

    rank = 0
    world = 1
    for raw in rank_candidates:
        if raw is None:
            continue
        try:
            rank = int(raw)
            break
        except ValueError:
            continue
    for raw in world_candidates:
        if raw is None:
            continue
        try:
            world = max(1, int(raw))
            break
        except ValueError:
            continue
    return rank, world


def _safe_dir(pathlike: Any) -> Path:
    try:
        return Path(str(pathlike)).expanduser()
    except Exception:
        return Path(tempfile.gettempdir()) / "neuralqx.operator.symbolic-logs"


def _short_session_id() -> str:
    return "".join(random.choice("0123456789abcdef") for _ in range(10))


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
        record.nk_rank = self.rank
        record.nk_world = self.world
        record.nk_session = self.session_id
        record.nk_thread = threading.current_thread().name
        record.nk_depth = _CALL_DEPTH.get()
        record.nk_call_id = _CALL_ID.get()
        record.nk_call_func = _CALL_FUNC.get()
        record.nk_call_tag = _CALL_TAG.get()
        return True


class _RecentBufferHandler(logging.Handler):
    """Keeps formatted recent log lines in-memory for diagnostics."""

    def __init__(self, formatter: logging.Formatter):
        super().__init__(level=logging.DEBUG)
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _EVENT_BUFFER.append(self.format(record))
        except Exception:
            pass


def _build_logger(settings: DebugSettings) -> logging.Logger:
    logger = logging.getLogger("neuralqx.operator.symbolic.debug")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    fmt = (
        "%(asctime)s | %(levelname)-8s | r%(nk_rank)d/%(nk_world)d | "
        "pid=%(process)d | thr=%(nk_thread)s | sess=%(nk_session)s | "
        "depth=%(nk_depth)d | cid=%(nk_call_id)s | %(message)s"
    )
    formatter = _IsoFormatter(fmt=fmt)

    buffer_handler = _RecentBufferHandler(formatter)
    buffer_handler.addFilter(
        _ContextFilter(settings.rank, settings.world, settings.session_id)
    )
    logger.addHandler(buffer_handler)
    _RT.buffer_handler = buffer_handler

    if settings.logfile is not None:
        file_handler = logging.FileHandler(settings.logfile, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(
            _ContextFilter(settings.rank, settings.world, settings.session_id)
        )
        logger.addHandler(file_handler)
        _RT.file_handler = file_handler
    else:
        _RT.file_handler = None

    return logger


def _should_emit(level: int) -> bool:
    with _STATE_LOCK:
        st = _RT.settings
    return bool(st and st.enabled and st.verbosity <= level)


def _emit(level: int, msg: str, *, tag: str | None = None) -> None:
    with _STATE_LOCK:
        st = _RT.settings
        logger = _RT.logger
    if st is None or logger is None or not st.enabled:
        return
    if st.verbosity > level:
        return

    depth = _CALL_DEPTH.get()
    indent = ("│  " * depth) + ("├─ " if depth > 0 else "")
    prefix = f"[{tag}] :: " if tag else ""
    logger.log(level, f"{indent}{prefix}{msg}")


def read_settings() -> DebugSettings:
    cfg = _get_cfg()

    enabled = _parse_bool(cfg.get("DEBUG"), default=False)
    verbosity = _parse_level(cfg.get("DEBUG_VERBOSITY"), default=logging.INFO)
    scopes = _normalized_scopes(cfg.get("DEBUG_SCOPES"))

    explicit_scoped_passes = _extract_scoped_passes(scopes)
    configured_passes = _normalize_name_tuple(cfg.get("DEBUG_PASSES"))
    pass_filter = tuple(dict.fromkeys((*configured_passes, *explicit_scoped_passes)))

    log_to_file = _parse_bool(cfg.get("DEBUG_LOG_TO_FILE"), default=True)
    configured_log_dir = cfg.get("DEBUG_LOG_DIR")
    if configured_log_dir:
        log_dir = _safe_dir(configured_log_dir)
    else:
        try:
            log_dir = _safe_dir(cfg.get_static("Cache Directory")) / "logs"
        except Exception:
            log_dir = Path(tempfile.gettempdir()) / "neuralqx.operator.symbolic-logs"

    rank, world = _mpi_rank_world()
    pid = os.getpid()
    session_id = _short_session_id()

    logfile: Path | None = None
    if enabled and log_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        logfile = (
            log_dir
            / f"neuralqx.operator.symbolic_{ts}_pid{pid}_rank{rank}_{session_id}.log"
        )

    return DebugSettings(
        enabled=enabled,
        verbosity=verbosity,
        scopes=scopes,
        pass_filter=pass_filter,
        log_to_file=log_to_file,
        log_dir=log_dir,
        session_id=session_id,
        rank=rank,
        world=world,
        pid=pid,
        logfile=logfile,
    )


def initialise(*, force: bool = False) -> Path | None:
    """Initialises the debug runtime and creates a session log file when enabled."""
    with _STATE_LOCK:
        if _RT.initialised and not force:
            return _RT.settings.logfile if _RT.settings else None

        settings = read_settings()
        _RT.settings = settings
        if settings.enabled:
            _RT.logger = _build_logger(settings)
        else:
            # Hard-off mode: keep runtime state but remove all logger handlers so that
            # DEBUG=False guarantees no debug output side effects.
            logger = logging.getLogger("neuralqx.operator.symbolic.debug")
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
            _RT.logger = None
            _RT.buffer_handler = None
            _RT.file_handler = None
        _RT.initialised = True

    if settings.enabled:
        _write_session_header()
        with _STATE_LOCK:
            if not _RT.atexit_registered:
                atexit.register(_atexit_summary)
                _RT.atexit_registered = True

    return settings.logfile


def refresh_settings(*, reinit: bool = False) -> None:
    """Refreshes debug settings from :mod:`neuralqx.operator.symbolic.configs`."""
    if reinit:
        initialise(force=True)
        return

    with _STATE_LOCK:
        if not _RT.initialised:
            return
        old = _RT.settings

    latest = read_settings()
    if old is None:
        with _STATE_LOCK:
            _RT.settings = latest
        return

    merged = dataclasses.replace(
        old,
        enabled=latest.enabled,
        verbosity=latest.verbosity,
        scopes=latest.scopes,
        pass_filter=latest.pass_filter,
        log_to_file=latest.log_to_file,
        log_dir=latest.log_dir,
    )
    with _STATE_LOCK:
        _RT.settings = merged


def _ensure_init_if_needed() -> None:
    with _STATE_LOCK:
        ready = _RT.initialised
    if not ready:
        initialise(force=False)


def is_enabled() -> bool:
    with _STATE_LOCK:
        st = _RT.settings
    return bool(st and st.enabled)


def is_scope_enabled(scope: str, *, pass_name: str | None = None) -> bool:
    """Returns whether ``scope`` is currently enabled by debug settings."""
    _ensure_init_if_needed()
    with _STATE_LOCK:
        st = _RT.settings

    if st is None or not st.enabled:
        return False

    scope_norm = _normalize_scope(scope)
    scopes = {_normalize_scope(s) for s in st.scopes}

    if "all" in scopes:
        if scope_norm != "passes" or not st.pass_filter:
            return True
        if pass_name is None:
            return True
        return pass_name.strip().lower() in st.pass_filter

    if scope_norm not in scopes:
        if pass_name is not None and f"pass:{pass_name.strip().lower()}" in scopes:
            return True
        return False

    if scope_norm == "passes" and st.pass_filter and pass_name is not None:
        return pass_name.strip().lower() in st.pass_filter

    return True


def get_logfile() -> Path | None:
    with _STATE_LOCK:
        st = _RT.settings
    return st.logfile if st else None


def current_settings() -> DebugSettings | None:
    """Returns the active settings object, if initialised."""
    with _STATE_LOCK:
        return _RT.settings


def _write_session_header() -> None:
    with _STATE_LOCK:
        st = _RT.settings
    if not st or not st.enabled:
        return

    lines = [
        "=== neuralqx.operator.symbolic Debug Session ===",
        f"started: {_dt.datetime.now().astimezone().isoformat(timespec='milliseconds')}",
        f"logfile: {st.logfile}",
        f"rank/world: {st.rank}/{st.world}",
        f"pid: {st.pid}",
        f"python: {sys.version.replace(os.linesep, ' ')}",
        f"platform: {platform.platform()}",
        f"cwd: {os.getcwd()}",
        f"scopes: {', '.join(st.scopes)}",
        f"pass_filter: {', '.join(st.pass_filter) if st.pass_filter else '<none>'}",
        "===========================",
    ]
    for line in lines:
        _emit(logging.INFO, line, tag="SESSION")


def _split_lines(text: str, max_chars: int) -> Iterable[str]:
    chunk: list[str] = []
    size = 0
    for line in text.splitlines():
        if size + len(line) + 1 > max_chars and chunk:
            yield "\n".join(chunk)
            chunk = []
            size = 0
        chunk.append(line)
        size += len(line) + 1
    if chunk:
        yield "\n".join(chunk)


def _is_array_like(x: Any) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _summarize_array(x: Any, *, hard: bool) -> str:
    try:
        shape = getattr(x, "shape", None)
        dtype = getattr(x, "dtype", None)
        cls = type(x).__name__
        out = (
            f"{cls}(shape={tuple(shape) if shape is not None else None}, dtype={dtype})"
        )
        if hard:
            device = getattr(x, "device", None)
            if device is not None:
                out += f", device={device}"
        return out
    except Exception:
        return f"{type(x).__name__}(array-like)"


def _summarize_mapping(m: Mapping[Any, Any], *, hard: bool, max_items: int = 8) -> str:
    try:
        items = list(m.items())[:max_items]
        inner = ", ".join(
            f"{_summarize_value(k, hard=hard)}: {_summarize_value(v, hard=hard)}"
            for k, v in items
        )
        more = "" if len(m) <= max_items else f", …(+{len(m) - max_items})"
        return f"{type(m).__name__}(len={len(m)}, {{{inner}{more}}})"
    except Exception:
        return f"{type(m).__name__}(mapping)"


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
        try:
            items = list(v)[:8]
            inner = ", ".join(_summarize_value(x, hard=hard) for x in items)
            more = "" if len(v) <= 8 else f", …(+{len(v) - 8})"
            return f"{type(v).__name__}(len={len(v)}, [{inner}{more}])"
        except Exception:
            return f"{type(v).__name__}(sequence)"
    try:
        rep = repr(v).replace("\n", "\\n")
        return rep[:max_str] + ("…" if len(rep) > max_str else "")
    except Exception:
        return f"<{type(v).__name__}>"


def _summarize_args(
    args: tuple[Any, ...], kwargs: dict[str, Any], *, hard: bool
) -> str:
    parts: list[str] = []
    if args:
        parts.append("args=" + _summarize_value(args, hard=hard))
    if kwargs:
        parts.append("kwargs=" + _summarize_mapping(kwargs, hard=hard))
    return " ".join(parts)


def dump_recent_events(*, n: int = 200, tag: str = "DUMP") -> None:
    """Logs the last ``n`` buffered events."""
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


def recent_event_count() -> int:
    """Returns the number of buffered recent debug events."""
    return len(_EVENT_BUFFER)


def log_once(level: int, key: str, msg: str, *, tag: str | None = None) -> None:
    """Emits one log record per process for the given ``key``."""
    with _ONCE_KEYS_LOCK:
        if key in _ONCE_KEYS:
            return
        _ONCE_KEYS.add(key)
    _emit(level, msg, tag=tag)


def event(
    msg: str,
    *,
    scope: str | None = None,
    pass_name: str | None = None,
    tag: str | None = None,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Structured debug event with scope and optional pass filtering."""
    _ensure_init_if_needed()

    if not _should_emit(level):
        return
    if scope is not None and not is_scope_enabled(scope, pass_name=pass_name):
        return

    payload_fields: dict[str, Any] = dict(fields)
    if scope is not None:
        payload_fields.setdefault("scope", _normalize_scope(scope))
    if pass_name is not None:
        payload_fields.setdefault("pass_name", pass_name)

    if payload_fields:
        kv = " ".join(
            f"{k}={_summarize_value(v, hard=True)}" for k, v in payload_fields.items()
        )
        _emit(level, f"{msg} | {kv}", tag=tag)
    else:
        _emit(level, msg, tag=tag)


def _find_external_caller() -> str:
    frame = None
    try:
        this_file = __file__
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back
        if frame is not None:
            frame = frame.f_back
        while frame:
            filename = frame.f_code.co_filename
            if filename != this_file and "functools" not in filename:
                return f"{os.path.basename(filename)}:{frame.f_lineno} in {frame.f_code.co_name}"
            frame = frame.f_back
    except Exception:
        pass
    finally:
        # Break reference cycles created by frame objects.
        del frame
    return "unknown"


_CALL_SEQ = 0
_CALL_SEQ_LOCK = threading.Lock()


def _next_call_id() -> str:
    global _CALL_SEQ
    with _CALL_SEQ_LOCK:
        _CALL_SEQ += 1
        seq = _CALL_SEQ
    with _STATE_LOCK:
        st = _RT.settings
    rank = st.rank if st else 0
    return f"{rank}-{seq}"


def trace(
    _fn: Callable[..., _T] | None = None,
    *,
    tag: str | None = None,
    log_args_at: int = logging.CRITICAL,
    log_return_at: int = logging.DEBUG,
    warn_slow_ms: float | None = 500.0,
    reraise: bool = True,
) -> Callable[..., _T] | Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Full-spectrum function-call instrumentation decorator."""

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
            token_depth = _CALL_DEPTH.set(depth + 1)
            token_id = _CALL_ID.set(call_id)
            token_func = _CALL_FUNC.set(qual)
            token_tag = _CALL_TAG.set(tag or "")

            start_ns = time.perf_counter_ns()
            ok = False
            try:
                if _should_emit(logging.INFO):
                    extra = ""
                    if _should_emit(log_args_at):
                        summarized = _summarize_args(
                            args, kwargs, hard=_should_emit(logging.CRITICAL)
                        )
                        extra = f" | {summarized}" if summarized else ""
                    _emit(
                        logging.INFO,
                        f"CALL {qual} caller={caller} cid={call_id}{extra}",
                        tag=tag,
                    )

                out = fn(*args, **kwargs)
                ok = True
                return out
            except Exception as exc:
                duration_ns = time.perf_counter_ns() - start_ns
                _emit(
                    logging.ERROR,
                    (
                        f"EXC  {qual} cid={call_id} after={duration_ns / 1e6:.3f}ms "
                        f"caller={caller} exc={type(exc).__name__}: {exc}"
                    ),
                    tag=tag,
                )
                if _should_emit(logging.ERROR):
                    tb = traceback.format_exc()
                    for chunk in _split_lines(tb, 2000):
                        _emit(logging.ERROR, chunk, tag=tag)
                with _STATS_LOCK:
                    _STATS_ERRORS[qual] += 1
                if reraise:
                    raise
                return None  # type: ignore[return-value]
            finally:
                duration_ns = time.perf_counter_ns() - start_ns
                with _STATS_LOCK:
                    _STATS_CALLS[qual] += 1
                    _STATS_TOTAL_NS[qual] += duration_ns
                    _STATS_MAX_NS[qual] = max(_STATS_MAX_NS[qual], duration_ns)

                if ok and _should_emit(logging.DEBUG):
                    message = (
                        f"RET  {qual} cid={call_id} ok after={duration_ns / 1e6:.3f}ms"
                    )
                    if _should_emit(log_return_at):
                        rv = locals().get("out", None)
                        message += f" | return={_summarize_value(rv, hard=_should_emit(logging.CRITICAL))}"
                    _emit(logging.DEBUG, message, tag=tag)

                if (
                    ok
                    and warn_slow_ms is not None
                    and duration_ns / 1e6 >= warn_slow_ms
                    and _should_emit(logging.WARNING)
                ):
                    _emit(
                        logging.WARNING,
                        (
                            f"SLOW {qual} cid={call_id} took={duration_ns / 1e6:.3f}ms "
                            f"threshold={warn_slow_ms:.1f}ms caller={caller}"
                        ),
                        tag=tag,
                    )

                _CALL_TAG.reset(token_tag)
                _CALL_FUNC.reset(token_func)
                _CALL_ID.reset(token_id)
                _CALL_DEPTH.reset(token_depth)

        return wrapper

    return deco if _fn is None else deco(_fn)


def timeit(
    _fn: Callable[..., _T] | None = None,
    *,
    tag: str | None = None,
    warn_ms: float | None = 250.0,
    level: int = logging.DEBUG,
) -> Callable[..., _T] | Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Lightweight timing decorator."""

    def deco(fn: Callable[..., _T]) -> Callable[..., _T]:
        qual = f"{fn.__module__}.{getattr(fn, '__qualname__', fn.__name__)}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            _ensure_init_if_needed()
            if not is_enabled():
                return fn(*args, **kwargs)

            t0 = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
                if _should_emit(level):
                    _emit(level, f"TIME {qual} took={elapsed_ms:.3f}ms", tag=tag)
                if (
                    warn_ms is not None
                    and elapsed_ms >= warn_ms
                    and _should_emit(logging.WARNING)
                ):
                    _emit(
                        logging.WARNING,
                        f"SLOW {qual} took={elapsed_ms:.3f}ms threshold={warn_ms:.1f}ms",
                        tag=tag,
                    )

        return wrapper

    return deco if _fn is None else deco(_fn)


def errors_only(
    _fn: Callable[..., _T] | None = None,
    *,
    tag: str | None = None,
    reraise: bool = True,
) -> Callable[..., _T] | Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorator that logs only exceptions."""

    def deco(fn: Callable[..., _T]) -> Callable[..., _T]:
        qual = f"{fn.__module__}.{getattr(fn, '__qualname__', fn.__name__)}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            _ensure_init_if_needed()
            if not is_enabled():
                return fn(*args, **kwargs)

            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                _emit(
                    logging.ERROR, f"EXC {qual}: {type(exc).__name__}: {exc}", tag=tag
                )
                if _should_emit(logging.ERROR):
                    tb = traceback.format_exc()
                    for chunk in _split_lines(tb, 2000):
                        _emit(logging.ERROR, chunk, tag=tag)
                with _STATS_LOCK:
                    _STATS_ERRORS[qual] += 1
                if reraise:
                    raise
                return None  # type: ignore[return-value]

        return wrapper

    return deco if _fn is None else deco(_fn)


def io_trace(
    _fn: Callable[..., _T] | None = None,
    *,
    tag: str | None = None,
    at_level: int = logging.DEBUG,
) -> Callable[..., _T] | Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorator that logs summarized input and output values."""

    def deco(fn: Callable[..., _T]) -> Callable[..., _T]:
        qual = f"{fn.__module__}.{getattr(fn, '__qualname__', fn.__name__)}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            _ensure_init_if_needed()
            if not is_enabled():
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


def _atexit_summary() -> None:
    try:
        with _STATE_LOCK:
            st = _RT.settings
        if not st or not st.enabled or st.verbosity > logging.INFO:
            return

        with _STATS_LOCK:
            calls = dict(_STATS_CALLS)
            errs = dict(_STATS_ERRORS)
            total_ns = dict(_STATS_TOTAL_NS)
            max_ns = dict(_STATS_MAX_NS)

        if not calls:
            return

        _emit(logging.INFO, "=== Debug summary ===", tag="SUMMARY")

        top_total = sorted(
            calls.keys(), key=lambda name: total_ns.get(name, 0), reverse=True
        )[:10]
        _emit(logging.INFO, "Top total time:", tag="SUMMARY")
        for name in top_total:
            _emit(
                logging.INFO,
                (
                    f"{name} calls={calls[name]} total_ms={total_ns[name] / 1e6:.1f} "
                    f"max_ms={max_ns[name] / 1e6:.1f} errors={errs.get(name, 0)}"
                ),
                tag="SUMMARY",
            )

        if errs:
            top_err = sorted(errs.items(), key=lambda kv: kv[1], reverse=True)[:10]
            _emit(logging.INFO, "Top errors:", tag="SUMMARY")
            for name, count in top_err:
                _emit(logging.INFO, f"{name} errors={count}", tag="SUMMARY")

        _emit(logging.INFO, "=== End debug summary ===", tag="SUMMARY")
    except Exception:
        pass


@contextlib.contextmanager
def tagged(tag: str):
    """Temporarily sets a default tag for nested logs."""
    token = _CALL_TAG.set(tag)
    try:
        yield
    finally:
        _CALL_TAG.reset(token)


__all__ = [
    "current_settings",
    "dump_recent_events",
    "errors_only",
    "event",
    "get_logfile",
    "initialise",
    "io_trace",
    "is_enabled",
    "is_scope_enabled",
    "log_once",
    "recent_event_count",
    "refresh_settings",
    "tagged",
    "timeit",
    "trace",
]
