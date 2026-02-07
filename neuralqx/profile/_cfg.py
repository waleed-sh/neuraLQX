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

import os
from dataclasses import dataclass

from typing import Any
from typing import Optional


def _try_import_cfg():
    """
    Try to import neuralqx.cfg without hard dependency.
    Returns the module or None.
    """
    try:
        from neuralqx import cfg  # type: ignore

        return cfg
    except Exception:
        return None


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off", ""}:
            return False
    return default


def _as_int(v: Any, default: int) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _as_float(v: Any, default: float) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env(name: str) -> Optional[str]:
    return os.environ.get(name)


def profiling_enabled() -> bool:
    """
    True if profiling is enabled.

    Important:
      - This function is intentionally NOT cached.

    We prefer reading the environment variable directly because neuraLQX's
    ConfigManager always keeps `os.environ["NQX_PROFILE"]` in sync.
    """
    ev = _env("NQX_PROFILE")
    if ev is not None:
        return _as_bool(ev, default=False)

    # fallback: try cfg if env var does not exist (rare)
    cfg = _try_import_cfg()
    if cfg is not None:
        try:
            return _as_bool(cfg.get("PROFILE"), default=False)
        except Exception:
            return False

    return False


def get_output_dir() -> str:
    """
    Directory to store profiling artifacts. Must be safe for distributed runs.

    Priority:
      1) cfg.get_static("Profiling Directory")
      2) env NQX_PROFILE_DIR
      3) ./nqx_profiles
    """
    cfg = _try_import_cfg()
    if cfg is not None:
        try:
            d = cfg.get_static("Profiling Directory")
            if d:
                return str(d)
        except Exception:
            pass
    d = _env("NQX_PROFILE_DIR")
    if d:
        return d
    return os.path.abspath("nqx_profiles")


def _get_cfg_value(key: str) -> Any:
    cfg = _try_import_cfg()
    if cfg is None:
        return None
    try:
        return cfg.get(key)
    except Exception:
        return None


def get_bool(key: str, default: bool = False, env_key: Optional[str] = None) -> bool:
    v = _get_cfg_value(key)
    if v is not None:
        return _as_bool(v, default=default)
    if env_key is None:
        env_key = "NQX_" + key
    return _as_bool(_env(env_key), default=default)


def get_int(key: str, default: int, env_key: Optional[str] = None) -> int:
    v = _get_cfg_value(key)
    if v is not None:
        return _as_int(v, default)
    if env_key is None:
        env_key = "NQX_" + key
    return _as_int(_env(env_key), default)


def get_float(key: str, default: float, env_key: Optional[str] = None) -> float:
    v = _get_cfg_value(key)
    if v is not None:
        return _as_float(v, default)
    if env_key is None:
        env_key = "NQX_" + key
    return _as_float(_env(env_key), default)


def get_str(key: str, default: str = "", env_key: Optional[str] = None) -> str:
    v = _get_cfg_value(key)
    if v is not None:
        try:
            return str(v)
        except Exception:
            return default
    if env_key is None:
        env_key = "NQX_" + key
    ev = _env(env_key)
    return ev if ev is not None else default


@dataclass(frozen=True)
class ProfilingConfig:
    enabled: bool
    trace: bool
    nvtx: bool
    jax_trace_annotations: bool
    jax_profiler_trace: bool
    metrics: bool
    sync: bool
    max_events: int
    sample_period_s: float
    mpi_aggregate_on_exit: bool

    @staticmethod
    def load() -> "ProfilingConfig":
        enabled = profiling_enabled()

        # these can be controlled even when enabled is False, they will be no-ops
        trace = get_bool("PROFILE_TRACE", default=True, env_key="NQX_PROFILE_TRACE")
        nvtx = get_bool("PROFILE_NVTX", default=False, env_key="NQX_PROFILE_NVTX")
        jax_trace_annotations = get_bool(
            "PROFILE_JAX_ANNOTATE", default=True, env_key="NQX_PROFILE_JAX_ANNOTATE"
        )
        jax_profiler_trace = get_bool(
            "PROFILE_JAX_TRACE", default=False, env_key="NQX_PROFILE_JAX_TRACE"
        )
        metrics = get_bool(
            "PROFILE_METRICS", default=False, env_key="NQX_PROFILE_METRICS"
        )
        sync = get_bool("PROFILE_SYNC", default=False, env_key="NQX_PROFILE_SYNC")
        max_events = get_int(
            "PROFILE_MAX_EVENTS", default=2_000_000, env_key="NQX_PROFILE_MAX_EVENTS"
        )
        sample_period_s = get_float(
            "PROFILE_SAMPLE_PERIOD_S",
            default=0.1,
            env_key="NQX_PROFILE_SAMPLE_PERIOD_S",
        )

        # MPI aggregation at exit can block (waiting for slow ranks). Default OFF for "no delay"
        mpi_aggregate = get_bool(
            "PROFILE_MPI_AGG", default=False, env_key="NQX_PROFILE_MPI_AGG"
        )

        return ProfilingConfig(
            enabled=enabled,
            trace=trace,
            nvtx=nvtx,
            jax_trace_annotations=jax_trace_annotations,
            jax_profiler_trace=jax_profiler_trace,
            metrics=metrics,
            sync=sync,
            max_events=max_events,
            sample_period_s=sample_period_s,
            mpi_aggregate_on_exit=mpi_aggregate,
        )
