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

"""
Distributed runtime helpers based on JAX process semantics.
"""

import inspect
import os
import platform
from dataclasses import dataclass
from typing import Any


def _get_env_int(keys: list[str], default: int) -> int:
    for k in keys:
        v = os.environ.get(k, "").strip()
        if not v:
            continue
        try:
            return int(v)
        except Exception:
            continue
    return int(default)


def detect_cpus_per_task() -> int:
    return _get_env_int(
        [
            "SLURM_CPUS_PER_TASK",
            "PBS_NCPUS",
            "OMP_NUM_THREADS",
        ],
        default=1,
    )


@dataclass(frozen=True)
class RuntimeInfo:
    """Runtime information for distributed execution."""

    backend: str
    rank: int
    size: int
    local_rank: int
    jax_available: bool
    multihost_utils_available: bool


def runtime_info() -> RuntimeInfo:
    """
    Detect runtime backend and process identity.

    Priority:
    - JAX process semantics when available.
    - Serial fallback otherwise.
    """

    try:
        import jax  # type: ignore

        rank_ = int(jax.process_index())
        size_ = int(jax.process_count())
        jax_ok = True
    except Exception:
        rank_, size_ = 0, 1
        jax_ok = False

    mh_ok = False
    if jax_ok:
        try:
            from jax.experimental import multihost_utils  # type: ignore

            del multihost_utils
            mh_ok = True
        except Exception:
            mh_ok = False

    if jax_ok and size_ > 1:
        backend = "jax"
    else:
        backend = "serial"

    local_rank_ = _get_env_int(
        [
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "SLURM_LOCALID",
            "LOCAL_RANK",
            "MPI_LOCALRANKID",
            "JAX_LOCAL_PROCESS_RANK",
        ],
        default=rank_ if size_ == 1 else 0,
    )

    return RuntimeInfo(
        backend=backend,
        rank=rank_,
        size=size_,
        local_rank=local_rank_,
        jax_available=jax_ok,
        multihost_utils_available=mh_ok,
    )


def process_index() -> int:
    """Global process index (rank-equivalent)."""

    return runtime_info().rank


def process_count() -> int:
    """Global process count (world-size equivalent)."""

    return runtime_info().size


def is_global_master() -> bool:
    """Return ``True`` when this process is global rank 0."""

    return process_index() == 0


def is_master() -> bool:
    """
    Alias for master-process check.

    Kept for migration compatibility with the previous MPI helpers.
    """

    return is_global_master()


def barrier(name: str = "neuralqx:distributed:barrier") -> None:
    """
    Global synchronisation across all JAX processes when available.

    Serial mode and missing multihost utilities degrade to a no-op.
    """

    info = runtime_info()
    if info.size <= 1:
        return

    try:
        from jax.experimental import multihost_utils as _mh  # type: ignore

        _mh.sync_global_devices(name)
    except Exception:
        # Keep barrier best-effort and never crash callers.
        return


def allgather(value: Any) -> Any:
    """
    Gather ``value`` from all processes.

    In serial mode this returns a 1-tuple ``(value,)``.
    """

    info = runtime_info()
    if info.size <= 1:
        return (value,)

    try:
        from jax.experimental import multihost_utils as _mh  # type: ignore

        return _mh.process_allgather(value)
    except Exception:
        return (value,)


def bcast(value: Any, *, root: int = 0) -> Any:
    """
    Broadcast ``value`` from ``root`` to all processes.

    Uses JAX multihost utils when available. Falls back to an allgather-based
    path.
    """

    info = runtime_info()
    if info.size <= 1:
        return value

    try:
        from jax.experimental import multihost_utils as _mh  # type: ignore

        fn = _mh.broadcast_one_to_all
        kwargs = {}
        sig = inspect.signature(fn)
        if "is_source" in sig.parameters:
            kwargs["is_source"] = info.rank == int(root)
        payload = value if info.rank == int(root) else None
        return fn(payload, **kwargs)
    except Exception:
        gathered = allgather(value if info.rank == int(root) else None)
        try:
            return gathered[int(root)]
        except Exception:
            return value


def mpi_any(value: Any) -> bool:
    """
    Logical OR across processes for a boolean-like value.

    Returned type is a Python ``bool``.
    """

    info = runtime_info()
    if info.size <= 1:
        return bool(value)

    try:
        gathered = allgather(bool(value))
        return any(bool(v) for v in gathered)
    except Exception:
        return bool(value)


def _stack_reduce(x: Any, *, op: str):
    """
    Reduce one JAX leaf by process-allgather + local reduction.
    """

    import jax.numpy as jnp  # type: ignore

    gathered = allgather(x)
    arr = jnp.asarray(gathered)
    if arr.ndim == 0:
        return arr
    if op == "sum":
        return jnp.sum(arr, axis=0)
    if op == "mean":
        return jnp.mean(arr, axis=0)
    raise ValueError(f"Unsupported reduction op `{op}`")


def _tree_reduce(x: Any, *, op: str) -> Any:
    """
    Reduce an arbitrary pytree across processes.
    """

    info = runtime_info()
    if info.size <= 1:
        return x

    try:
        import jax  # type: ignore

        return jax.tree_util.tree_map(lambda leaf: _stack_reduce(leaf, op=op), x)
    except Exception:
        return _stack_reduce(x, op=op)


def mpi_sum_jax(x: Any, *, token: Any = None, communicator=None):  # noqa: ARG001
    """
    Compatibility helper: sum value/pytree across processes.

    Returns ``(reduced, token)`` to mirror previous ``mpi_sum_jax`` signature.
    """

    return _tree_reduce(x, op="sum"), token


def mpi_mean_jax(x: Any, *, token: Any = None, communicator=None):  # noqa: ARG001
    """
    Compatibility helper: mean value/pytree across processes.

    Returns ``(reduced, token)`` to mirror previous ``mpi_mean_jax`` signature.
    """

    return _tree_reduce(x, op="mean"), token


def print0(*args, **kwargs) -> None:
    """Print from global process 0 only."""

    if is_global_master():
        print(*args, **kwargs)


def get_distributed_info_dict() -> dict[str, Any]:
    """Small serialisable runtime descriptor for checkpoint metadata."""

    info = runtime_info()
    hostnames = allgather(platform.node())
    try:
        n_hosts = len(set(hostnames))
        processes_on_host = sum(1 for h in hostnames if h == platform.node())
    except Exception:
        n_hosts = 1
        processes_on_host = 1

    return {
        "enabled": info.size > 1,
        "backend": info.backend,
        "rank": info.rank,
        "size": info.size,
        "local_rank": info.local_rank,
        "n_hosts": n_hosts,
        "processes_on_host": processes_on_host,
    }


def _parse_gpu_env_count(val: str | None) -> str:
    if not val:
        return "unknown"
    if "," in val:
        try:
            return str(len([v for v in val.split(",") if v != ""]))
        except Exception:
            return "unknown"
    try:
        return str(int(val))
    except Exception:
        return "unknown"


def check_distributed(
    return_dict: bool = False,
    extended: bool = True,
) -> dict[str, str] | None:
    """
    Print (or return) diagnostic information about the distributed runtime.

    Works for both CPU-only and GPU-enabled environments.
    """

    info_rt = runtime_info()

    details: dict[str, str] = {}
    details["distributed_enabled"] = "True" if info_rt.size > 1 else "False"
    details["execution_mode"] = (
        "multi-process distributed"
        if info_rt.size > 1
        else "single-process (serial collectives)"
    )
    details["backend"] = str(info_rt.backend)
    details["jax_available"] = str(info_rt.jax_available)
    details["multihost_utils_available"] = str(info_rt.multihost_utils_available)

    hostnames = allgather(platform.node())
    try:
        unique_hosts = sorted({str(h) for h in hostnames})
    except Exception:
        unique_hosts = [platform.node()]

    details["n_processes"] = str(info_rt.size)
    details["n_hosts"] = str(len(unique_hosts))
    details["processes_on_host (this host)"] = str(
        sum(1 for h in hostnames if h == platform.node())
    )
    details["hostname (this process)"] = platform.node()
    details["local_rank"] = str(info_rt.local_rank)

    cpus_per_node = os.cpu_count() or 1
    cpus_per_task = detect_cpus_per_task()
    total_cpus = int(info_rt.size) * int(cpus_per_task)
    details["cpus_per_node (rank 0)"] = str(cpus_per_node)
    details["cpus_per_task"] = str(cpus_per_task)
    details["total_cpu_cores_used"] = str(total_cpus)

    jax_ok = False
    jax_gpu_count_rank0 = 0
    jax_local_gpu_count = 0
    jax_local_device_count = 0
    jax_global_device_count = 0
    jax_default_backend = "unknown"
    try:
        import jax  # type: ignore

        jax_default_backend = str(jax.default_backend())
        jax_local_devices = list(jax.local_devices())
        jax_global_devices = list(jax.devices())
        jax_local_device_count = len(jax_local_devices)
        jax_global_device_count = len(jax_global_devices)
        jax_local_gpu_count = len(
            [d for d in jax_local_devices if getattr(d, "platform", None) == "gpu"]
        )
        jax_gpus = [d for d in jax.devices() if getattr(d, "platform", None) == "gpu"]
        jax_gpu_count_rank0 = len(jax_gpus)
        jax_ok = True
    except Exception:
        jax_ok = False

    details["jax_runtime_importable"] = "True" if jax_ok else "False"
    details["jax_default_backend"] = jax_default_backend
    details["jax_local_device_count"] = str(jax_local_device_count)
    details["jax_global_device_count"] = str(jax_global_device_count)
    details["jax_local_gpu_count"] = str(jax_local_gpu_count)
    details["jax_gpu_count (rank 0)"] = str(jax_gpu_count_rank0)

    gpu_env = (
        os.environ.get("SLURM_GPUS")
        or os.environ.get("SLURM_GPUS_ON_NODE")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    details["total_gpus_allocated (env)"] = _parse_gpu_env_count(gpu_env)
    details["cuda_visible_devices (env)"] = os.environ.get(
        "CUDA_VISIBLE_DEVICES", "unset"
    )
    details["jax_platform_name (env)"] = os.environ.get("JAX_PLATFORM_NAME", "unset")
    details["nqx_jax_distributed (env)"] = os.environ.get("NQX_JAX_DISTRIBUTED", "0")

    if return_dict:
        return details

    if is_master():
        dist_enabled = info_rt.size > 1
        if not dist_enabled and jax_local_gpu_count > 1:
            status_msg = (
                "Single process with multiple local GPUs (collectives are serial)."
            )
        elif not dist_enabled:
            status_msg = "Single-process execution (serial collectives)."
        else:
            status_msg = "Distributed runtime is enabled."

        summary: dict[str, str | int] = {
            "Status": status_msg,
            "Execution mode": details["execution_mode"],
            "Number of hosts": len(unique_hosts),
            "Total processes": info_rt.size,
            "Backend": info_rt.backend,
            "JAX default backend": jax_default_backend,
            "JAX local devices": jax_local_device_count,
            "JAX local GPUs": jax_local_gpu_count,
            "JAX global devices": jax_global_device_count,
            "CPUs per task": cpus_per_task,
            "Total CPU cores used": total_cpus,
            "GPUs visible to JAX (rank 0)": jax_gpu_count_rank0,
            "Total GPUs allocated (env/Slurm)": details["total_gpus_allocated (env)"],
        }

        max_key_len = max(len(k) for k in summary)
        max_val_len = max(len(str(v)) for v in summary.values())
        total_width = 2 + max_key_len + 3 + max_val_len

        header = "Distributed Summary"
        padding = max(0, (total_width - len(header)) // 2)
        print("=" * (total_width + 1))
        print(" " * padding + header)
        print("=" * (total_width + 1))
        for k, v in summary.items():
            print(f"  {k:<{max_key_len}} : {v}")
        print("=" * (total_width + 1))

        if extended:
            print("\nDetails:\n" + "-" * (total_width + 1))
            longest = max(len(k) for k in details)
            for key in sorted(details.keys()):
                print(f"{key:<{longest}} : {details[key]}")
            print("-" * (total_width + 1) + "\n")

    return None


_SNAPSHOT = runtime_info()
available: bool = _SNAPSHOT.size > 1
backend: str = _SNAPSHOT.backend
rank: int = _SNAPSHOT.rank
n_nodes: int = _SNAPSHOT.size
node_number: int = _SNAPSHOT.rank
local_rank: int = _SNAPSHOT.local_rank

# Keep old MPI naming for easier search/replace migrations for now
mpi_bcast = bcast

__all__ = [
    "RuntimeInfo",
    "available",
    "backend",
    "rank",
    "n_nodes",
    "node_number",
    "local_rank",
    "runtime_info",
    "process_index",
    "process_count",
    "is_global_master",
    "is_master",
    "barrier",
    "allgather",
    "bcast",
    "mpi_bcast",
    "mpi_any",
    "mpi_sum_jax",
    "mpi_mean_jax",
    "print0",
    "detect_cpus_per_task",
    "check_distributed",
    "get_distributed_info_dict",
]
