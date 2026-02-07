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
Some MPI helpers or functionalities which have been adapted to be MPI friendly
"""

import os
import platform

import sys
import types

from tqdm.auto import tqdm as _TQDM_ORIG

from .core import (
    all_hostnames,
    available,
    MPI,
    comm,
    detect_cpus_per_task,
    hostname,
    is_master,
    local_ranks,
    n_nodes,
    rank,
)
from ... import cfg

"""
MPI-aware tqdm wrapper which works for both `tqdm`  and  `tqdm.auto`
"""


def tqdm(*args, _orig=_TQDM_ORIG, **kwargs):
    """
    MPI-aware tqdm:
      - rank-0: normal animated bar
      - workers: dummy bar, all method calls are safe, but nothing is drawn
    """
    kwargs.setdefault("disable", rank != 0)
    return _orig(*args, **kwargs)


# Monkey-patch both `tqdm` and `tqdm.auto` so every future import, no matter the order, receives the
# MPI-aware version


def _patch(module_name: str) -> None:
    """
    Ensure that `import <module_name>; <module_name>.tqdm` is our wrapper.
    Creates a shim if the module hasn't been imported yet.
    """

    mod = sys.modules.get(module_name)
    if isinstance(mod, types.ModuleType):
        # module already imported: patch in place
        mod.tqdm = tqdm
    else:
        # create lightweight shim so the first import gets our wrapper
        shim = types.ModuleType(module_name)
        shim.tqdm = tqdm
        sys.modules[module_name] = shim


# top-level `import tqdm`
_patch("tqdm")

# `from tqdm.auto import tqdm`
_patch("tqdm.auto")

# tidy globals, should be safe, wrapper holds ref
del sys, types, _patch, _TQDM_ORIG


def check_mpi(
    return_dict: bool = False,
    extended: bool = True,
) -> dict[str, str] | None:
    """
    Print (or return) diagnostic information about the neuraLQX MPI runtime.
    Uses the MPI utilities from ._core.
    """

    # no MPI requested
    if not available:
        if return_dict:
            return {"mpi_enabled": "False"}

        if is_master():
            header = "MPI Summary"
            status_msg = "MPI is disabled"
            total_width = max(len(header), len(status_msg)) + 20

            print("=" * total_width)
            print(f"{header:^{total_width}}")
            print("=" * total_width)
            print(f"{status_msg:^{total_width}}")
            print("=" * total_width)
            print()

        return None

    # prepare info dictionary
    info: dict[str, str] = {}
    info["mpi4py_available"] = str(available)

    try:
        import mpi4jax

        info["mpi4jax_available"] = "True"
    except Exception:
        info["mpi4jax_available"] = "False"

    # MPI ranks and nodes
    size = comm.Get_size()
    hostnames = all_hostnames()
    unique_nodes = set(hostnames)

    info["n_ranks"] = str(size)
    info["n_nodes"] = str(len(unique_nodes))

    # CPU information
    cpus_per_node = os.cpu_count() or 1
    cpus_per_task = detect_cpus_per_task()
    total_cpus = size * cpus_per_task

    info["cpus_per_node (rank 0)"] = str(cpus_per_node)
    info["cpus_per_task"] = str(cpus_per_task)
    info["total_cpu_cores_used"] = str(total_cpus)

    # GPU information (rank 0 only)
    autodetect = os.environ.get("NETKET_MPI_AUTODETECT_LOCAL_GPU", 0) == "1"
    info["gpu_allocation_mode"] = (
        "NETKET_AUTODETECT" if autodetect else "JAX_VISIBLE_DEVICES"
    )

    jax_gpu_count_rank0 = 0
    try:
        import jax

        # prefer filtering by platform to handle older/newer JAX versions consistently
        jax_gpus = [d for d in jax.devices() if getattr(d, "platform", None) == "gpu"]
        jax_gpu_count_rank0 = len(jax_gpus)
        jax_ok = True
    except Exception:
        jax_ok = False

    info["jax_available"] = "True" if jax_ok else "False"
    # info["available_gpus (rank 0, JAX)"] = str(jax_gpu_count_rank0)

    # if NetKet autodetect is used, enforce a sanity check with JAX:
    # - if JAX sees no GPUs, we must not claim 1 GPU/rank.
    gpu_sanity_ok = True
    if autodetect:
        if jax_ok and jax_gpu_count_rank0 >= 1:
            info["gpus_per_rank (autodetect)"] = "1"
        else:
            info["gpus_per_rank (autodetect)"] = "0"
            gpu_sanity_ok = False

    info["gpu_sanity_check"] = (
        "OK" if gpu_sanity_ok else "FAIL (autodetect requested but Jax sees no GPU)"
    )

    # parse total GPUs allocated from environment/SLURM if possible
    slurm_gpus_env = (
        os.environ.get("SLURM_GPUS")
        or os.environ.get("SLURM_GPUS_ON_NODE")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
    )

    def _parse_gpu_env(val: str | None) -> str:
        if not val:
            return "unknown"
        # CUDA_VISIBLE_DEVICES is comma-separated, SLURM_GPUS often a number
        if "," in val:
            try:
                return str(len([v for v in val.split(",") if v != ""]))
            except Exception:
                return "unknown"
        try:
            # handles plain integers
            return str(int(val))
        except Exception:
            return "unknown"

    total_gpus_allocated = _parse_gpu_env(slurm_gpus_env)
    info["total_gpus_allocated (env)"] = total_gpus_allocated

    # MPI library information
    v_major, v_minor = MPI.Get_version()
    info["mpi4py | MPI version"] = f"({v_major}, {v_minor})"
    try:
        info["mpi4py | MPI library_version"] = MPI.Get_library_version().rstrip("\x00")
    except AttributeError:
        info["mpi4py | MPI library_version"] = "unknown"

    # CUDA-aware info (from cfg)
    try:
        cuda_aware = bool(cfg.get("MPI_CUDA"))
        info["CUDA-aware MPI"] = "True (ENV_VAR)" if cuda_aware else "False (ENV_VAR)"
    except Exception:
        info["CUDA-aware MPI"] = "Unknown (ENV_VAR)"

    # Python environment
    if is_master():
        info["python_implementation"] = platform.python_implementation()
        info["python_version"] = platform.python_version()

    if return_dict:
        return info

    if is_master():
        summary: dict[str, str | int] = {
            "Number of nodes": len(unique_nodes),
            "Total MPI ranks (tasks)": size,
            "Total CPUs available per node (rank 0)": cpus_per_node,
            "CPUs per MPI task": cpus_per_task,
            "Total CPU cores used across ranks": total_cpus,
            "GPU allocation mode": info["gpu_allocation_mode"],
        }

        if autodetect:
            summary["GPUs per rank (NetKet autodetect)"] = info[
                "gpus_per_rank (autodetect)"
            ]

        summary["Total GPUs allocated (env/Slurm)"] = total_gpus_allocated

        # table formatting
        max_key_len = max(len(k) for k in summary)
        max_val_len = max(len(str(v)) for v in summary.values())
        total_width = 2 + max_key_len + 3 + max_val_len

        header = "MPI Summary"
        padding = max(0, (total_width - len(header)) // 2)
        print("=" * (total_width + 1))
        print(" " * padding + header)
        print("=" * (total_width + 1))
        for k, v in summary.items():
            print(f"  {k:<{max_key_len}} : {v}")
        print("=" * (total_width + 1))

        # if autodetect was requested but JAX found no GPUs, add a clear warning.
        if autodetect and not gpu_sanity_ok:
            warn = (
                "WARNING: NETKET_MPI_AUTODETECT_LOCAL_GPU=1 was set, "
                "but Jax reports 0 GPUs on rank 0. Running on CPU."
            )
            print(f"\n{warn}\n")

        # optionally print the detailed dump
        if extended:
            print("\nDetails:\n" + "-" * (total_width + 1))
            longest = max(len(k) for k in info)
            for key in sorted(info.keys()):
                print(f"{key:<{longest}} : {info[key]}")
            print("-" * (total_width + 1) + "\n")

    return None


def get_mpi_info_dict() -> dict:
    """
    Returns detailed MPI information for serialization
    """
    return {
        "enabled": available,
        "n_nodes": len(set(all_hostnames())) if available else 1,
        "ranks_per_node": local_ranks() if available else 1,
    }
