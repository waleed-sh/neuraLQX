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
Instrumentation and profiling tools for neuraLQX workflows.

Typical usage (public API):
    from neuralqx.profiling import profile, section, step

    @profile(cat="vmc")
    def vmc_iteration(...):
        with section("sampling", cat="vmc"):
            ...
        with section("expectation", cat="vmc"):
            ...

    for i in range(n_steps):
        with step(i, name="VMC"):
            vmc_iteration(...)

Configuration:
- Environment variables:
    NQX_PROFILE=1
    NQX_PROFILE_DIR=/path
    NQX_PROFILE_TRACE=1
    NQX_PROFILE_NVTX=1
    NQX_PROFILE_JAX_TRACE=1
    NQX_PROFILE_METRICS=1
    NQX_PROFILE_SYNC=1
"""

from .profiler import get_profiler, Profiler
from .decorators import profile, section, step
from ._cfg import profiling_enabled as enabled


def flush() -> None:
    """
    Force writing profiling artifacts now.

    In scripts the profiler flushes automatically at process exit.
    In Jupyter notebooks the kernel might stay alive, so call this
    to create output files immediately.
    """
    try:
        get_profiler().flush()
    except Exception:
        pass


__all__ = [
    "Profiler",
    "get_profiler",
    "enabled",
    "flush",
    "profile",
    "section",
    "step",
]
