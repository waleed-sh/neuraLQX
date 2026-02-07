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

from contextlib import contextmanager

from typing import Any
from typing import Optional

_NVTX = None
_NVTX_KIND = None


def _load_nvtx():
    global _NVTX, _NVTX_KIND
    if _NVTX is not None:
        return
    # try python nvtx package
    try:
        import nvtx  # type: ignore

        _NVTX = nvtx
        _NVTX_KIND = "nvtx"
        return
    except Exception:
        pass
    # try CuPy NVTX
    try:
        import cupy  # type: ignore

        _NVTX = cupy.cuda.nvtx
        _NVTX_KIND = "cupy"
        return
    except Exception:
        pass
    _NVTX = False
    _NVTX_KIND = None


def available() -> bool:
    _load_nvtx()
    return bool(_NVTX)


@contextmanager
def range(name: str, payload: Optional[int] = None) -> Any:
    """
    NVTX range for Nsight Systems/Compute.
    No-op if NVTX unavailable
    """
    _load_nvtx()
    if not _NVTX:
        yield
        return

    try:
        if _NVTX_KIND == "nvtx":
            # python-nvtx supports annotate context manager
            with _NVTX.annotate(name, payload=payload):
                yield
        else:
            # cupy.cuda.nvtx
            if payload is None:
                _NVTX.RangePush(name)
                try:
                    yield
                finally:
                    _NVTX.RangePop()
            else:
                _NVTX.RangePush(
                    name
                )  # cupy doesn't reliably support payload on all builds
                try:
                    yield
                finally:
                    _NVTX.RangePop()
    except Exception:
        # never fail user simulation due to profiling
        yield
