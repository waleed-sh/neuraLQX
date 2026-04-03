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
neuraLQX startup bootstrap.

This module is executed automatically at Python interpreter startup via
``neuralqx_boot.pth`` in site-packages, which means the environment
variables below are always set before JAX, jaxlib, or NetKet are imported,
regardless of the order the user writes their imports.

Do **not** add anything here that requires a third-party import.
"""

import os

#
#
#   Performance / memory defaults

# setdefault: users and cluster admins can override via the shell environment
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

#
#
#   Silence third-party output

os.environ["NETKET_NO_TIPS"] = "False"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#
#
#   x64 precision

# JAX and NetKet both read their x64 flags at import time. We honour
# NQX_ENABLE_X64 (neuraLQX-native) first, then fall back to JAX_ENABLE_X64,
# then default to True to match NetKet's own default.


def _parse_bool(val: str) -> bool:
    return val.strip().lower() not in ("0", "false", "no", "off")


_x64 = _parse_bool(
    os.environ.get("NQX_ENABLE_X64", os.environ.get("JAX_ENABLE_X64", "1"))
)
os.environ["JAX_ENABLE_X64"] = "1" if _x64 else "0"
os.environ["NETKET_ENABLE_X64"] = "1" if _x64 else "0"
