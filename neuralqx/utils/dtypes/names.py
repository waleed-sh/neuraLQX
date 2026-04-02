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

"""Canonical dtype-name definitions for neuraLQX."""

from __future__ import annotations

REAL_DTYPE_NAMES: tuple[str, ...] = ("float16", "bfloat16", "float32", "float64")
COMPLEX_DTYPE_NAMES: tuple[str, ...] = ("complex64", "complex128")
INDEX_DTYPE_NAMES: tuple[str, ...] = ("int32", "int64")

__all__ = [
    "COMPLEX_DTYPE_NAMES",
    "INDEX_DTYPE_NAMES",
    "REAL_DTYPE_NAMES",
]
