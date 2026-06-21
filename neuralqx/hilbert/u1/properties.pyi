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

import numpy as np

from neuralqx.hilbert.local import AbstractLocalSpace

class U1PropertiesMixin:
    @property
    def size(self) -> int: ...
    @property
    def tiny_size(self) -> int: ...
    @property
    def q_min(self) -> int | float: ...
    @property
    def q_max(self) -> int | float: ...
    @property
    def q_step(self) -> int | float: ...
    @property
    def local_size(self) -> int: ...
    @property
    def local_sizes(self) -> tuple[int, ...]: ...
    @property
    def local_spaces(self) -> tuple[AbstractLocalSpace, ...]: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def dimension(self) -> int: ...
    @property
    def dimensions_pretty(self) -> str: ...
    @property
    def is_indexable(self) -> bool: ...
