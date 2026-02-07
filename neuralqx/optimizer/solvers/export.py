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

from typing import Tuple
from typing import List

from .types import SolverConfig


def _flatten(prefix: str, d: dict) -> Tuple[List[str], List[str]]:
    fields, values = [], []
    for k, v in d.items():
        fields.append(f"{prefix}{k}")
        values.append(str(v))
    return fields, values


def export_info(solver_cfg: SolverConfig) -> tuple[list[str], list[str]]:
    fields: List[str] = []
    values: List[str] = []

    # solver type
    fields.append("Preconditioner Solver")
    values.append(type(solver_cfg).name)

    return fields, values
