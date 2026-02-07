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

from dataclasses import asdict
from dataclasses import is_dataclass

from typing import Tuple
from typing import List
from typing import Any
from typing import Union

from .types import LearningRateSpec


def _flatten(prefix: str, d: dict) -> Tuple[List[str], List[str]]:
    fields, values = [], []
    for k, v in d.items():
        fields.append(f"{prefix}{k}")
        values.append(str(v))
    return fields, values


def export_info(
    optimizer_cfg: Any, lr_spec: Union[float, LearningRateSpec]
) -> tuple[list[str], list[str]]:
    fields: List[str] = []
    values: List[str] = []

    # optimiser type
    fields.append("Optimizer")
    values.append(type(optimizer_cfg).__name__)

    # optimiser params
    if is_dataclass(optimizer_cfg):
        f, v = _flatten("Optimizer param: ", asdict(optimizer_cfg))
        fields += f
        values += v

    # learning rate / schedule
    if isinstance(lr_spec, (float, int)):
        fields.append("Learning rate")
        values.append(str(lr_spec))
    else:
        fields.append("Scheduler type")
        values.append(type(lr_spec).__name__)
        if is_dataclass(lr_spec):
            f, v = _flatten("Scheduler param: ", asdict(lr_spec))
            fields += f
            values += v

    return fields, values
