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

from typing import List, Tuple

from neuralqx.utils import distributed as _dist


def export_info(
    sampler_type: str,
    number_of_samples: int,
    *,
    number_of_chains: int,
    number_of_sweeps: int,
    machine_pow: int,
    reset_chains: bool,
) -> Tuple[List[str], List[str]]:

    fields: List[str] = []
    values: List[str] = []

    fields.append("Sampler type")
    values.append(str(sampler_type))

    fields.append("Number of samples")
    values.append(str(number_of_samples))

    if sampler_type == "Exact Sampler":
        fields.extend(
            ["Number of chains", "Number of sweeps", "Machine power", "Reset chains"]
        )
        values.extend([None, None, None, None])
        return fields, values

    fields.append(
        "Number of chains per process" if _dist.available else "Number of chains"
    )
    values.append(str(number_of_chains))

    fields.append("Number of sweeps")
    values.append(str(number_of_sweeps))

    fields.append("Machine power")
    values.append(str(machine_pow))

    fields.append("Reset chains")
    values.append(str(reset_chains))

    return fields, values
