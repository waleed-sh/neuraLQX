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


"""Small U(1) utility functions."""

from .arithmetic import modular_add
from .arithmetic import signed_modular_sum
from .arithmetic import wrap_values
from .dimensions import scientific_int
from .freeze import freeze_gauge_fixing
from .local_range import u1_local_range
from .local_range import validate_gauge_dimensions

__all__ = [
    "freeze_gauge_fixing",
    "modular_add",
    "scientific_int",
    "signed_modular_sum",
    "u1_local_range",
    "validate_gauge_dimensions",
    "wrap_values",
]
