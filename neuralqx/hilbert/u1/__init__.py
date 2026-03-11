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
U(1) Hilbert-space subpackage.
"""

from . import index  # noqa: F401  — registers states_to_numbers/numbers_to_states
from .operations import random as _operations_random  # noqa: F401
from .operations import flip as _operations_flip  # noqa: F401

from .interface import HilbertU1

__all__ = [
    "HilbertU1",
]
