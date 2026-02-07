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
This module implements several physical models in neuraLQX.
"""

from . import wcl
from . import qr
from . import spherical

from .qr import LqxSingleVertexQR
from .wcl import LqxWCL
from .spherical import LqxBojowaldSwiderski

from .abstract_lqx_interface import AbstractLqxInterface
from .abstract_lqx_model import AbstractLqxModel

__all__ = [
    "wcl",
    "qr",
    "spherical",
    # direct hooks
    "LqxSingleVertexQR",
    "LqxWCL",
    "LqxBojowaldSwiderski",
    # abstract classes
    "AbstractLqxInterface",
    "AbstractLqxModel",
]
