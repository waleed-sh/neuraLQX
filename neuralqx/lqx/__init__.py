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


"""Model facades that construct geometric operators and quantum constraints.

Each facade binds a Hilbert space and graph to a model's operator factories.
The model classes document their charge conventions, dimensional restrictions,
and the linear or quadratic constraints returned by their default factories.
"""

from .base import LQXModel
from .qr import LqxQR
from .spherical import LqxSphericalBojowaldSwiderski
from .spherical import LqxSphericalBoSw

# TODO: finish those then release
# from .su2 import LqxEuclideanSU2
# from .su2 import LqxSU2
from .wcl import LqxWCL
from .wcl import LqxWCL1D
from .wcl import LqxWCL3D
from .wcl import LqxWCL4D

__all__ = [
    "LQXModel",
    # "LqxEuclideanSU2",
    "LqxQR",
    # "LqxSU2",
    "LqxSphericalBoSw",
    "LqxSphericalBojowaldSwiderski",
    "LqxWCL",
    "LqxWCL1D",
    "LqxWCL3D",
    "LqxWCL4D",
]
