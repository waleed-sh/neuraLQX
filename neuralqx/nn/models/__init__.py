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
This module contains the implementation of different neural networks used in the simulations
"""

from ._revnet import RevNet, RevNetAuto
from ._resnet import ResNet
from ._revenetc import RevNetComplex
from ._cfvtnet import CFVTNet, make_cfvtnet_kwargs
from ._graph_nqs import LocalGraphNQS, make_local_graph_nqs_kwargs

__all__ = [
    "RevNet",
    "RevNetAuto",
    "ResNet",
    "RevNetComplex",
    "make_cfvtnet_kwargs",
    "CFVTNet",
    "LocalGraphNQS",
    "make_local_graph_nqs_kwargs",
]
