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


"""NNX layers for custom wavefunctions. NNX is the default backend."""

from . import linen
from .composition import Activation
from .composition import Sequential
from .convolution import Conv
from .embedding import Embed
from .graph import GraphConv
from .linear import Dense
from .linear import Linear
from .normalization import LayerNorm
from .normalization import RMSNorm

__all__ = [
    "Activation",
    "Conv",
    "Dense",
    "Embed",
    "GraphConv",
    "LayerNorm",
    "Linear",
    "RMSNorm",
    "Sequential",
    "linen",
]
