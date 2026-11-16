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


"""Scalar log-wavefunction models, with NNX as the default backend."""

from . import linen
from .autoregressive import AutoregressiveRNN
from .base import LogWavefunction
from .cnn import CNN
from .gnn import GNN
from .jastrow import Jastrow
from .mlp import MLP
from .rbm import RBM

__all__ = [
    "CNN",
    "GNN",
    "MLP",
    "RBM",
    "AutoregressiveRNN",
    "Jastrow",
    "LogWavefunction",
    "linen",
]
