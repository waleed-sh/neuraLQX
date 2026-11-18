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


"""Neural wavefunctions, reusable layers and registered JAX activations.

NNX is the default backend. Construct an initialized model with, for example,
``nn.MLP(hilbert.size, rngs=0)`` and pass it directly to ``VariationalState``.
Linen users can use ``nn.models.linen.MLP`` or convert a layer with
``nn.to_linen(nn.layers.Linear, in_features, out_features)``.

Models expose exact connected-state reuse as an optional capability. Estimator
and gradient code can use it without relying on a model's parameter names.
"""

from . import activations
from . import layers
from . import models
from . import projectors
from . import utils
from .activations import activation
from .activations import get_activation
from .activations import reim
from .models import CNN
from .models import GNN
from .models import MLP
from .models import RBM
from .models import AutoregressiveRNN
from .models import Jastrow
from .models import LogWavefunction
from .utils import ConnectionPolicy
from .utils.bridge import to_linen

__all__ = [
    "CNN",
    "GNN",
    "MLP",
    "RBM",
    "AutoregressiveRNN",
    "ConnectionPolicy",
    "Jastrow",
    "LogWavefunction",
    "activation",
    "activations",
    "get_activation",
    "layers",
    "models",
    "projectors",
    "reim",
    "to_linen",
    "utils",
]
