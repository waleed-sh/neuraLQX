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


"""Linen constructors using the same implementation as the default NNX models.

The arguments match the NNX constructors except for ``rngs``. Initialize the
returned Linen module with ``model.init(key, states)``. Use :func:`to_linen`
for custom neuraLQX modules or for individual NNX layers.
"""

from ..utils.bridge import to_linen
from . import autoregressive
from . import cnn
from . import gnn
from . import jastrow
from . import mlp
from . import rbm


def MLP(*args, **kwargs):
    """Construct the Linen form of :class:`neuralqx.nn.models.MLP`."""
    return to_linen(mlp.MLP, *args, **kwargs)


def RBM(*args, **kwargs):
    """Construct the Linen form of :class:`neuralqx.nn.models.RBM`."""
    return to_linen(rbm.RBM, *args, **kwargs)


def Jastrow(*args, **kwargs):
    """Construct the Linen form of :class:`neuralqx.nn.models.Jastrow`."""
    return to_linen(jastrow.Jastrow, *args, **kwargs)


def CNN(*args, **kwargs):
    """Construct the Linen form of :class:`neuralqx.nn.models.CNN`."""
    return to_linen(cnn.CNN, *args, **kwargs)


def GNN(*args, **kwargs):
    """Construct the Linen form of :class:`neuralqx.nn.models.GNN`."""
    return to_linen(gnn.GNN, *args, **kwargs)


def AutoregressiveRNN(*args, **kwargs):
    """Construct the Linen form of :class:`neuralqx.nn.models.AutoregressiveRNN`."""
    return to_linen(autoregressive.AutoregressiveRNN, *args, **kwargs)


__all__ = ["CNN", "GNN", "MLP", "RBM", "AutoregressiveRNN", "Jastrow"]
