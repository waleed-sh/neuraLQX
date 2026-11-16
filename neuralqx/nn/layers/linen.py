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


"""Linen constructors for neuraLQX layers, sharing their native NNX kernels.

Constructor arguments match the NNX layers except for ``rngs``. Supply keys
through the returned module's ``init`` method. Layers can be nested inside
ordinary Linen modules, including ones using neuraLQX activation functions.
"""

from ..utils.bridge import to_linen
from . import composition
from . import convolution
from . import embedding
from . import graph
from . import linear
from . import normalization


def Linear(*args, **kwargs):
    """Construct the Linen form of :class:`neuralqx.nn.layers.Linear`."""
    return to_linen(linear.Linear, *args, **kwargs)


Dense = Linear
"""Alias for the Linen :func:`Linear` constructor."""


def Conv(*args, **kwargs):
    """Construct the Linen form of :class:`neuralqx.nn.layers.Conv`."""
    return to_linen(convolution.Conv, *args, **kwargs)


def Embed(*args, **kwargs):
    """Construct the Linen form of :class:`neuralqx.nn.layers.Embed`."""
    return to_linen(embedding.Embed, *args, **kwargs)


def LayerNorm(*args, **kwargs):
    """Construct the Linen form of :class:`neuralqx.nn.layers.LayerNorm`."""
    return to_linen(normalization.LayerNorm, *args, **kwargs)


def RMSNorm(*args, **kwargs):
    """Construct the Linen form of :class:`neuralqx.nn.layers.RMSNorm`."""
    return to_linen(normalization.RMSNorm, *args, **kwargs)


def Activation(*args, **kwargs):
    """Construct a Linen layer for a registered activation or custom callable."""
    return to_linen(composition.Activation, *args, **kwargs)


def GraphConv(*args, **kwargs):
    """Construct the Linen form of :class:`neuralqx.nn.layers.GraphConv`."""
    return to_linen(graph.GraphConv, *args, **kwargs)


__all__ = [
    "Activation",
    "Conv",
    "Dense",
    "Embed",
    "GraphConv",
    "LayerNorm",
    "Linear",
    "RMSNorm",
]
