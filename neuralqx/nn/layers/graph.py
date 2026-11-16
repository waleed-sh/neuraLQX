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


"""Message passing over a fixed graph without dense adjacency matrices."""

from flax import nnx

from ..activations import get_activation
from ..utils.initialization import as_rngs
from ..utils.topology import mean_neighbors
from .linear import Linear


class GraphConv(nnx.Module):
    r"""Combine a node's features with its mean incoming-neighbor features.

    For directed edges :math:`u\to v`, the layer evaluates
    :math:`h'_v=\sigma(W_s h_v + W_n\,\mathrm{mean}_{u\to v} h_u+b)`.
    An isolated node receives a zero neighbor vector. Edges are provided as
    integer sender and receiver arrays, so the forward pass costs linear work
    in the edge count and does not allocate a dense adjacency matrix.
    """

    self_map: Linear
    """Affine map of each node's own features."""

    neighbor_map: Linear
    """Bias-free map of the mean incoming-neighbor features."""

    activation: object
    """Resolved activation applied after adding the two maps."""

    def __init__(self, in_features, out_features, *, activation="silu", rngs, **kwargs):
        """Construct independent self and neighbor maps using one RNG stream."""
        rngs = as_rngs(rngs)
        self.self_map = Linear(in_features, out_features, rngs=rngs, **kwargs)
        self.neighbor_map = Linear(
            in_features, out_features, use_bias=False, rngs=rngs, **kwargs
        )
        self.activation = get_activation(activation, dtype=self.self_map.dtype)

    def __call__(self, inputs, senders, receivers, inverse_degree):
        """Update arrays of shape ``(..., nodes, features)`` on the supplied graph."""
        messages = mean_neighbors(inputs, senders, receivers, inverse_degree)
        return self.from_aggregated(inputs, messages)

    def from_aggregated(self, inputs, neighbors):
        """Update features from already aggregated incoming-neighbor means.

        Connected-state evaluators can update these means from local changes
        instead of gathering every edge again. The affine maps and activation
        are identical to those in ordinary message passing.
        """
        return self.activation(self.self_map(inputs) + self.neighbor_map(neighbors))
