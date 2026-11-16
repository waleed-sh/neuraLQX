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


"""Composable NNX layers with explicit stage boundaries."""

from flax import nnx

from ..activations import get_activation


class Sequential(nnx.Sequential):
    """Apply an ordered sequence of modules or pure functions.

    Modules remain separately addressable through ``layers``. This preserves
    useful stage boundaries for future pipeline scheduling. This container does
    not itself schedule stages on devices or change the semantics of its layers.
    """

    layers: nnx.List
    """Ordered modules and callables, using the native NNX container."""

    def __init__(self, *layers):
        """Keep the supplied modules and functions in their evaluation order."""
        super().__init__(*layers)


class Activation(nnx.Module):
    """Turn a registered activation or callable into a composable NNX layer.

    The name is resolved once at construction. Calls invoke the selected
    function directly. Keyword arguments, such as GELU's ``approximate`` flag,
    are fixed when the layer is constructed.
    """

    function: object
    """Resolved activation callable."""

    options: tuple
    """Fixed keyword arguments passed to the activation."""

    def __init__(self, function="tanh", **kwargs):
        """Resolve an activation and bind its optional keyword arguments."""
        self.function = get_activation(function)
        self.options = tuple(kwargs.items())

    def __call__(self, inputs):
        """Apply the configured activation to an array."""
        return self.function(inputs, **dict(self.options))
