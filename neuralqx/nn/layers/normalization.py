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


"""Per-configuration normalization without running batch statistics."""

from flax import nnx

from ..utils.initialization import as_rngs


class LayerNorm(nnx.LayerNorm):
    """Normalize the selected feature axes and apply trainable scale and bias.

    This layer uses no running statistics, which keeps a wavefunction independent
    of the other configurations in an estimator batch. All native NNX options,
    including reduction axes, precision-related dtype options and parameter
    metadata, are accepted. Normalizing spatial axes destroys spatial locality.
    """

    scale: nnx.Param
    """Multiplicative feature scale when ``use_scale=True``."""

    bias: nnx.Param
    """Additive feature bias when ``use_bias=True``."""

    def __init__(self, num_features, *, rngs, **kwargs):
        """Create native NNX layer normalization with an RNG stream or seed."""
        super().__init__(num_features, rngs=as_rngs(rngs), **kwargs)


class RMSNorm(nnx.RMSNorm):
    """Normalize features by their root mean square without subtracting the mean.

    Unlike batch normalization, the output for one configuration does not
    depend on the batch composition. Native NNX epsilon, axes, dtype and
    parameter metadata options are forwarded unchanged.
    """

    scale: nnx.Param
    """Multiplicative feature scale when ``use_scale=True``."""

    def __init__(self, num_features, *, rngs, **kwargs):
        """Create native NNX RMS normalization with an RNG stream or seed."""
        super().__init__(num_features, rngs=as_rngs(rngs), **kwargs)
