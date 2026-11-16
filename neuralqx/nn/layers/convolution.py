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


"""Channels-last convolutions with kernel selection at JAX lowering time."""

from flax import nnx

import jax

from ..utils.convolution import conv1d
from ..utils.initialization import as_rngs
from ..utils.initialization import positive_int
from ..utils.partitioning import partitioned


class Conv(nnx.Conv):
    """Channels-last convolution in one or more spatial dimensions.

    Inputs have shape ``(..., spatial_axes, channels)``. Use ``padding="CIRCULAR"``
    for periodic lattices. Strides, dilation, groups, masks, precision and dtype
    are forwarded to the native NNX convolution. Optional ``kernel_axes`` and
    ``bias_axes`` annotate parameter dimensions without choosing a device mesh.
    """

    kernel: nnx.Param
    """Convolution weights with spatial, input-channel and output-channel axes."""

    bias: nnx.Param
    """Output-channel bias, optionally disabled with ``use_bias=False``."""

    in_features: int
    """Number of channels in the input."""

    out_features: int
    """Number of output channels."""

    kernel_size: tuple
    """Width of the kernel along each spatial axis."""

    dtype: object
    """Computation dtype, or ``None`` for native Flax dtype promotion."""

    param_dtype: object
    """Storage dtype of the convolution parameters."""

    precision: object
    """Precision requested for convolution and matrix contractions."""

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        *,
        rngs,
        kernel_axes=None,
        bias_axes=None,
        **kwargs,
    ):
        """Create a convolution with native NNX options and partition metadata."""
        kernel_init = kwargs.pop("kernel_init", nnx.initializers.lecun_normal())
        bias_init = kwargs.pop("bias_init", nnx.initializers.zeros_init())
        super().__init__(
            positive_int(in_features, "in_features"),
            positive_int(out_features, "out_features"),
            kernel_size,
            rngs=as_rngs(rngs),
            kernel_init=partitioned(kernel_init, kernel_axes),
            bias_init=partitioned(bias_init, bias_axes),
            **kwargs,
        )

    def __call__(self, inputs, out_sharding=None):
        """Convolve features, selecting a short-kernel CPU path when applicable.

        GPU execution keeps the native convolution. Masks, groups, dilation,
        explicit output sharding and custom convolution functions also retain
        the native NNX implementation.
        """
        unit_stride = all(
            value in (None, 1, (1,), [1])
            for value in (self.strides, self.input_dilation, self.kernel_dilation)
        )
        specialized = (
            self.kernel_size == (3,)
            and inputs.ndim >= 2
            and inputs.size >= 1024
            and isinstance(self.padding, str)
            and self.padding.upper() in {"CIRCULAR", "VALID"}
            and unit_stride
            and self.feature_group_count == 1
            and self.mask is None
            and self.preferred_element_type is None
            and out_sharding is None
            and self.conv_general_dilated is jax.lax.conv_general_dilated
        )
        if not specialized:
            return super().__call__(inputs, out_sharding=out_sharding)
        inputs, kernel, bias = self.promote_dtype(
            (
                inputs,
                self.kernel[...],
                self.bias[...] if self.bias is not None else None,
            ),
            dtype=self.dtype,
        )
        return conv1d(
            inputs, kernel, bias, padding=self.padding.upper(), precision=self.precision
        )
