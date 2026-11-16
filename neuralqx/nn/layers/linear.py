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


"""Affine maps with explicit parameter and computation dtypes."""

import jax.numpy as jnp
from flax import nnx

from ..utils.initialization import as_rngs
from ..utils.initialization import positive_int
from ..utils.partitioning import partitioned


class Linear(nnx.Linear):
    r"""Apply :math:`y = xW + b` to the final feature axis.

    Leading batch dimensions are preserved. This is Flax's native NNX affine
    kernel, with optional mesh-axis annotations on its parameters. Initializers,
    dot precision, compute dtype and parameter dtype remain explicit so a model
    can use the same layer on a CPU or on an accelerator.

    Parameters
    ----------
    in_features, out_features : int
        Input and output feature dimensions.
    rngs : flax.nnx.Rngs, int, or PRNG key
        Parameter initialization stream or seed.
    kernel_axes, bias_axes : tuple, optional
        Partition names for the kernel and bias dimensions.
    **kwargs
        Native ``flax.nnx.Linear`` options, including ``use_bias``, ``dtype``,
        ``param_dtype``, ``precision``, ``kernel_init`` and ``bias_init``.
    """

    kernel: nnx.Param
    """Weight matrix of shape ``(in_features, out_features)``."""

    bias: nnx.Param
    """Output bias, absent when ``use_bias=False``."""

    in_features: int
    """Number of input coordinates in each feature vector."""

    out_features: int
    """Number of output coordinates in each feature vector."""

    dtype: object
    """Computation dtype, or ``None`` for native Flax dtype promotion."""

    param_dtype: object
    """Storage dtype of the affine parameters."""

    precision: object
    """Precision requested for dense and sparse-update contractions."""

    def __init__(
        self,
        in_features,
        out_features,
        *,
        rngs,
        kernel_axes=None,
        bias_axes=None,
        **kwargs,
    ):
        """Allocate the affine parameters using the supplied RNG stream."""
        kernel_init = kwargs.pop("kernel_init", nnx.initializers.lecun_normal())
        bias_init = kwargs.pop("bias_init", nnx.initializers.zeros_init())
        super().__init__(
            positive_int(in_features, "in_features"),
            positive_int(out_features, "out_features"),
            rngs=as_rngs(rngs),
            kernel_init=partitioned(kernel_init, kernel_axes),
            bias_init=partitioned(bias_init, bias_axes),
            **kwargs,
        )

    def sparse_delta(self, sites, differences):
        """Apply only the changed input coordinates to the linear part.

        ``sites`` and ``differences`` have shape ``(..., changes)``. The result
        is the exact affine-output difference and therefore excludes the bias.
        This operation remains differentiable with respect to the kernel.
        """
        dtype = self.dtype or jnp.result_type(differences, self.kernel[...])
        weights = jnp.asarray(self.kernel[...], dtype)[sites]
        return jnp.einsum(
            "...k,...kh->...h",
            jnp.asarray(differences, dtype),
            weights,
            precision=self.precision,
        )


Dense = Linear
"""Alias for :class:`Linear`."""
