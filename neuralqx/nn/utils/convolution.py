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


"""Short convolution kernels selected at JAX lowering time."""

import jax.numpy as jnp

import jax
from neuralqx.utils.errors import NeuralqxValueError


def conv1d(inputs, kernel, bias, *, padding, precision=None):
    """Apply a unit-stride, ungrouped channels-last one-dimensional convolution.

    Inputs and parameters must already share a computation dtype. ``padding``
    is ``CIRCULAR`` or ``VALID``. Large batches with a three-site kernel use
    shifted matrix contractions on CPU. Other shapes and accelerator platforms
    use the native convolution primitive. Selection happens at lowering, so
    compiling the same function for a GPU does not retain a CPU-specific path.

    Both implementations evaluate the same cross-correlation and use ordinary
    JAX differentiation. Only floating-point accumulation order can differ.
    """
    if padding not in {"CIRCULAR", "VALID"}:
        raise NeuralqxValueError("conv1d supports CIRCULAR and VALID padding.")
    leading = inputs.shape[:-2]
    x = inputs.reshape((-1, *inputs.shape[-2:]))
    width = kernel.shape[0]

    def native(x, kernel):
        if padding == "CIRCULAR":
            x = jnp.pad(
                x, ((0, 0), ((width - 1) // 2, width // 2), (0, 0)), mode="wrap"
            )
        return jax.lax.conv_general_dilated(
            x,
            kernel,
            (1,),
            "VALID",
            dimension_numbers=("NWC", "WIO", "NWC"),
            precision=precision,
        )

    def matrix_contractions(x, kernel):
        def shifted(index):
            if padding == "CIRCULAR":
                return jnp.roll(x, (width - 1) // 2 - index, axis=-2)
            length = x.shape[-2] - width + 1
            return x[..., index : index + length, :]

        result = jnp.matmul(shifted(0), kernel[0], precision=precision)
        for index in range(1, width):
            result = result + jnp.matmul(
                shifted(index), kernel[index], precision=precision
            )
        return result

    if (
        width == 3
        and inputs.size >= 1024
        and (padding == "CIRCULAR" or x.shape[-2] >= width)
    ):
        value = jax.lax.platform_dependent(
            x, kernel, cpu=matrix_contractions, default=native
        )
    else:
        value = native(x, kernel)
    if bias is not None:
        value = value + bias
    return value.reshape((*leading, *value.shape[-2:]))
