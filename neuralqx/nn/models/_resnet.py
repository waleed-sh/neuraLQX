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

"""
Implementation of a ResNet like neural network
"""

import jax.numpy as jnp
import flax.linen as nn


class ResNet(nn.Module):
    """
    A ResNet like neural network composed of several residual blocks with skip connections
    """

    # the number of convolutional blocks
    num_blocks: int = 2

    # number of CNN features should there be in the residual blocks
    block_features: int = 60

    # number of CNN features in the main (first) convolutional block
    main_conv: int = 60

    @staticmethod
    def conv_block(
        x,
        features,
        kernel_size,
    ):
        x = nn.Conv(
            features,
            kernel_size=kernel_size,
            padding="CIRCULAR",
        )(x)

        x = nn.BatchNorm(
            use_running_average=True,
            momentum=0.9,
            epsilon=1e-5,
        )(x)

        return x

    def residual_block(
        self,
        x,
        features,
        kernel_size,
    ):
        shortcut = x

        x = self.conv_block(x, features, kernel_size)

        x = nn.hard_silu(x)

        x = self.conv_block(x, features, kernel_size)

        x += shortcut

        x = nn.hard_silu(x)

        return x

    @nn.compact
    def __call__(self, x):

        # main (first) convolution
        x = nn.Conv(
            self.main_conv,
            kernel_size=(1,),
            strides=(1,),
            padding="CIRCULAR",
        )(x)

        x = nn.BatchNorm(
            use_running_average=True,
            momentum=0.9,
            epsilon=1e-5,
        )(x)

        x = nn.hard_silu(x)

        x = nn.max_pool(x, window_shape=(1,))

        # residual blocks
        for _ in range(self.num_blocks):
            x = self.residual_block(x, self.block_features, (1,))

        # global average pooling
        x = nn.avg_pool(x, window_shape=(1,))

        # fully connected layer
        x = nn.Dense(1)(x)

        return jnp.sum(x, axis=-1)
