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


"""Partition annotations without committing a model to one device mesh.

Annotations describe tensor axes. They do not choose a tensor-parallel or
pipeline-parallel execution schedule, and do not introduce collectives on
unpartitioned models.
"""

from flax import nnx


def partitioned(initializer, axes=None):
    """Attach optional NNX partition names to a parameter initializer.

    ``axes`` has one entry per tensor dimension. Entries are mesh-axis names,
    tuples of mesh-axis names, or ``None`` for an unpartitioned dimension.
    Leave it as ``None`` to use the initializer without metadata. With named
    axes, initialize under the intended JAX mesh and use Flax's partition
    utilities to place the resulting state.
    """
    return (
        initializer if axes is None else nnx.with_partitioning(initializer, tuple(axes))
    )
