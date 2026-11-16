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


"""Trainable embeddings for discrete state labels."""

from flax import nnx

from ..utils.initialization import as_rngs
from ..utils.initialization import positive_int
from ..utils.partitioning import partitioned


class Embed(nnx.Embed):
    """Look up a trainable feature vector for each integer label.

    Labels must lie in ``[0, num_embeddings)``. The layer appends a feature axis
    to the input shape. ``embedding_axes`` optionally names the label and feature
    axes for partitioning. Native NNX dtype and initializer options are accepted.
    """

    embedding: nnx.Param
    """Table of shape ``(num_embeddings, features)``."""

    num_embeddings: int
    """Number of valid integer labels."""

    features: int
    """Number of learned coordinates per label."""

    dtype: object
    """Output computation dtype, or ``None`` for native promotion."""

    param_dtype: object
    """Storage dtype of the embedding table."""

    def __init__(
        self, num_embeddings, features, *, rngs, embedding_axes=None, **kwargs
    ):
        """Initialize an embedding table from an RNG stream or seed."""
        initializer = kwargs.pop("embedding_init", nnx.initializers.normal(stddev=1.0))
        super().__init__(
            positive_int(num_embeddings, "num_embeddings"),
            positive_int(features, "features"),
            rngs=as_rngs(rngs),
            embedding_init=partitioned(initializer, embedding_axes),
            **kwargs,
        )
