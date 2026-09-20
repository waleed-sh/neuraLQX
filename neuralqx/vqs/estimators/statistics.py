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


"""VQS estimator statistics wrappers."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.utils.stats import OnlineStats
from neuralqx.utils.stats import Stats
from neuralqx.utils.stats import online_statistics
from neuralqx.utils.stats import statistics
from neuralqx.utils.struct import Struct


class LocalEstimators(Struct):
    """Per-sample local estimator values."""

    data: Any

    def to_stats(self) -> Stats:
        """Summarize local estimator samples."""
        return statistics(self.data)


def tree_l2_norm(tree: Any) -> jax.Array:
    """Return the global L2 norm of all numeric leaves in a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    total = sum(jnp.sum(jnp.abs(jnp.asarray(leaf)) ** 2) for leaf in leaves)
    return jnp.sqrt(total)


__all__ = [
    "LocalEstimators",
    "OnlineStats",
    "Stats",
    "online_statistics",
    "statistics",
    "tree_l2_norm",
]
