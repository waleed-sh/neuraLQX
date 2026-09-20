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


"""One-shot chain-aware Monte Carlo statistics."""

from __future__ import annotations

from typing import Any

import jax

from .core import Stats
from .online import OnlineStats
from .rhat import split_r_hat
from .shape import as_chain_matrix


def statistics(values: Any, *, chain_axis: int | None = 0) -> Stats:
    """Compute scalar Monte Carlo statistics for ``values``.

    Rank-2 and higher inputs are interpreted as chain-major by default: axis 0
    enumerates independent chains, and all remaining axes are samples inside
    each chain. This matches neuraLQX sampler output and remains compatible with
    sample-axis sharding because reductions happen on the JAX array itself.
    """
    matrix = as_chain_matrix(values, chain_axis=chain_axis)
    return _statistics_from_matrix(matrix)


@jax.jit
def _statistics_from_matrix(matrix: jax.Array) -> Stats:
    online = OnlineStats.empty(matrix.shape[0], matrix.dtype).update(
        matrix, chain_axis=0
    )
    stats = online.to_stats()
    return stats.replace(r_hat=split_r_hat(matrix, stats.variance))


__all__ = ["statistics"]
