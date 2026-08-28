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


"""Scheduling utilities shared by distributed planners."""

from __future__ import annotations

import jax.numpy as jnp

import jax


def schedule_balanced_slots(
    costs: tuple[int, ...],
    partitions: int,
) -> jax.Array:
    """Return contiguous, cost-balanced slots for a sharded leading axis.

    ``NamedSharding`` gives each device a contiguous local slice of a sharded
    one-dimensional array. The scheduler therefore explicitly constructs those
    slices: high-cost entries are assigned first to the currently lightest slice,
    subject to each slice having the same static slot count. Padding slots use
    the sentinel index ``len(costs)`` and are expected to map to zero work.
    """
    if partitions <= 0:
        raise ValueError("Partition count must be positive.")
    if not costs:
        raise ValueError("Scheduling requires at least one cost.")

    n_items = len(costs)
    slots_per_partition = (n_items + int(partitions) - 1) // int(partitions)
    loads = [0] * int(partitions)
    bins: list[list[int]] = [[] for _ in range(int(partitions))]
    order = sorted(
        range(n_items),
        key=lambda index: (
            -int(costs[index]),
            index,
        ),
    )

    for index in order:
        available = [
            partition
            for partition, entries in enumerate(bins)
            if len(entries) < slots_per_partition
        ]
        target = min(
            available,
            key=lambda partition: (
                loads[partition],
                len(bins[partition]),
                partition,
            ),
        )
        bins[target].append(index)
        loads[target] += int(costs[index])

    pad_index = n_items
    scheduled = []
    for entries in bins:
        scheduled.extend(entries)
        scheduled.extend([pad_index] * (slots_per_partition - len(entries)))
    return jnp.asarray(scheduled, dtype=jnp.int32)


__all__ = ["schedule_balanced_slots"]
