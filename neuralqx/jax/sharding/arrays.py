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


"""Small predicates for modern JAX array sharding."""

from __future__ import annotations

from typing import Any

import jax


def is_jax_array(value: Any) -> bool:
    """Return whether ``value`` is a JAX array object."""
    return isinstance(value, jax.Array)


def is_sharded_array(value: Any) -> bool:
    """Return whether ``value`` is partitioned or replicated over many devices."""
    if not isinstance(value, jax.Array):
        return False
    try:
        return len(value.sharding.device_set) > 1
    except AttributeError:
        return False


def is_distributed_array(value: Any) -> bool:
    """Return whether a JAX array spans non-local processes."""
    if not isinstance(value, jax.Array):
        return False
    return bool(is_sharded_array(value) and not value.is_fully_addressable)


__all__ = ["is_distributed_array", "is_jax_array", "is_sharded_array"]
