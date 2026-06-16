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


"""Callable-backed discrete Hilbert constraints.

This module provides a small adapter that turns a user-supplied predicate into
the standard discrete constraint interface. The callable should be compatible
with JAX array inputs and return a boolean mask over the batch dimensions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from neuralqx.utils.struct import field

from .abstract import AbstractDiscreteConstraint


class CallableDiscreteConstraint(AbstractDiscreteConstraint):
    """A discrete constraint implemented by a user-provided predicate.

    Args:
        fn: Callable receiving a state array and returning truth values with
            shape ``states.shape[:-1]``.
        name: Optional descriptive name used for hashing and diagnostics.
        hash_key: Optional stable key used instead of the callable identity.

    Attributes:
        fn: Callable receiving a state array and returning truth values with
            shape ``states.shape[:-1]``.
        name: Optional descriptive name used for hashing and diagnostics.
        hash_key: Optional stable key used instead of the callable identity.
    """

    fn: Callable[[Any], Any] = field(pytree=False, compare=False)
    """Callable receiving a state array and returning truth values with shape ``states.shape[:-1]``."""
    name: str | None = field(static=True, default=None)
    """Optional descriptive name used for hashing and diagnostics."""
    hash_key: Any | None = field(static=True, default=None)
    """Optional stable key used instead of the callable identity."""

    def __call__(self, states: Any) -> jnp.ndarray:
        """Evaluates the wrapped predicate as a boolean JAX array.

        Args:
            states: State array passed directly to :attr:`fn`.

        Returns:
            Boolean array produced by the callable and coerced to ``jnp.bool_``.
        """
        return jnp.asarray(self.fn(states), dtype=jnp.bool_)

    def __hash__(self) -> int:
        """Returns a hash for static constraint dispatch and caching."""
        key = self.hash_key if self.hash_key is not None else id(self.fn)
        return hash((type(self), self.name, key))


__all__ = ["CallableDiscreteConstraint"]
