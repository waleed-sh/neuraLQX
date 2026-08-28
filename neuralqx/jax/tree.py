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


"""Small pytree algebra helpers used by differentiable estimators."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def tree_conj(tree: Any) -> Any:
    """Return the elementwise complex conjugate of every array leaf."""
    return jax.tree_util.tree_map(jnp.conjugate, tree)


def tree_axpy(a: Any, x: Any, y: Any) -> Any:
    """Return ``a * x + y`` leafwise."""

    def axpy_leaf(x_leaf: Any, y_leaf: Any) -> Any:
        dtype = jnp.result_type(jnp.asarray(x_leaf), jnp.asarray(y_leaf))
        return jnp.asarray(a, dtype=dtype) * x_leaf + y_leaf

    return jax.tree_util.tree_map(axpy_leaf, x, y)


def tree_cast_like(tree: Any, target: Any) -> Any:
    """Cast array leaves in ``tree`` to the dtype of matching ``target`` leaves."""

    def cast(value: Any, reference: Any) -> Any:
        ref = jnp.asarray(reference)
        return jnp.asarray(value, dtype=ref.dtype)

    return jax.tree_util.tree_map(cast, tree, target)


def tree_size(tree: Any) -> int:
    """Return the total number of scalar entries in a pytree."""
    return int(sum(jnp.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(tree)))


def tree_any_complex(tree: Any) -> bool:
    """Return whether any leaf has a complex dtype."""
    return any(jnp.iscomplexobj(leaf) for leaf in jax.tree_util.tree_leaves(tree))


def tree_all_real(tree: Any) -> bool:
    """Return whether all leaves have real dtypes."""
    return not tree_any_complex(tree)


__all__ = [
    "tree_all_real",
    "tree_any_complex",
    "tree_axpy",
    "tree_cast_like",
    "tree_conj",
    "tree_size",
]
