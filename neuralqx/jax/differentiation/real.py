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


"""Real-coordinate views of complex pytrees."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

import jax


@dataclass(frozen=True)
class _LeafSpec:
    is_complex: bool


def split_tree_to_real(tree: Any) -> tuple[Any, Callable[[Any], Any]]:
    """Split complex leaves into real-coordinate leaves.

    Real leaves are left untouched. Complex leaves ``z`` become two real leaves
    ``Re(z), Im(z)``. The returned callable reassembles a compatible
    real-coordinate pytree back to the original complex/real pytree structure.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    real_leaves: list[Any] = []
    specs: list[_LeafSpec] = []
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        is_complex = bool(jnp.iscomplexobj(arr))
        specs.append(_LeafSpec(is_complex=is_complex))
        if is_complex:
            real_leaves.append(jnp.real(arr))
            real_leaves.append(jnp.imag(arr))
        else:
            real_leaves.append(leaf)

    real_tree = tuple(real_leaves)

    def reassemble(real_value: Any) -> Any:
        values = list(jax.tree_util.tree_leaves(real_value))
        rebuilt: list[Any] = []
        index = 0
        for spec in specs:
            if spec.is_complex:
                rebuilt.append(values[index] + 1j * values[index + 1])
                index += 2
            else:
                rebuilt.append(values[index])
                index += 1
        if index != len(values):
            raise ValueError("Real-coordinate pytree has incompatible structure.")
        return jax.tree_util.tree_unflatten(treedef, rebuilt)

    return real_tree, reassemble


def real_coordinate_size(tree: Any) -> int:
    """Return the number of scalar real coordinates in ``tree``."""
    real_tree, _ = split_tree_to_real(tree)
    return int(
        sum(jnp.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(real_tree))
    )


__all__ = ["real_coordinate_size", "split_tree_to_real"]
