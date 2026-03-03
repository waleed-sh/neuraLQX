#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def params_are_complex(params) -> bool:
    return any(jnp.iscomplexobj(x) for x in jax.tree_util.tree_leaves(params))


def make_grad_qgt_compatible(params, grad):
    """Drop spurious imaginary Monte Carlo noise for real-parameter pytrees."""
    if params_are_complex(params):
        return grad
    return jax.tree_util.tree_map(lambda x: jnp.real(x), grad)


def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_add_scaled(a, b, scale):
    return jax.tree_util.tree_map(lambda x, y: x + scale * y, a, b)


def tree_scale(a, scale):
    return jax.tree_util.tree_map(lambda x: scale * x, a)


def tree_zeros_like(a):
    return jax.tree_util.tree_map(jnp.zeros_like, a)


def same_treedef(a, b) -> bool:
    return jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)


def get_stats_mean(stats_obj: Any) -> float:
    if hasattr(stats_obj, "mean"):
        return float(jnp.asarray(jnp.real(stats_obj.mean)))
    if hasattr(stats_obj, "Mean"):
        return float(jnp.asarray(jnp.real(stats_obj.Mean)))
    return float(jnp.asarray(jnp.real(stats_obj)))
