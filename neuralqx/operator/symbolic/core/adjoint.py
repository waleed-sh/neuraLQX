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


"""Exact reverse connectivity from bounded local update supports."""

from itertools import product
from math import prod

import jax.numpy as jnp
import numpy as np

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.utils.errors import NeuralqxValueError


class ExplicitAdjointOperator(ComputationalOperator):
    """Keep both supplied operators as pytree children for native transforms."""

    forward: ComputationalOperator
    reverse: ComputationalOperator

    name = property(lambda self: self.forward.name)
    dtype = property(lambda self: self.forward.dtype)
    is_hermitian = property(lambda self: self.forward.is_hermitian)
    is_diagonal = property(lambda self: self.forward.is_diagonal)
    max_conn_size = property(lambda self: self.forward.max_conn_size)
    adjoint = property(lambda self: self.reverse)

    def _get_conn_padded_kernel(self, state):
        return self.forward._get_conn_padded_kernel(state)


class AdjointOperator(ComputationalOperator):
    """Reverse a compiled row kernel over its possible source configurations.

    Candidate sources differ only on sites the forward rules can update.
    Local candidates are checked against the complete Hilbert space constraint.
    No global basis ranking or enumeration is required.
    """

    operator: ComputationalOperator
    values: jax.Array
    changed: jax.Array

    dtype = property(lambda self: self.operator.dtype)
    is_hermitian = property(lambda self: self.operator.is_hermitian)
    is_diagonal = property(lambda self: self.operator.is_diagonal)
    max_conn_size = property(lambda self: int(self.values.shape[0]))
    adjoint = property(lambda self: self.operator)

    def _get_conn_padded_kernel(self, state):
        candidates = jnp.where(self.changed, self.values.astype(state.dtype), state)
        valid = self.hilbert.is_valid(candidates)
        safe = jnp.where(valid[:, None], candidates, state)
        targets, amplitudes = self.operator._get_conn_padded_batch_kernel(safe)
        matches = jnp.all(targets == state, axis=-1)
        weights = jnp.conj(jnp.sum(jnp.where(matches, amplitudes, 0), axis=-1))
        # Overlapping update supports can generate the same source. Each
        # candidate's weight already contains all forward contributions.
        order = jnp.lexsort(
            tuple(candidates[:, i] for i in range(self.hilbert.size - 1, -1, -1))
        )
        ordered = candidates[order]
        first = jnp.concatenate(
            (jnp.ones(1, bool), jnp.any(ordered[1:] != ordered[:-1], axis=-1))
        )
        unique = jnp.zeros_like(valid).at[order].set(first)
        weights = jnp.where(valid & unique, weights, 0)
        return jnp.where((weights != 0)[:, None], safe, state), weights


def adjoint_operator(operator, *, max_candidates=65536):
    """Construct an exact local inverse with an explicit allocation bound."""
    if operator._adjoint_fn is not None:
        return operator._adjoint_fn(operator)
    supports = operator._support
    if supports is None:
        supports = (tuple(range(operator.hilbert.size)),)
    spaces = operator.hilbert.local_spaces
    count = sum(prod(spaces[i].local_size for i in support) for support in supports)
    if count > max_candidates:
        raise NeuralqxValueError(
            f"The generic symbolic adjoint requires {count} local candidates, exceeding "
            f"{max_candidates}. Supply an adjoint_factory when constructing the compiled operator."
        )
    values = np.zeros((count, operator.hilbert.size), dtype=operator.hilbert.dtype)
    changed = np.zeros_like(values, dtype=bool)
    offset = 0
    for support in supports:
        local = [np.asarray(spaces[i].all_values()) for i in support]
        for row in product(*local):
            values[offset, list(support)] = row
            changed[offset, list(support)] = True
            offset += 1
    return AdjointOperator(
        operator.hilbert, operator, jnp.asarray(values), jnp.asarray(changed)
    )
