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


"""
Multi-Operator Metropolis-Hastings transition rules compatible with NetKet's
`MetropolisSampler` backend.

Features
------------
    - JAX-fast path (`MultiHamiltonianRuleJax`) for GPU/TPU execution. Works with
      `netket.operator.DiscreteJaxOperator` and `ComputationalJaxOperator` (which
      subclasses NetKet's DiscreteJaxOperator)
    - Numba/CPU fallback (`MultiHamiltonianRuleNumba`) for classic
      `netket.operator.DiscreteOperator` via `jax.pure_callback`
    - Correct detailed balance by treating the operator index as an auxiliary
      latent variable in the proposal, acceptance ratio uses the same operator on
      the forward/backward legs:

        q((i, σ) → (i, σ')) = p(i) · 1/n_conn_i(σ)

      => log-correction = log n_conn_i(σ) − log n_conn_i(σ').
    - It has two selection modes
      - `choose_per_chain=False` (default): one random operator is drawn once per
        sweep and applied to the whole batch (maximal throughput, best for many identical operators
        — e.g. list of copies)
      - `choose_per_chain=True`: every chain draws its own operator (more mixing, slightly less
        throughput)
    - Optional operator probabilities (`p_ops`), defaulting to uniform

Implementation notes
--------------------
    - JAX path uses `lax.switch` and vectorised primitives
    - Per-chain selection path relies on `vmap` over chains and computes exactly one operator per
      chain to avoid K-fold work
    - Numba path is implemented in a single `pure_callback` with a fast, fully Numpy/Numba inner
      loop, no NetKet NumPy-sampler is used

"""

from __future__ import annotations

from typing import Sequence

import math
import jax
import jax.numpy as jnp
import numpy as np

from netket.utils import struct
from netket.sampler.rules.base import MetropolisRule
from netket.operator import DiscreteOperator
from netket.operator import DiscreteJaxOperator

from neuralqx.operators.types._discrete_operator import (
    DiscreteOperator as DiscreteOperatorNqx,
)

DiscreteOperatorT = (DiscreteOperator, DiscreteOperatorNqx)


def _is_known_safe_jax(op):
    # accept NetKet’s own JAX operators
    # this will also accept our ComputationalJaxOperator as it subclasses NetKet's
    # DiscreteJaxOperator
    return isinstance(op, DiscreteJaxOperator)


def _canonical_probs(
    p_ops: jnp.ndarray | None, K: int, dtype=jnp.float32
) -> jnp.ndarray:
    """Return a (K,) probability vector (uniform if None)."""

    if p_ops is None:
        return jnp.full((K,), 1.0 / K, dtype=dtype)

    p_ops = jnp.asarray(p_ops, dtype=dtype)
    if p_ops.shape != (K,):
        raise ValueError(f"p_ops must have shape {(K,)}, got {p_ops.shape}")

    # normalise robustly
    s = jnp.sum(p_ops)

    return jnp.where(s > 0, p_ops / s, jnp.full_like(p_ops, 1.0 / K))


def _sample_categorical(key: jax.Array, p: jnp.ndarray) -> jax.Array:
    """Categorical sample in [0,K) given probs p with shape (K,)."""

    # Gumbel-max trick is statically shaped and good on TPU/GPU

    g = -jnp.log(
        -jnp.log(jax.random.uniform(key, shape=p.shape, minval=0.0, maxval=1.0))
    )

    return jnp.argmax(jnp.log(p) + g)


@struct.dataclass
class MultiHamiltonianRuleJax(MetropolisRule):
    """Metropolis rule over a mixture of JAX-compatible discrete operators.

    The proposal is constructed as:
        1) Draw an operator index i ~ Categorical(p_ops),
        2) Draw σ' uniformly among the connected states of operator i.

    Acceptance log-correction uses only operator i (auxiliary-variable MH),
    ensuring detailed balance without summing over operators.

    Parameters
    ----------
    operators : Sequence[DiscreteJaxOperator]
        List/tuple of JAX operators (e.g. `ComputationalJaxOperator`, `LocalOperatorJax`).
    p_ops : array-like, shape (K,), optional
        Probabilities to pick each operator. Defaults to uniform.
    choose_per_chain : bool
        If False (default), a single operator index is drawn for the whole batch
        per sweep (fastest). If True, each chain draws its own operator.
    """

    operators: tuple = struct.field(pytree_node=False)
    """The list of operators to sample according to."""

    p_ops: jnp.ndarray | None = struct.field(pytree_node=True, default=None)
    """The probability for each operator to be selected. If None, this will be uniform."""

    choose_per_chain: bool = struct.field(pytree_node=False, default=False)
    """
    If False, a single operator index is drawn for the whole batch per sweep, otherwise each chain
    draws its own operator
    """

    def __init__(
        self,
        operators: Sequence[DiscreteJaxOperator],
        p_ops: np.ndarray | jnp.ndarray | None = None,
        *,
        choose_per_chain: bool = False,
    ):
        if not isinstance(operators, (list, tuple)) or len(operators) == 0:
            raise ValueError(
                "`operators` must be a non-empty list/tuple of Jax operators"
            )

        for op in operators:
            if not isinstance(op, DiscreteJaxOperator):
                raise TypeError(
                    f"All operators must be DiscreteJaxOperator-compatible, got {type(op)}."
                )

        self.operators = tuple(operators)
        self.p_ops = None if p_ops is None else jnp.asarray(p_ops)
        self.choose_per_chain = bool(choose_per_chain)

    @staticmethod
    def _one_op_transition(op: DiscreteJaxOperator, key: jax.Array, x: jnp.ndarray):
        """Single-operator HamiltonianRuleJax transition on a batch x."""

        xp, mels = op.get_conn_padded(x)

        # count non-zeros per chain
        nonzeros = jnp.abs(mels) > 0
        n_conn = jnp.sum(nonzeros, axis=-1)

        # sample a uniform index in [0, n_conn)
        # guard n_conn>=1: if 0, keep state (self-loop) and 0 correction
        n_conn_safe = jnp.maximum(n_conn, 1)
        rand_i = jax.random.randint(
            key, shape=(x.shape[0],), minval=0, maxval=n_conn_safe
        )

        # build mask selecting the rand-th nonzero
        nz_cum = (jnp.cumsum(nonzeros, axis=-1)) * nonzeros
        mask = nz_cum == jnp.expand_dims(rand_i + 1, -1)
        x_proposed = (xp * jnp.expand_dims(mask, -1)).sum(axis=1, dtype=xp.dtype)

        # compute backward count with *same* operator
        n_conn_proposed = op.n_conn(x_proposed)
        n_conn_proposed = jnp.maximum(n_conn_proposed, 1)
        log_prob_corr = jnp.log(n_conn_safe) - jnp.log(n_conn_proposed)

        # if n_conn == 0, force identity proposal and zero correction
        x_proposed = jnp.where(jnp.expand_dims(n_conn > 0, -1), x_proposed, x)
        log_prob_corr = jnp.where(
            n_conn > 0, log_prob_corr, jnp.zeros_like(log_prob_corr)
        )
        return x_proposed.astype(x.dtype), log_prob_corr

    def transition(self, sampler, machine, parameters, state, key, x):

        K = len(self.operators)
        p = _canonical_probs(self.p_ops, K)

        if not self.choose_per_chain:

            # one operator for the whole batch (most efficient)
            key_i, key_move = jax.random.split(key)
            i = _sample_categorical(key_i, p)

            def branch_fun(j):
                # close over j
                def _f(x_in_key):
                    x_in, k = x_in_key
                    return self._one_op_transition(self.operators[j], k, x_in)

                return _f

            x_proposed, log_prob_corr = jax.lax.switch(
                i,
                tuple(branch_fun(j) for j in range(K)),
                operand=(x, key_move),
            )
            return x_proposed, log_prob_corr

        # per-chain operator selection (more mixing)
        key_i, key_moves = jax.random.split(key)
        keys_i = jax.random.split(key_i, x.shape[0])
        keys_mv = jax.random.split(key_moves, x.shape[0])
        p_b = jnp.broadcast_to(p, (x.shape[0], K))

        def per_chain(x_i, ki, km):
            idx = _sample_categorical(ki, p)

            # switch over operators for this chain only
            def branch(j):
                def _f(_):
                    xp, lpc = self._one_op_transition(
                        self.operators[j], km, x_i[None, :]
                    )
                    return xp[0], lpc[0]

                return _f

            return jax.lax.switch(idx, tuple(branch(j) for j in range(K)), operand=None)

        x_prop, lpc = jax.vmap(per_chain, in_axes=(0, 0, 0))(x, keys_i, keys_mv)
        return x_prop, lpc

    def __repr__(self):
        # debug helper
        kind = "per-chain" if self.choose_per_chain else "per-sweep"
        return f"MultiHamiltonianRuleJax(K={len(self.operators)}, mode={kind})"


@struct.dataclass
class MultiHamiltonianRuleNumba(MetropolisRule):
    """Metropolis rule over a mixture of classic (non-JAX) `DiscreteOperator`s.

    This implementation executes within a `jax.pure_callback`, running fully on CPU with
    Numpy/Numba. It's compatible with NetKet's default JAX sampler, and does not use the Numpy
    sampler backend.

    The proposal follows the same auxiliary-variable MH construction as the JAX variant, acceptance
    log-correction uses the chosen operator only.
    """

    operators: tuple = struct.field(pytree_node=False)
    """The list of operators to sample according to."""

    p_ops: jnp.ndarray | None = struct.field(pytree_node=True, default=None)
    """The probability for each operator to be selected. If None, this will be uniform."""

    choose_per_chain: bool = struct.field(pytree_node=False, default=False)
    """
    If False, a single operator index is drawn for the whole batch per sweep, otherwise each chain
    draws its own operator
    """

    def __init__(
        self,
        operators: Sequence[DiscreteOperatorT],
        p_ops: np.ndarray | None = None,
        *,
        choose_per_chain: bool = False,
    ):
        if not isinstance(operators, (list, tuple)) or len(operators) == 0:
            raise ValueError("`operators` must be a non-empty list/tuple of operators")

        for op in operators:
            if not isinstance(op, DiscreteOperatorT):
                raise TypeError(
                    f"All operators must be DiscreteOperator, got {type(op)}"
                )

        self.operators = tuple(operators)
        self.p_ops = None if p_ops is None else np.asarray(p_ops, dtype=np.float32)
        self.choose_per_chain = bool(choose_per_chain)

    def transition(self, sampler, machine, parameters, state, key, x):
        B = x.shape[0]
        K = len(self.operators)
        log_prob_dtype = jax.dtypes.canonicalize_dtype(float)

        # RNG: generate op choices and uniform floats for connection selection
        key_ops, key_conn = jax.random.split(key)
        if self.p_ops is None:
            op_float = jax.random.uniform(key_ops, shape=(B,))
            # convert uniform float -> categorical index with uniform probabilities
            op_idx = jnp.floor(op_float * K).astype(jnp.int32)
        else:
            # multinomial per-chain, we rely on host-side categoricals in callback for exact probs
            op_idx = jax.random.randint(key_ops, (B,), 0, K)
        rand_conn = jax.random.uniform(key_conn, shape=(B,))

        def _callback(v: np.ndarray, idx: np.ndarray, r: np.ndarray):
            # v : (B,D), idx : (B,), r : (B,)  (NumPy arrays on host)
            x_prop = np.empty_like(v)
            log_corr = np.zeros((v.shape[0],), dtype=log_prob_dtype)

            # if probabilities provided, precompute categorical cutpoints
            p = self.p_ops
            if p is not None:
                p = p / p.sum()
                cdf = np.cumsum(p)

            for b in range(v.shape[0]):
                # possibly override idx[b] using p-operators if requested
                if p is not None and self.choose_per_chain:
                    u = np.random.random()
                    i = int(np.searchsorted(cdf, u, side="right"))
                else:
                    i = int(idx[b]) % K
                op = self.operators[i]

                sections = np.empty(1, dtype=np.int32)
                vp, _ = op.get_conn_flattened(v[b : b + 1], sections)

                low = 0
                high = sections[0]
                if high - low <= 0:
                    # no move
                    x_prop[b] = v[b]
                    log_corr[b] = 0.0
                    continue

                n_rand = low + int(math.floor(r[b] * (high - low)))
                x_prop[b] = vp[n_rand]

                # correction term using SAME operator
                sections2 = np.empty(1, dtype=np.int32)
                op.n_conn(x_prop[b : b + 1], sections2)
                n1 = max(high - low, 1)
                n2 = max(sections2[0] - 0, 1)
                log_corr[b] = math.log(n1) - math.log(n2)

            return x_prop, log_corr

        x_proposed, log_prob_corr = jax.pure_callback(
            _callback,
            (
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct((B,), log_prob_dtype),
            ),
            x,
            op_idx,
            rand_conn,
            # vmap_method="scan",
        )
        return x_proposed, log_prob_corr

    def __repr__(self):
        kind = "per-chain" if self.choose_per_chain else "per-sweep"
        return f"MultiHamiltonianRuleNumba(K={len(self.operators)}, mode={kind})"


def MultiHamiltonianRule(
    operators: Sequence[DiscreteOperatorT | DiscreteJaxOperator],
    p_ops: np.ndarray | jnp.ndarray | None = None,
    *,
    choose_per_chain: bool = False,
) -> MetropolisRule:
    """Construct a multi-operator Hamiltonian rule, dispatching to JAX/Numba.

    :param operators: sequence of operators. Either all JAX-compatible operators
     (DiscreteJaxOperator, including ComputationalJaxOperator) or all classic CPU
     `DiscreteOperator`s
    :param p_ops: optional probabilities for operator selection (uniform if None)
    :param choose_per_chain: draw an operator per chain (True) or per sweep/batch (False)
    """

    if len(operators) == 0:
        raise ValueError("operators list must be non-empty")

    from netket.operator._lazy import WrappedOperator

    stripped_operators = []

    for op in operators:
        if isinstance(op, WrappedOperator):
            stripped_operators.append(op.parent)
        else:
            stripped_operators.append(op)

    # check homogeneity of types
    all_safe_jax = all(_is_known_safe_jax(op) for op in stripped_operators)
    all_safe_cpu = all(isinstance(op, DiscreteOperatorT) for op in stripped_operators)

    if not (all_safe_jax or all_safe_cpu):
        raise TypeError(
            "All operators must be either Jax (DiscreteJaxOperator) or CPU (DiscreteOperator), "
            "not mixed."
        )

    # dev: technically, DiscreteJaxOperators subclass DiscreteOperators, so when we have a JAX
    #      operators, the all_safe_cpu will be truthy, so we need to check for JAX first. It is
    #      not the case the other way around
    if all_safe_jax:
        return MultiHamiltonianRuleJax(
            stripped_operators, p_ops, choose_per_chain=choose_per_chain
        )
    else:
        return MultiHamiltonianRuleNumba(
            stripped_operators, p_ops, choose_per_chain=choose_per_chain
        )

    # dev: handle the case for a mixture of JAX and non-JAX types. If left as-is, this causes a
    #      crash because it will use the Numba rule, and it has a pure-callback, and that will crash
    #      with NetKet's DiscreteJaxOperator.n_conn as it will try to set values for it via slicing
    #      and not via .at[].set[] etc.
