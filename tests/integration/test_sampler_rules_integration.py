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

from __future__ import annotations

import os
import sys
import inspect
import importlib
import types
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("NQS_VERBOSE", "False")

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _ensure_real_netket():
    nk = pytest.importorskip("netket")

    looks_fake = not hasattr(nk, "__version__") or not hasattr(nk, "sampler")
    if looks_fake:
        for name in list(sys.modules.keys()):
            if name == "netket" or name.startswith("netket."):
                sys.modules.pop(name, None)
        nk = importlib.import_module("netket")

    if not hasattr(nk, "sampler") or not hasattr(nk.sampler, "MetropolisSampler"):
        pytest.skip("Real NetKet not available (or a stubbed netket is active).")

    return nk


nk = _ensure_real_netket()
neuralqx = pytest.importorskip("neuralqx")

from neuralqx.samplers.rules._gauge_samplers import (
    RandomU1GaugeSampler,
    U1GaugeSampler,
    U1GaugeSamplerNonzero,
    U1InvariantPlaquetteSampler,
)
from neuralqx.samplers.rules._metropolis_samplers import (
    MetropolisKLocalRule,
)
from neuralqx.samplers.rules._mu_sampler import MuSampler
from neuralqx.samplers.rules._weighted_sampler import WeightedSamplerRule
from neuralqx.samplers.rules._hamiltonian_list import MultiHamiltonianRule


def _const_logpdf(_: Any, sigma: jax.Array) -> jax.Array:

    if sigma.ndim == 1:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.zeros((sigma.shape[0],), dtype=jnp.float32)


def _construct_with_accepted_kwargs(callable_obj, **kwargs):

    sig = inspect.signature(callable_obj)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return callable_obj(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return callable_obj(**accepted)


def _mk_metropolis_sampler(hilbert, rule, *, n_chains: int, sweep_size: int = 1):

    variants = [
        dict(
            hilbert=hilbert,
            rule=rule,
            n_chains_per_rank=n_chains,
            sweep_size=sweep_size,
            reset_chains=False,
            machine_pow=1,
        ),
        dict(
            hilbert=hilbert,
            rule=rule,
            n_chains=n_chains,
            sweep_size=sweep_size,
            reset_chains=False,
            machine_pow=1,
        ),
        dict(
            hilbert=hilbert,
            rule=rule,
            n_chains=n_chains,
            sweep_size=sweep_size,
            reset_chains=False,
            machine_power=1,
        ),
    ]
    last = None
    for kw in variants:
        try:
            return _construct_with_accepted_kwargs(nk.sampler.MetropolisSampler, **kw)
        except TypeError as e:
            last = e
    raise last


def _mk_pt_sampler(
    hilbert, rule, *, n_chains: int, sweep_size: int = 1, n_replicas: int = 2
):

    variants = [
        dict(
            hilbert=hilbert,
            rule=rule,
            n_replicas=n_replicas,
            betas="linear",
            n_chains_per_rank=n_chains,
            sweep_size=sweep_size,
            reset_chains=False,
            machine_pow=1,
        ),
        dict(
            hilbert=hilbert,
            rule=rule,
            n_replicas=n_replicas,
            betas="linear",
            n_chains=n_chains,
            sweep_size=sweep_size,
            reset_chains=False,
            machine_pow=1,
        ),
        dict(
            hilbert=hilbert,
            rule=rule,
            n_replicas=n_replicas,
            betas="linear",
            n_chains=n_chains,
            sweep_size=sweep_size,
            reset_chains=False,
            machine_power=1,
        ),
    ]
    last = None
    for kw in variants:
        try:
            return _construct_with_accepted_kwargs(
                nk.sampler.ParallelTemperingSampler, **kw
            )
        except (TypeError, ValueError) as e:
            last = e
    raise last


def _init_reset_state(sampler, seed: int):

    try:
        st = sampler.init_state(_const_logpdf, None, seed=seed)
    except TypeError:
        st = sampler.init_state(_const_logpdf, None, seed=jax.random.PRNGKey(seed))

    try:
        st = sampler.reset(_const_logpdf, None, state=st)
    except TypeError:
        st = sampler.reset(_const_logpdf, None, st)

    return st


def _sampler_sample(sampler, state, chain_length: int):

    sig = inspect.signature(sampler.sample)
    if "chain_length" in sig.parameters:
        return sampler.sample(
            _const_logpdf, None, state=state, chain_length=chain_length
        )

    if "n_samples" in sig.parameters:
        return sampler.sample(_const_logpdf, None, state=state, n_samples=chain_length)

    return sampler.sample(_const_logpdf, None, state=state, chain_length=chain_length)


def _sample_chain(sampler, *, seed: int, chain_length: int):
    st = _init_reset_state(sampler, seed)
    samples, st2 = _sampler_sample(sampler, st, chain_length=chain_length)
    return np.asarray(samples), st2


def _standardize_samples(
    samples: np.ndarray, *, chain_length: int, hilbert_size: int
) -> np.ndarray:

    a = np.asarray(samples)
    if a.ndim < 3:
        raise ValueError(f"Expected samples rank>=3, got shape={a.shape}")

    d_axes = [i for i, s in enumerate(a.shape) if s == hilbert_size]
    if not d_axes:
        d_axis = a.ndim - 1
    else:
        d_axis = d_axes[-1]

    t_axes = [i for i, s in enumerate(a.shape) if s == chain_length and i != d_axis]
    if not t_axes:
        raise ValueError(
            f"Could not locate chain_length axis={chain_length} in shape={a.shape}"
        )

    t_axis = (d_axis - 1) if (d_axis - 1) in t_axes else t_axes[0]

    perm = [t_axis] + [i for i in range(a.ndim) if i not in (t_axis, d_axis)] + [d_axis]
    b = np.transpose(a, axes=perm)

    T = b.shape[0]
    D = b.shape[-1]
    if T != chain_length or D != hilbert_size:
        raise ValueError(f"Bad standardized axes: got T={T},D={D} from shape={a.shape}")

    B = int(np.prod(b.shape[1:-1])) if b.ndim > 2 else 1
    return b.reshape((T, B, D))


def _flatten_samples(samples_tbd: np.ndarray) -> np.ndarray:
    T, B, D = samples_tbd.shape
    return samples_tbd.reshape((T * B, D))


def _hamming_step_counts(samples_tbd: np.ndarray) -> np.ndarray:
    s0 = samples_tbd[:-1]
    s1 = samples_tbd[1:]
    return np.sum(s0 != s1, axis=-1)


def _changed_indices_per_step(samples_tbd: np.ndarray) -> np.ndarray:

    changed = samples_tbd[1:] != samples_tbd[:-1]
    has = np.any(changed, axis=-1)
    idx = np.argmax(changed, axis=-1).astype(np.int32)
    idx[~has] = -1
    return idx


def _warmup_jax_operator(op, hilb_size: int):

    try:
        x = jnp.zeros((1, hilb_size), dtype=jnp.int8)
        _ = op.get_conn_padded(x)
        _ = op.n_conn(x)
    except Exception:
        pass


@pytest.fixture(scope="module")
def u1_gauge_hilbert():
    graph = neuralqx.graph.Graph([(0, 1), (0, 2), (0, 3), (3, 2), (1, 2)])
    H = neuralqx.hilbert.u1.HilbertU1(
        graph,
        cutoff=1,
        is_gauge_invariant=True,
        auto_constraint=1,
        gauge_dimensions=1,
    )
    return H, H.hilbert_netket


@pytest.mark.parametrize(
    "RuleCls",
    [U1GaugeSampler, RandomU1GaugeSampler, U1InvariantPlaquetteSampler],
)
def test_gauge_rules_metropolis_preserve_gauge_invariance(u1_gauge_hilbert, RuleCls):
    H, hilb = u1_gauge_hilbert
    sampler = _mk_metropolis_sampler(hilb, RuleCls(), n_chains=32, sweep_size=1)

    raw, _ = _sample_chain(sampler, seed=0, chain_length=25)
    s = _standardize_samples(raw, chain_length=25, hilbert_size=hilb.size)
    flat = _flatten_samples(s)

    gi = np.asarray(H.hilbert.is_gauge_invariant(jnp.asarray(flat)))
    assert gi.all(), "Gauge rule produced samples outside the gauge-invariant subspace."
    assert np.any(
        np.any(flat != flat[0], axis=1)
    ), "Sampler appears stuck (all samples identical)."


def test_u1_gauge_nonzero_rule_never_hits_vacuum(u1_gauge_hilbert):
    H, hilb = u1_gauge_hilbert
    sampler = _mk_metropolis_sampler(
        hilb, U1GaugeSamplerNonzero(), n_chains=32, sweep_size=1
    )

    raw, _ = _sample_chain(sampler, seed=1, chain_length=30)
    s = _standardize_samples(raw, chain_length=30, hilbert_size=hilb.size)
    flat = _flatten_samples(s)

    gi = np.asarray(H.hilbert.is_gauge_invariant(jnp.asarray(flat)))
    assert gi.all(), "Nonzero gauge rule produced non-gauge-invariant samples."
    assert not np.any(
        np.all(flat == 0, axis=1)
    ), "Nonzero gauge rule produced the vacuum state."


def test_random_u1_gauge_rule_parallel_tempering_branch_preserves_gauge_invariance(
    u1_gauge_hilbert,
):
    H, hilb = u1_gauge_hilbert
    sampler = _mk_pt_sampler(
        hilb, RandomU1GaugeSampler(), n_chains=16, sweep_size=1, n_replicas=2
    )

    raw, _ = _sample_chain(sampler, seed=2, chain_length=20)
    s = _standardize_samples(raw, chain_length=20, hilbert_size=hilb.size)
    flat = _flatten_samples(s)

    gi = np.asarray(H.hilbert.is_gauge_invariant(jnp.asarray(flat)))
    assert gi.all(), "PT + RandomU1GaugeSampler produced non-gauge-invariant samples."


def test_metropolis_klocal_rule_respects_max_hamming_distance():
    hi = nk.hilbert.Spin(s=0.5, N=10)
    K = 3
    sampler = _mk_metropolis_sampler(
        hi, MetropolisKLocalRule(n_flips=K), n_chains=64, sweep_size=1
    )

    raw, _ = _sample_chain(sampler, seed=3, chain_length=25)
    s = _standardize_samples(raw, chain_length=25, hilbert_size=hi.size)

    local_states = np.asarray(hi.local_states)
    assert np.isin(s, local_states).all()

    diff = _hamming_step_counts(s)
    assert diff.max() <= K, f"Observed step with >{K} site changes (max={diff.max()})."


def test_mu_sampler_enforces_nonzero_on_forbidden_indices(monkeypatch):
    hi = nk.hilbert.Spin(s=1.0, N=8)
    forbidden = np.array([1, 3, 6], dtype=np.int32)

    g = types.SimpleNamespace(edges_mu_idx=jnp.asarray(forbidden))
    try:
        monkeypatch.setattr(hi, "graph", g, raising=False)
    except Exception:
        monkeypatch.setattr(hi.__class__, "graph", g, raising=False)

    sampler = _mk_metropolis_sampler(hi, MuSampler(), n_chains=64, sweep_size=1)

    raw, _ = _sample_chain(sampler, seed=4, chain_length=30)
    s = _standardize_samples(raw, chain_length=30, hilbert_size=hi.size)

    assert np.all(
        s[..., forbidden] != 0
    ), "MuSampler produced zero on forbidden indices."


def test_weighted_sampler_rule_integration_runs_in_netket_sampler():
    hi = nk.hilbert.Spin(s=0.5, N=12)
    rule_local = nk.sampler.rules.LocalRule()
    rule_k2 = MetropolisKLocalRule(n_flips=2)

    wr = WeightedSamplerRule(
        probabilities=jnp.asarray([0.5, 0.5]), rules=(rule_local, rule_k2)
    )
    sampler = _mk_metropolis_sampler(hi, wr, n_chains=64, sweep_size=1)

    raw, _ = _sample_chain(sampler, seed=5, chain_length=10)
    s = _standardize_samples(raw, chain_length=10, hilbert_size=hi.size)
    assert s.shape[0] == 10


def _sigmax_ops_jax(hi):
    ops = [
        nk.operator.spin.sigmax(hi, 0).to_jax_operator(),
        nk.operator.spin.sigmax(hi, 1).to_jax_operator(),
    ]

    for op in ops:
        _warmup_jax_operator(op, hi.size)
    return ops


def _sigmax_ops_numba(hi):
    if not hasattr(nk.operator, "LocalOperator"):
        pytest.skip("NetKet does not expose LocalOperator in this version.")
    try:
        ops = [
            nk.operator.spin.sigmax(hi, 0),
            nk.operator.spin.sigmax(hi, 1),
        ]
    except TypeError:
        pytest.skip("This NetKet version does not support LocalOperator for sigmax().")
    return ops


def test_multihamiltonian_rule_jax_runs_and_choose_per_chain_behavior():

    hi = nk.hilbert.Spin(s=0.5, N=6)
    ops = _sigmax_ops_jax(hi)

    rule_sweep = MultiHamiltonianRule(ops, choose_per_chain=False)
    sampler_sweep = _mk_metropolis_sampler(hi, rule_sweep, n_chains=128, sweep_size=1)
    raw_sweep, _ = _sample_chain(sampler_sweep, seed=6, chain_length=25)
    s_sweep = _standardize_samples(raw_sweep, chain_length=25, hilbert_size=hi.size)

    diff = _hamming_step_counts(s_sweep)
    assert (
        diff.max() <= 1
    ), f"sigmax should flip at most 1 spin per step (max={diff.max()})."

    idx_sweep = _changed_indices_per_step(s_sweep)
    for t in range(idx_sweep.shape[0]):
        moved = idx_sweep[t][idx_sweep[t] != -1]
        assert moved.size > 0
        assert (
            np.unique(moved).size == 1
        ), "Expected same flipped site across all chains (choose_per_chain=False)."

    rule_chain = MultiHamiltonianRule(ops, choose_per_chain=True)
    sampler_chain = _mk_metropolis_sampler(hi, rule_chain, n_chains=256, sweep_size=1)
    raw_chain, _ = _sample_chain(sampler_chain, seed=7, chain_length=40)
    s_chain = _standardize_samples(raw_chain, chain_length=40, hilbert_size=hi.size)

    diff2 = _hamming_step_counts(s_chain)
    assert (
        diff2.max() <= 1
    ), f"sigmax should flip at most 1 spin per step (max={diff2.max()})."

    idx_chain = _changed_indices_per_step(s_chain)
    saw_mixed = False
    for t in range(idx_chain.shape[0]):
        moved = idx_chain[t][idx_chain[t] != -1]
        if moved.size > 0 and np.unique(moved).size > 1:
            saw_mixed = True
            break
    assert (
        saw_mixed
    ), "Expected at least one step with mixed flipped sites (choose_per_chain=True)."


def test_multihamiltonian_rule_numba_runs_on_cpu_operators():

    hi = nk.hilbert.Spin(s=0.5, N=6)
    ops = _sigmax_ops_numba(hi)

    rule = MultiHamiltonianRule(ops, choose_per_chain=False)
    sampler = _mk_metropolis_sampler(hi, rule, n_chains=64, sweep_size=1)

    raw, _ = _sample_chain(sampler, seed=8, chain_length=20)
    s = _standardize_samples(raw, chain_length=20, hilbert_size=hi.size)

    diff = _hamming_step_counts(s)
    assert (
        diff.max() <= 1
    ), f"sigmax should flip at most 1 spin per step (max={diff.max()})."

    local_states = np.asarray(hi.local_states)
    assert np.isin(s, local_states).all()
