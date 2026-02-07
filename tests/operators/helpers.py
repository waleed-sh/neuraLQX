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

import functools
import pytest

import numpy as np


@functools.lru_cache(maxsize=None)
def _jax():
    return pytest.importorskip("jax")


@functools.lru_cache(maxsize=None)
def _jnp():
    jax = _jax()
    import jax.numpy as jnp

    return jnp


@functools.lru_cache(maxsize=None)
def _nk():
    return pytest.importorskip("netket")


@functools.lru_cache(maxsize=None)
def _nqx():
    return pytest.importorskip("neuralqx")


@functools.lru_cache(maxsize=None)
def _nn():
    pytest.importorskip("flax")
    import flax.linen as nn

    return nn


import flax.linen as nn

_SENTINEL = object()


def _getattr_any(obj, names, default=_SENTINEL):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    if default is _SENTINEL:
        raise AttributeError(f"None of {names} found on {type(obj)}")
    return default


def stats_as_dict(stats):
    mean = _getattr_any(stats, ["Mean", "mean"])
    variance = _getattr_any(stats, ["Variance", "variance"], default=None)
    sigma = _getattr_any(stats, ["Sigma", "error_of_mean", "sigma"], default=None)
    rhat = _getattr_any(stats, ["Rhat", "R_hat", "rhat"], default=None)
    return dict(mean=mean, sigma=sigma, variance=variance, rhat=rhat)


def _as_float(x) -> float:
    x = np.asarray(x)
    if np.iscomplexobj(x):
        x = np.maximum(np.abs(x.real), np.abs(x.imag))
    return float(np.max(x))


def assert_means_close(stats_a, stats_b, *, k_sigma=8.0, floor=3e-3):
    da = stats_as_dict(stats_a)
    db = stats_as_dict(stats_b)
    ma = np.asarray(da["mean"])
    mb = np.asarray(db["mean"])

    sa = 0.0 if da["sigma"] is None else _as_float(da["sigma"])
    sb = 0.0 if db["sigma"] is None else _as_float(db["sigma"])
    tol = max(floor, k_sigma * max(sa, sb))

    diff = ma - mb
    if np.iscomplexobj(diff):
        assert (
            abs(diff.real) <= tol and abs(diff.imag) <= tol
        ), f"Mean mismatch: {ma} vs {mb} (tol={tol})"
    else:
        assert abs(float(diff)) <= tol, f"Mean mismatch: {ma} vs {mb} (tol={tol})"


def assert_stats_strict(stats_a, stats_b, *, rtol=1e-5, atol=1e-6):
    da = stats_as_dict(stats_a)
    db = stats_as_dict(stats_b)

    for key, rr, aa in [
        ("mean", rtol, atol),
        ("variance", 1e-4, 1e-5),
        ("sigma", 1e-4, 1e-5),
        ("rhat", 1e-4, 1e-5),
    ]:
        xa = da.get(key)
        xb = db.get(key)
        if xa is None or xb is None:
            continue
        xa = np.asarray(xa)
        xb = np.asarray(xb)
        assert np.allclose(xa, xb, rtol=rr, atol=aa), f"{key} mismatch: {xa} vs {xb}"


def assert_stats_allclose(stats_a, stats_b, request=None):

    assert_means_close(stats_a, stats_b)

    if (
        request is not None
        and request.node.get_closest_marker("strict_stats") is not None
    ):
        assert_stats_strict(stats_a, stats_b)


def dense_from_get_conn_padded(op):

    hilb = op.hilbert
    assert hilb.is_indexable
    x = np.asarray(hilb.all_states(), dtype=np.int64)  # (n_states, D)

    xp, mels = op.get_conn_padded(x)
    xp = np.asarray(xp)
    mels = np.asarray(mels)

    n, D = x.shape
    out = np.zeros((n, n), dtype=np.result_type(mels.dtype, np.complex128))

    j = np.asarray(hilb.states_to_numbers(xp.reshape(-1, D))).reshape(mels.shape)
    i = np.broadcast_to(np.arange(n)[:, None], mels.shape)

    np.add.at(out, (i.reshape(-1), j.reshape(-1)), mels.reshape(-1))
    return out


def contains_comp_leaf(op) -> bool:
    try:
        from neuralqx.operators.types import (
            ComputationalOperator,
            ComputationalJaxOperator,
        )

        comp_types = (ComputationalOperator, ComputationalJaxOperator)
    except Exception:
        comp_types = ()

    if comp_types and isinstance(op, comp_types):
        return True

    for attr in ("A", "B", "parent", "op", "operator"):
        if hasattr(op, attr):
            child = getattr(op, attr)
            if child is None:
                continue
            if isinstance(child, (tuple, list)):
                if any(contains_comp_leaf(c) for c in child):
                    return True
            else:
                if contains_comp_leaf(child):
                    return True
    return False


def exact_wavefunction(vstate):
    hilb = vstate.hilbert
    states = np.asarray(hilb.all_states(), dtype=np.int64)
    logpsi = np.asarray(vstate.log_value(states))
    psi = np.exp(logpsi)
    norm = np.sqrt(np.sum(np.abs(psi) ** 2))
    psi = psi / norm
    return psi, states


def exact_expectation_from_dense(psi, M):
    return np.vdot(psi, M @ psi)


def make_mcstate(hilbert, *, seed=0, n_samples=4096, n_chains=32):

    sampler = None
    try:
        sampler = _nk().sampler.ExactSampler(hilbert)
    except Exception:
        sampler = _nk().sampler.MetropolisLocal(hilbert, n_chains=n_chains)

    model = TinyComplexModel(hidden=8)

    rng_key = _jax().random.PRNGKey(seed)

    attempts = []
    constructors = [
        lambda: _nqx().vqs.MCState(sampler, model, seed=seed, n_samples=n_samples),
        lambda: _nqx().vqs.MCState(
            sampler, model, rng_key=rng_key, n_samples=n_samples
        ),
        lambda: _nqx().vqs.MCState(sampler, model, n_samples=n_samples),
    ]
    for c in constructors:
        try:
            st = c()
            _ = st.parameters
            return st
        except TypeError as e:
            attempts.append(str(e))
            continue
    raise TypeError(
        "Unable to construct neuralqx.vqs.MCState. Attempts:\n" + "\n".join(attempts)
    )


class TinyComplexModel(nn.Module):
    hidden: int = 8

    @nn.compact
    def __call__(self, x):
        x = x.astype(_jnp().float32)
        h = nn.Dense(self.hidden)(x)
        h = _jnp().tanh(h)
        re = nn.Dense(1)(h)[..., 0]
        im = nn.Dense(1)(h)[..., 0]
        return re + 1j * im
