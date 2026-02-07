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

import importlib
import numpy as np
import pytest
import functools

import flax.linen as nn


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


def import_functional_local_operator():
    """Robustly import the FunctionalLocalOperator from likely neuraLQX locations."""
    candidates = [
        ("neuralqx.operators.types", "FunctionalLocalOperator"),
        ("neuralqx.operators.types.numba", "FunctionalLocalOperator"),
        ("neuralqx.operators", "FunctionalLocalOperator"),
    ]
    errors = []
    for mod, name in candidates:
        try:
            m = importlib.import_module(mod)
            return getattr(m, name)
        except Exception as e:
            errors.append((mod, name, repr(e)))
    raise ImportError("Could not import FunctionalLocalOperator. Tried: " + str(errors))


def make_local_operator(hilbert, opmat, site: int, *, dtype=None, mel_cutoff=None):
    """Create a NetKet LocalOperator robustly across signature changes."""
    tried = []
    if dtype is None:
        dtype = opmat.dtype
    kwargs = {}
    if mel_cutoff is not None:
        kwargs["mel_cutoff"] = mel_cutoff

    for attempt in [
        lambda: _nk().operator.LocalOperator(
            hilbert, opmat, acting_on=[site], dtype=dtype, **kwargs
        ),
        lambda: _nk().operator.LocalOperator(
            hilbert, operators=opmat, acting_on=[site], dtype=dtype, **kwargs
        ),
        lambda: _nk().operator.LocalOperator(
            hilbert, operators=[opmat], acting_on=[[site]], dtype=dtype, **kwargs
        ),
        lambda: _nk().operator.LocalOperator(
            hilbert, operators=[opmat], acting_on=[site], dtype=dtype, **kwargs
        ),
    ]:
        try:
            return attempt()
        except TypeError as e:
            tried.append(str(e))
    raise TypeError(
        "Unable to construct nk.operator.LocalOperator. Tried:\n" + "\n".join(tried)
    )


def make_functional_local_operator(
    hilbert, opmat, site: int, *, mels_func, dtype=None, mel_cutoff=None
):
    """Create neuraLQX FunctionalLocalOperator robustly across signature changes."""
    FLO = import_functional_local_operator()
    tried = []
    if dtype is None:
        dtype = opmat.dtype
    kwargs = {"dtype": dtype, "mels_func": mels_func}
    if mel_cutoff is not None:
        kwargs["mel_cutoff"] = mel_cutoff

    for attempt in [
        lambda: FLO(hilbert, opmat, [site], **kwargs),
        lambda: FLO(hilbert, operators=opmat, acting_on=[site], **kwargs),
        lambda: FLO(hilbert, operators=[opmat], acting_on=[[site]], **kwargs),
        lambda: FLO(hilbert, operators=[opmat], acting_on=[site], **kwargs),
    ]:
        try:
            return attempt()
        except TypeError as e:
            tried.append(str(e))
    raise TypeError(
        "Unable to construct FunctionalLocalOperator. Tried:\n" + "\n".join(tried)
    )


def conn_as_pairs(hilbert, xp, mels, x):
    """Return list of per-sample sorted (state_number, mel) pairs, ignoring padded zeros."""
    xp = np.asarray(xp)
    mels = np.asarray(mels)
    x = np.asarray(x)

    batch = x.shape[0]
    D = x.shape[1]
    out = []
    for b in range(batch):
        pairs = []
        for k in range(xp.shape[1]):
            mel = mels[b, k]
            if np.isclose(mel, 0.0):
                continue
            st = xp[b, k]
            num = int(hilbert.states_to_numbers(st.reshape(1, D))[0])
            pairs.append((num, complex(mel)))
        pairs.sort(key=lambda t: t[0])
        out.append(pairs)
    return out


def assert_conn_transformed(hilbert, op_base, op_fun, x, transform, atol=1e-12):
    xp0, m0 = op_base.get_conn_padded(x)
    xpf, mf = op_fun.get_conn_padded(x)

    p0 = conn_as_pairs(hilbert, xp0, m0, x)
    pf = conn_as_pairs(hilbert, xpf, mf, x)

    assert len(p0) == len(pf)
    for a, b in zip(p0, pf):
        assert [i for i, _ in a] == [i for i, _ in b]
        for (_, mel0), (_, melf) in zip(a, b):
            assert np.allclose(transform(mel0), melf, atol=atol)


def dense_nonzero_transform(dense, transform):
    dense = np.asarray(dense)
    out = np.zeros_like(dense)
    mask = dense != 0
    out[mask] = transform(dense[mask])
    return out


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


def make_mcstate(
    hilbert, *, seed=0, n_samples=4096, n_chains=16, n_discard_per_chain=64
):
    sampler = _nk().sampler.MetropolisLocal(hilbert, n_chains=n_chains)
    model = TinyComplexModel(hidden=8)
    rng_key = _jax().random.PRNGKey(seed)

    tried = []
    for attempt in [
        lambda: _nqx().vqs.MCState(
            sampler,
            model,
            seed=seed,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
        ),
        lambda: _nqx().vqs.MCState(
            sampler,
            model,
            rng_key=rng_key,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
        ),
        lambda: _nqx().vqs.MCState(
            sampler,
            model,
            seed=seed,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
        ),
        lambda: _nqx().vqs.MCState(
            sampler,
            model,
            rng_key=rng_key,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
        ),
        lambda: _nqx().vqs.MCState(
            sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain
        ),
        lambda: _nqx().vqs.MCState(
            sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain
        ),
        lambda: _nqx().vqs.MCState(sampler, model, n_samples=n_samples),
    ]:
        try:
            st = attempt()
            _ = st.parameters
            return st
        except TypeError as e:
            tried.append(str(e))
            continue
    raise TypeError(
        "Unable to construct neuralqx.vqs.MCState. Tried:\n" + "\n".join(tried)
    )


def exact_expectation_from_dense(vstate, dense):
    """Compute exact <psi|O|psi>/<psi|psi> by enumerating all states."""
    hilb = vstate.hilbert
    assert hilb.is_indexable
    states = np.asarray(hilb.all_states(), dtype=np.int64)
    logpsi = np.asarray(vstate.log_value(states))
    psi = np.exp(logpsi)
    num = psi.conj() @ (np.asarray(dense) @ psi)
    den = psi.conj() @ psi
    return num / den
