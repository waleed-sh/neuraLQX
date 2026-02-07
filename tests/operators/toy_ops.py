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

import numpy as np
import pytest


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


def _raise_on_ket(sigma, site):
    out = np.array(sigma, copy=True)
    if out[site] < 0:
        out[site] = -out[site]
        return out, 1.0
    return sigma, 0.0


def _lower_on_ket(sigma, site):
    out = np.array(sigma, copy=True)
    if out[site] > 0:
        out[site] = -out[site]
        return out, 1.0
    return sigma, 0.0


class KetSigmaz(_nqx().operators.types.ComputationalOperator):

    def __init__(self, hilbert, site: int):
        super().__init__(hilbert)
        self.site = int(site)

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return np.float64

    def _get_conn_padded_kernel(self, x_local):
        x_local = np.asarray(x_local)
        single = x_local.ndim == 2
        if single:
            x_local = x_local[None, :, :]

        mels = np.where(x_local[:, :, self.site] > 0, 1.0, -1.0).astype(np.float64)[
            :, :, None
        ]
        xp = x_local[:, :, None, :]
        return (xp[0], mels[0]) if single else (xp, mels)


class KetSigmap(_nqx().operators.types.ComputationalOperator):

    def __init__(self, hilbert, site: int):
        super().__init__(hilbert)
        self.site = int(site)

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return np.complex128

    def _get_conn_padded_kernel(self, x_local):
        x_local = np.asarray(x_local)
        single = x_local.ndim == 2
        if single:
            x_local = x_local[None, :, :]
        B, N, D = x_local.shape
        xp = np.empty((B, N, 1, D), dtype=x_local.dtype)
        mels = np.empty((B, N, 1), dtype=np.complex128)
        for b in range(B):
            for n in range(N):
                s = x_local[b, n]
                sp, mel = _raise_on_ket(s, self.site)
                xp[b, n, 0] = sp
                mels[b, n, 0] = mel
        return (xp[0], mels[0]) if single else (xp, mels)


class KetSigmam(_nqx().operators.types.ComputationalOperator):

    def __init__(self, hilbert, site: int):
        super().__init__(hilbert)
        self.site = int(site)

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return np.complex128

    def _get_conn_padded_kernel(self, x_local):
        x_local = np.asarray(x_local)
        single = x_local.ndim == 2
        if single:
            x_local = x_local[None, :, :]
        B, N, D = x_local.shape
        xp = np.empty((B, N, 1, D), dtype=x_local.dtype)
        mels = np.empty((B, N, 1), dtype=np.complex128)
        for b in range(B):
            for n in range(N):
                s = x_local[b, n]
                sp, mel = _lower_on_ket(s, self.site)
                xp[b, n, 0] = sp
                mels[b, n, 0] = mel
        return (xp[0], mels[0]) if single else (xp, mels)


def _raise_on_ket_jax(sigma, site):
    flip = sigma[site] < 0
    sigma2 = sigma.at[site].set(_jnp().where(flip, -sigma[site], sigma[site]))
    mel = _jnp().where(flip, 1.0, 0.0)
    return sigma2, mel


def _lower_on_ket_jax(sigma, site):
    flip = sigma[site] > 0
    sigma2 = sigma.at[site].set(_jnp().where(flip, -sigma[site], sigma[site]))
    mel = _jnp().where(flip, 1.0, 0.0)
    return sigma2, mel


@_jax()._src.tree_util.register_pytree_node_class
class KetSigmazJax(_nqx().operators.types.ComputationalJaxOperator):
    def __init__(self, hilbert, site: int):
        super().__init__(hilbert)
        self.site = int(site)

    def tree_flatten(self):
        children = ()
        aux_data = (self.hilbert, self.site)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        hilbert, site = aux_data
        return cls(hilbert=hilbert, site=site)

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self):
        return _jnp().float64

    def _get_conn_padded(self, x):
        x = _jnp().asarray(x, dtype=_jnp().int64)
        single = x.ndim == 2
        if single:
            x = x[None, :, :]
        xp = x[:, :, None, :]
        mels = (
            _jnp()
            .where(x[:, :, self.site] > 0, 1.0, -1.0)
            .astype(self.dtype)[:, :, None]
        )
        return (xp[0], mels[0]) if single else (xp, mels)


@_jax()._src.tree_util.register_pytree_node_class
class KetSigmapJax(_nqx().operators.types.ComputationalJaxOperator):
    def __init__(self, hilbert, site: int):
        super().__init__(hilbert)
        self.site = int(site)

    def tree_flatten(self):
        children = ()
        aux_data = (self.hilbert, self.site)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        hilbert, site = aux_data
        return cls(hilbert=hilbert, site=site)

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return _jnp().complex128

    def _get_conn_padded(self, x):
        x = _jnp().asarray(x, dtype=_jnp().int64)
        single = x.ndim == 2
        if single:
            x = x[None, :, :]

        def one_sigma(s):
            return _raise_on_ket_jax(s, self.site)

        sp, mel = _jax().vmap(_jax().vmap(one_sigma))(x)
        xp = sp[:, :, None, :]
        mels = mel.astype(self.dtype)[:, :, None]
        return (xp[0], mels[0]) if single else (xp, mels)


@_jax()._src.tree_util.register_pytree_node_class
class KetSigmamJax(_nqx().operators.types.ComputationalJaxOperator):
    def __init__(self, hilbert, site: int):
        super().__init__(hilbert)
        self.site = int(site)

    def tree_flatten(self):
        children = ()
        aux_data = (self.hilbert, self.site)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        hilbert, site = aux_data
        return cls(hilbert=hilbert, site=site)

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def is_hermitian(self) -> bool:
        return False

    @property
    def dtype(self):
        return _jnp().complex128

    def _get_conn_padded(self, x):
        x = _jnp().asarray(x, dtype=_jnp().int64)
        single = x.ndim == 2
        if single:
            x = x[None, :, :]

        def one_sigma(s):
            return _lower_on_ket_jax(s, self.site)

        sp, mel = _jax().vmap(_jax().vmap(one_sigma))(x)
        xp = sp[:, :, None, :]
        mels = mel.astype(self.dtype)[:, :, None]
        return (xp[0], mels[0]) if single else (xp, mels)
