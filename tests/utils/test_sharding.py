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


import importlib
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def jax():
    return pytest.importorskip("jax")


@pytest.fixture
def jnp(jax):
    return pytest.importorskip("jax.numpy")


@pytest.fixture
def smod():
    return importlib.import_module("neuralqx.utils.jax.sharding")


class DummyConnPadded:

    def __init__(self, max_conn: int, mel_dtype):
        self.max_conn = int(max_conn)
        self.mel_dtype = mel_dtype

    def __call__(self, x):
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"Expected x.ndim==2, got {x.ndim}")

        b, n = x.shape
        xp = np.repeat(x[:, None, :], self.max_conn, axis=1)
        mels = np.zeros((b, self.max_conn), dtype=self.mel_dtype)
        return xp, mels


def _maybe_put_sharded(jax, x):

    try:
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        axis = "S"
        try:
            abstract_mesh = jax.sharding.get_abstract_mesh()
            names = tuple(getattr(abstract_mesh, "axis_names", ()) or ())
            if names:
                axis = str(names[0])
        except Exception:
            pass

        devices = jax.devices()
        mesh = Mesh(devices, axis_names=(axis,))
        sharding = NamedSharding(mesh, P(axis))

        return jax.device_put(x, sharding)
    except Exception:
        return x


def test_replicate_sharding_returns_callable(smod):
    f_py = DummyConnPadded(max_conn=3, mel_dtype=np.float32)
    wrapped = smod.replicate_sharding(f_py)
    assert callable(wrapped)


def test_replicate_sharding_shapes_values_and_dtypes_under_jit(smod, jax, jnp):
    ndev = max(1, jax.device_count())

    batch = ndev * 2
    n_sites = 5
    max_conn = 4

    f_py = DummyConnPadded(max_conn=max_conn, mel_dtype=np.float32)
    wrapped = smod.replicate_sharding(f_py)

    x = jnp.arange(batch * n_sites, dtype=jnp.int32).reshape(batch, n_sites)
    x = _maybe_put_sharded(jax, x)

    xp, mels = jax.jit(wrapped)(x)

    xp_np, mels_np, x_np = map(jax.device_get, (xp, mels, x))

    assert xp_np.shape == (batch, max_conn, n_sites)
    assert mels_np.shape == (batch, max_conn)

    expected_xp = np.repeat(np.asarray(x_np)[:, None, :], max_conn, axis=1)
    np.testing.assert_array_equal(xp_np, expected_xp)

    assert mels_np.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        mels_np, np.zeros((batch, max_conn), dtype=np.float32)
    )

    assert xp_np.dtype == np.asarray(x_np).dtype


def test_replicate_sharding_works_without_jit_too(smod, jax, jnp):
    ndev = max(1, jax.device_count())
    batch = ndev
    n_sites = 3

    f_py = DummyConnPadded(max_conn=2, mel_dtype=np.complex64)
    wrapped = smod.replicate_sharding(f_py)

    x = jnp.ones((batch, n_sites), dtype=jnp.float32)
    x = _maybe_put_sharded(jax, x)

    xp, mels = wrapped(x)
    xp_np, mels_np = map(jax.device_get, (xp, mels))

    assert xp_np.shape == (batch, 2, n_sites)
    assert mels_np.shape == (batch, 2)
    assert mels_np.dtype == np.dtype(np.complex64)


def test_missing_required_attributes_fails_fast(smod, jax, jnp):
    def bad_callback(x):
        return x, x

    wrapped = smod.replicate_sharding(bad_callback)

    x = jnp.zeros((1, 2), dtype=jnp.float32)

    with pytest.raises(AttributeError):
        _ = wrapped(x)


def test_input_rank_mismatch_raises(smod, jax, jnp):
    f_py = DummyConnPadded(max_conn=2, mel_dtype=np.float32)
    wrapped = smod.replicate_sharding(f_py)

    x = jnp.arange(5, dtype=jnp.int32)

    with pytest.raises(Exception):
        _ = wrapped(x)
