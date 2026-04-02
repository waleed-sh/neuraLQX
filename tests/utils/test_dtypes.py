# Copyright (c) 2026 The neuraLQX Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

import neuralqx.utils.dtypes as dtypes
from neuralqx.utils.dtypes import names
from neuralqx.utils.dtypes.policy import DTypePolicy

pytestmark = pytest.mark.requires_jax


EXPECTED_PUBLIC_API = {
    "DTypePolicy",
    "array_complex",
    "array_index",
    "array_real",
    "dtype_name_map",
    "full_complex",
    "full_real",
    "get_dtype_policy",
    "jax_complex_dtype",
    "jax_index_dtype",
    "jax_real_dtype",
    "names",
    "ones_complex",
    "ones_real",
    "zeros_complex",
    "zeros_index",
    "zeros_real",
}


def _cfg(
    *,
    real: str = "float32",
    complex: str = "complex64",
    index: str = "int32",
    enable_x64: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        DTYPE_REAL=real,
        DTYPE_COMPLEX=complex,
        DTYPE_INDEX=index,
        ENABLE_X64=enable_x64,
    )


def _clear_related_env(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = list(os.environ.keys())
    for key in keys:
        key_upper = key.upper()
        if (
            key_upper.startswith("NQX_")
            or key_upper.startswith("NETKET_")
            or key_upper.startswith("MPI4JAX_")
        ):
            monkeypatch.delenv(key, raising=False)

    for key in [
        "JAX_ENABLE_X64",
        "JAX_PLATFORM_NAME",
        "OMP_NUM_THREADS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    ]:
        monkeypatch.delenv(key, raising=False)


class TestNamesModule:
    def test_declares_expected_dtype_name_sets(self) -> None:
        assert names.REAL_DTYPE_NAMES == ("float16", "bfloat16", "float32", "float64")
        assert names.COMPLEX_DTYPE_NAMES == ("complex64", "complex128")
        assert names.INDEX_DTYPE_NAMES == ("int32", "int64")
        assert len(set(names.REAL_DTYPE_NAMES)) == len(names.REAL_DTYPE_NAMES)
        assert len(set(names.COMPLEX_DTYPE_NAMES)) == len(names.COMPLEX_DTYPE_NAMES)
        assert len(set(names.INDEX_DTYPE_NAMES)) == len(names.INDEX_DTYPE_NAMES)


class TestDTypePolicy:
    def test_normalizes_whitespace_is_frozen_and_serializes(self) -> None:
        policy = DTypePolicy(
            real=" float32 ",
            complex=" complex64 ",
            index=" int32 ",
        )

        assert policy.real == "float32"
        assert policy.complex == "complex64"
        assert policy.index == "int32"
        assert policy.as_dict() == {
            "real": "float32",
            "complex": "complex64",
            "index": "int32",
        }

        with pytest.raises(AttributeError):
            policy.real = "float64"

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("real", "", "real must be a non-empty string"),
            ("real", "float128", "real must be one of"),
            ("complex", "complex256", "complex must be one of"),
            ("index", "uint32", "index must be one of"),
        ],
    )
    def test_rejects_invalid_declared_fields(
        self,
        field: str,
        value: str,
        match: str,
    ) -> None:
        kwargs = {
            "real": "float32",
            "complex": "complex64",
            "index": "int32",
        }
        kwargs[field] = value

        with pytest.raises(ValueError, match=match):
            DTypePolicy(**kwargs)

    def test_from_config_uses_explicit_config_without_promotion(self) -> None:
        policy = DTypePolicy.from_config(
            _cfg(
                real="bfloat16",
                complex="complex128",
                index="int64",
                enable_x64=False,
            )
        )

        assert policy == DTypePolicy(
            real="bfloat16",
            complex="complex128",
            index="int64",
        )

    def test_from_config_promotes_x64_compatible_names(self) -> None:
        policy = DTypePolicy.from_config(
            _cfg(
                real="float32",
                complex="complex64",
                index="int32",
                enable_x64=True,
            )
        )

        assert policy == DTypePolicy(
            real="float64",
            complex="complex128",
            index="int64",
        )

    def test_from_config_leaves_non_promotable_names_unchanged(self) -> None:
        policy = DTypePolicy.from_config(
            _cfg(
                real="bfloat16",
                complex="complex128",
                index="int64",
                enable_x64=True,
            )
        )

        assert policy == DTypePolicy(
            real="bfloat16",
            complex="complex128",
            index="int64",
        )

    def test_from_config_uses_global_cfg_when_omitted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stub_cfg_module = types.ModuleType("neuralqx.configs")
        stub_cfg_module.cfg = _cfg(
            real="float16",
            complex="complex64",
            index="int32",
            enable_x64=False,
        )

        monkeypatch.setitem(sys.modules, "neuralqx.configs", stub_cfg_module)

        policy = DTypePolicy.from_config()

        assert policy == DTypePolicy(
            real="float16",
            complex="complex64",
            index="int32",
        )


class TestRuntimeAccessors:
    def test_dtype_name_map_returns_plain_detached_mapping(self) -> None:
        cfg = _cfg(
            real="float32",
            complex="complex64",
            index="int32",
        )

        mapping = dtypes.dtype_name_map(cfg)
        mapping["real"] = "mutated"

        assert dtypes.dtype_name_map(cfg) == {
            "real": "float32",
            "complex": "complex64",
            "index": "int32",
        }

    def test_get_dtype_policy_delegates_to_policy_factory(self) -> None:
        cfg = _cfg(
            real="float32",
            complex="complex64",
            index="int32",
            enable_x64=True,
        )

        assert dtypes.get_dtype_policy(cfg) == DTypePolicy.from_config(cfg)

    def test_jax_dtype_accessors_reflect_effective_policy(self) -> None:
        cfg = _cfg(
            real="float32",
            complex="complex64",
            index="int32",
            enable_x64=True,
        )

        assert dtypes.jax_real_dtype(cfg) == jnp.dtype("float64")
        assert dtypes.jax_complex_dtype(cfg) == jnp.dtype("complex128")
        assert dtypes.jax_index_dtype(cfg) == jnp.dtype("int64")


class TestArrayFactories:
    def test_real_array_factories_apply_real_dtype_and_values(self) -> None:
        cfg = _cfg(real="float32")

        zeros = dtypes.zeros_real((2, 3), config_manager=cfg)
        ones = dtypes.ones_real(4, config_manager=cfg)
        full = dtypes.full_real((2,), 3.5, config_manager=cfg)
        array = dtypes.array_real([1, 2, 3], config_manager=cfg)

        assert zeros.shape == (2, 3)
        assert zeros.dtype == jnp.dtype("float32")
        assert jnp.all(zeros == 0)

        assert ones.shape == (4,)
        assert ones.dtype == jnp.dtype("float32")
        assert jnp.all(ones == 1)

        assert full.dtype == jnp.dtype("float32")
        assert jnp.all(full == jnp.array([3.5, 3.5], dtype=jnp.float32))

        assert array.dtype == jnp.dtype("float32")
        assert jnp.all(array == jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))

    def test_complex_array_factories_apply_complex_dtype_and_values(self) -> None:
        cfg = _cfg(complex="complex64")

        zeros = dtypes.zeros_complex((2,), config_manager=cfg)
        ones = dtypes.ones_complex((3,), config_manager=cfg)
        full = dtypes.full_complex((2,), 1 + 2j, config_manager=cfg)
        array = dtypes.array_complex([1, 2j], config_manager=cfg)

        assert zeros.dtype == jnp.dtype("complex64")
        assert jnp.all(zeros == 0)

        assert ones.dtype == jnp.dtype("complex64")
        assert jnp.all(ones == 1)

        assert full.dtype == jnp.dtype("complex64")
        assert jnp.all(full == jnp.array([1 + 2j, 1 + 2j], dtype=jnp.complex64))

        assert array.dtype == jnp.dtype("complex64")
        assert jnp.all(array == jnp.array([1 + 0j, 0 + 2j], dtype=jnp.complex64))

    def test_index_array_factories_apply_index_dtype(self) -> None:
        cfg = _cfg(index="int32")

        zeros = dtypes.zeros_index((2, 2), config_manager=cfg)
        array = dtypes.array_index([1, 2, 3], config_manager=cfg)

        assert zeros.dtype == jnp.dtype("int32")
        assert zeros.shape == (2, 2)
        assert jnp.all(zeros == 0)

        assert array.dtype == jnp.dtype("int32")
        assert jnp.array_equal(array, jnp.array([1, 2, 3], dtype=jnp.int32))


class TestPublicApi:
    def test_reexports_expected_public_surface(self) -> None:
        assert set(dtypes.__all__) == EXPECTED_PUBLIC_API
        assert dtypes.names is names
        assert dtypes.DTypePolicy is DTypePolicy
