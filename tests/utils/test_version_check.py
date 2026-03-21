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


import types
import pytest


@pytest.fixture
def vmod():
    import neuralqx.utils.module.version_check as vmod_

    return vmod_


def test_version_strict_dev_ordering(vmod):
    assert vmod.Version("1.1.0.dev1") < vmod.Version("1.1.0.dev2")
    assert vmod.Version("1.1.0.dev2") < vmod.Version("1.1.0")


def test_version_compares_with_strings_and_sequences(vmod):
    assert vmod.Version("1.2.9") > "1.2.8"
    assert vmod.Version("1.2.9") > (1, 2, 8)
    assert vmod.Version("1.2.9") >= [1, 2, 9]


def test_version_invalid_string_raises(vmod):
    with pytest.raises(vmod.InvalidVersion):
        vmod.Version("nightly-local-build")


def test_neuralqx_version_is_string_like(vmod):
    v = vmod.NeuralqxVersion("1.2.3")
    assert isinstance(v, str)
    assert str(v) == "1.2.3"
    assert v.startswith("1.2")


def test_neuralqx_version_uses_semantic_comparison(vmod):
    v = vmod.NeuralqxVersion("1.1.0.dev2")
    assert v > "1.1.0.dev1"
    assert v < "1.1.0"
    assert v > (1, 0, 9)


def test_neuralqx_version_invalid_comparison_raises(vmod):
    v = vmod.NeuralqxVersion("1.2.0")
    with pytest.raises(vmod.InvalidVersion):
        _ = v > "parrot"


def test_version_metadata(vmod):
    v = vmod.Version("2.4.1")
    assert v.source == "2.4.1"
    assert v.canonical == "2.4.1"
    assert v.major == 2
    assert v.minor == 4
    assert v.patch == 1
    assert v.release == (2, 4, 1)
    assert v.as_tuple(width=3) == (2, 4, 1)


def test_normalize_version_strict(vmod):
    assert vmod.normalize_version("1.2.3.dev4", strict=True) == (1, 2, 3)


def test_normalize_version_non_strict_invalid_to_zeroes(vmod):
    assert vmod.normalize_version("parrot", strict=False) == (0, 0, 0)


def test_normalize_version_strict_invalid_raises(vmod):
    with pytest.raises(vmod.InvalidVersion):
        vmod.normalize_version("parrot", strict=True)


def test_get_module_version_from_module_object(vmod):
    m = types.SimpleNamespace(__version__="3.17.1")
    assert vmod.get_module_version(m) == vmod.Version("3.17.1")


def test_get_module_version_missing___version___defaults_to_zero(vmod):
    m = types.SimpleNamespace()
    assert vmod.get_module_version(m) == vmod.Version("0.0.0")


def test_get_module_version_invalid_strict_behavior(vmod):
    m = types.SimpleNamespace(__version__="nightly")
    with pytest.raises(vmod.InvalidVersion):
        vmod.get_module_version(m, strict=True)

    assert vmod.get_module_version(m, strict=False) is None


def test_get_module_neuralqx_version(vmod):
    m = types.SimpleNamespace(__version__="9.8.7.dev0")
    out = vmod.get_module_neuralqx_version(m)
    assert isinstance(out, vmod.NeuralqxVersion)
    assert out == "9.8.7.dev0"
    assert out < "9.8.7"


def test_get_module_version_string_from_module_object(vmod):
    m = types.SimpleNamespace(__version__="9.8.7.dev0")
    assert vmod.get_module_version_string(m) == "9.8.7.dev0"


def test_get_module_version_string_missing___version___returns_unknown(vmod):
    m = types.SimpleNamespace()
    assert vmod.get_module_version_string(m) == "unknown"


def test_get_module_version_string_imports_when_given_string(vmod, monkeypatch):
    dummy = types.SimpleNamespace(__version__="0.1.0+cuda")

    def fake_import(name):
        assert name == "dummy_pkg"
        return dummy

    monkeypatch.setattr(
        "neuralqx.utils.module.version.module_version.import_module",
        fake_import,
    )
    assert vmod.get_module_version_string("dummy_pkg") == "0.1.0+cuda"


def test_get_module_version_string_propagates_import_error(vmod, monkeypatch):
    def fake_import(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "neuralqx.utils.module.version.module_version.import_module",
        fake_import,
    )
    with pytest.raises(ModuleNotFoundError):
        vmod.get_module_version_string("no_such_pkg")


def test_top_level___version___is_str_compatible_and_semantic(vmod):
    import neuralqx as nqx
    from neuralqx import _version as raw_version

    assert isinstance(nqx.__version__, str)
    assert isinstance(nqx.__version__, vmod.NeuralqxVersion)
    assert str(nqx.__version__) == raw_version.__version__
    assert nqx.__version__ == nqx.version


def test_top_level_version_info_exposes_strict_version(vmod):
    import neuralqx as nqx

    assert isinstance(nqx.version_info, vmod.Version)
    assert nqx.version_info == vmod.Version(str(nqx.__version__))
