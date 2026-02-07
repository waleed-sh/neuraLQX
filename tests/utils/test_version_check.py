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


@pytest.mark.parametrize(
    "s, expected",
    [
        ("0", (0, 0, 0)),
        ("1", (1, 0, 0)),
        ("1.2", (1, 2, 0)),
        ("1.2.3", (1, 2, 3)),
        ("10.20.30", (10, 20, 30)),
        ("1.2.3rc1", (1, 2, 3)),
        ("1.2.3.dev4", (1, 2, 3)),
        ("1.2.3.post1", (1, 2, 3)),
        ("1.2.3+cuda11", (1, 2, 3)),
        ("v1.2.3", (1, 2, 3)),
        ("version 1.2.3", (1, 2, 3)),
        ("1.2rc1", (1, 2, 0)),
        ("1rc1", (1, 0, 0)),
        ("01.002.0003", (1, 2, 3)),
        ("junk 2.3.4 then 9.9.9", (2, 3, 4)),
    ],
)
def test_normalize_version_parses_common_forms(vmod, s, expected):
    assert vmod.normalize_version(s) == expected


@pytest.mark.parametrize(
    "s, expected",
    [
        ("rc1", (1, 0, 0)),
        ("dev1", (1, 0, 0)),
        ("post2", (2, 0, 0)),
        ("alpha9", (9, 0, 0)),
    ],
)
def test_normalize_version_prefix_suffix_digit_grab(vmod, s, expected):
    assert vmod.normalize_version(s) == expected


def test_normalize_version_handles_digits_not_as_version(vmod):
    assert vmod.normalize_version("built 2025-12-21") == (2025, 0, 0)


def test_get_module_version_from_module_object(vmod):
    m = types.SimpleNamespace(__version__="3.17.1")
    assert vmod.get_module_version(m) == (3, 17, 1)


def test_get_module_version_missing___version___defaults_to_0(vmod):
    m = types.SimpleNamespace()
    assert vmod.get_module_version(m) == (0, 0, 0)


def test_get_module_version_weird___version___string(vmod):
    m = types.SimpleNamespace(__version__="3.17.1rc2+local")
    assert vmod.get_module_version(m) == (3, 17, 1)


def test_get_module_version_imports_when_given_string(vmod, monkeypatch):
    dummy = types.SimpleNamespace(__version__="1.2.3")

    def fake_import(name):
        assert name == "dummy_pkg"
        return dummy

    monkeypatch.setattr(vmod.importlib, "import_module", fake_import, raising=True)
    assert vmod.get_module_version("dummy_pkg") == (1, 2, 3)


def test_get_module_version_propagates_import_error(vmod, monkeypatch):
    def fake_import(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(vmod.importlib, "import_module", fake_import, raising=True)
    with pytest.raises(ModuleNotFoundError):
        vmod.get_module_version("no_such_pkg")


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

    monkeypatch.setattr(vmod.importlib, "import_module", fake_import, raising=True)
    assert vmod.get_module_version_string("dummy_pkg") == "0.1.0+cuda"


def test_get_module_version_string_propagates_import_error(vmod, monkeypatch):
    def fake_import(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(vmod.importlib, "import_module", fake_import, raising=True)
    with pytest.raises(ModuleNotFoundError):
        vmod.get_module_version_string("no_such_pkg")
