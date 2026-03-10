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
import os
import sys
import pytest


def _clear_related_env(monkeypatch):
    keys = list(os.environ.keys())
    for k in keys:
        ku = k.upper()
        if (
            ku.startswith("NQX_")
            or ku.startswith("NETKET_")
            or ku.startswith("MPI4JAX_")
        ):
            monkeypatch.delenv(k, raising=False)

    for k in [
        "JAX_PLATFORM_NAME",
        "OMP_NUM_THREADS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    ]:
        monkeypatch.delenv(k, raising=False)


def _import_config(monkeypatch, env=None):

    _clear_related_env(monkeypatch)
    env = env or {}
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))

    sys.modules.pop("neuralqx.configs", None)
    return importlib.import_module("neuralqx.configs")


@pytest.fixture
def cfgmod_factory(monkeypatch):
    def _factory(env=None):
        return _import_config(monkeypatch, env=env)

    return _factory


def test_new_error_content_includes_help_link(cfgmod_factory):
    m = cfgmod_factory()
    msg = m.new_error_content("Hello\n  World")
    assert "Hello" in msg
    assert "neuralqx.readthedocs.io" in msg
    assert "guides/errors.html" in msg
    assert (
        "============================================================================="
        in msg
    )


def test_configerror_wraps_message_with_help_link(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigError) as ex:
        raise m.ConfigError("Bad config.")
    s = str(ex.value)
    assert "Bad config." in s
    assert "guides/errors.html" in s


def test_readonlydict_rejects_set_and_del(cfgmod_factory):
    m = cfgmod_factory()
    d = m.ReadOnlyDict({"A": 1})
    with pytest.raises(TypeError, match="read-only"):
        d["A"] = 2
    with pytest.raises(TypeError, match="read-only"):
        del d["A"]


def test_configschema_register_uppercases_and_stores_configitem(cfgmod_factory):
    m = cfgmod_factory()
    schema = m.ConfigSchema()

    called = {}

    def cb(old, new):
        called["old"] = old
        called["new"] = new

    schema.register(
        env_name="my_key",
        env_val_type=int,
        env_default_val=7,
        env_desc="desc",
        runtime=True,
        callback=cb,
    )

    sch = schema.get_schema()
    assert "MY_KEY" in sch
    item = sch["MY_KEY"]
    assert item.default == 7
    assert item.mutable is True
    assert item.parser is int
    assert item.description == "desc"
    assert item.callback is cb


@pytest.mark.parametrize(
    "s",
    ["true", "True", "1", "yes", "on", "t", "y", "  TRUE  "],
)
def test_parse_bool_true(cfgmod_factory, s):
    m = cfgmod_factory()
    assert m.ConfigManager._parse_bool(s) is True


@pytest.mark.parametrize(
    "s",
    ["false", "False", "0", "no", "off", "f", "n", "  0  "],
)
def test_parse_bool_false(cfgmod_factory, s):
    m = cfgmod_factory()
    assert m.ConfigManager._parse_bool(s) is False


def test_parse_bool_invalid_raises(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(ValueError):
        m.ConfigManager._parse_bool("maybe")


def test_get_env_case_insensitive_returns_value_when_only_lowercase_key_exists(
    cfgmod_factory, monkeypatch
):
    m = cfgmod_factory()

    monkeypatch.delenv("NQX_VERBOSE", raising=False)

    monkeypatch.setenv("nqx_verbose", "true")
    assert m.ConfigManager._get_env_case_insensitive("NQX_VERBOSE") == "true"


def test_get_env_case_insensitive_prefers_first_inserted_when_multiple_case_variants_exist(
    cfgmod_factory, monkeypatch
):
    m = cfgmod_factory()

    monkeypatch.setenv("nqx_verbose", "true")

    assert (
        m.ConfigManager._get_env_case_insensitive("NQX_VERBOSE")
        == os.environ["NQX_VERBOSE"]
    )
    assert m.ConfigManager._get_env_case_insensitive("NQX_VERBOSE") == "True"


def test_configmanager_is_singleton_within_module(cfgmod_factory):
    m = cfgmod_factory()
    a = m.ConfigManager()
    b = m.ConfigManager()
    assert a is b
    assert m.cfg is a


def test_initialization_reads_env_case_insensitive_and_normalizes_env_key(
    cfgmod_factory,
):
    m = cfgmod_factory(env={"nqx_debug": "true"})
    assert m.cfg.get("DEBUG") is True
    assert os.environ["NQX_DEBUG"] == "True"


def test_defaults_are_set_in_os_environ_on_import(cfgmod_factory):
    m = cfgmod_factory()
    assert "NQX_DEBUG" in os.environ
    assert "NQX_VERBOSE" in os.environ


def test_get_unknown_key_raises_configerror(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigError, match="is not defined"):
        m.cfg.get("DOES_NOT_EXIST")


def test_set_unknown_key_raises_configerror(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigError, match="is not defined"):
        m.cfg.set("DOES_NOT_EXIST", 1)


def test_set_immutable_key_raises_configerror(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigError, match="not runtime editable"):
        m.cfg.set("LOG_LEVEL", "DEBUG")


def test_set_parsing_error_is_wrapped_as_configerror(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigError) as ex:
        m.cfg.set("DEBUG", "not-a-bool")
    assert "Error parsing value" in str(ex.value) or "Cannot parse boolean" in str(
        ex.value
    )


def test_register_callback_and_set_invokes_callback(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    seen = []

    def cb(old, new):
        seen.append((old, new))

    m.cfg.register_callback("DEBUG", cb)
    m.cfg.set("DEBUG", "1")
    assert m.cfg.get("DEBUG") is True
    assert seen[-1] == (False, True)


def test_get_auto_syncs_mutable_config_when_env_changed_and_triggers_callback(
    cfgmod_factory, monkeypatch
):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    cfg = m.cfg

    seen = []

    def cb(old, new):
        seen.append((old, new))

    cfg.register_callback("DEBUG", cb)

    monkeypatch.setenv("NQX_DEBUG", "1")

    assert cfg.get("DEBUG") is True

    assert os.environ["NQX_DEBUG"] == "True"
    assert seen[-1] == (False, True)


def test_to_dict_returns_copy(cfgmod_factory):
    m = cfgmod_factory()
    cfg = m.cfg

    d = cfg.to_dict()
    assert isinstance(d, dict)
    d["DEBUG"] = "corrupt"

    assert cfg.get("DEBUG") in (True, False)


def test_get_env_type_formats_callable_and_class_parsers(cfgmod_factory):
    m = cfgmod_factory()
    cfg = m.cfg

    assert cfg._get_env_type(cfg._parse_bool) == "bool"

    assert cfg._get_env_type(int) == "int"

    assert cfg._get_env_type(str) == "str"


def test_print_config_emits_table(cfgmod_factory, capsys):
    m = cfgmod_factory()
    m.cfg.print_config()
    out = capsys.readouterr().out

    assert "Variable" in out

    assert "Description" in out
    assert "NQX_DEBUG" in out


def test_warn_unused_prints_for_unknown_nqx_variables(
    cfgmod_factory, capsys, monkeypatch
):
    _ = cfgmod_factory(env={"NQX_SOMETHING_UNUSED": "abc"})
    out = capsys.readouterr().out
    assert "not\nrecognized by neuraLQX and will be ignored" in out
    assert "NQX_SOMETHING_UNUSED" in out
    assert "abc" in out


def test_statics_is_readonly_and_contains_expected_keys(cfgmod_factory):
    m = cfgmod_factory()
    s = m.cfg.statics
    assert isinstance(s, m.ReadOnlyDict)
    for k in [
        "Errors Directory",
        "Cache Directory",
        "Leach Directory",
    ]:
        assert k in s
        assert isinstance(s[k], str)

    with pytest.raises(TypeError, match="read-only"):
        s["X"] = "y"


def test_get_static_missing_key_raises(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(KeyError, match="does not exist"):
        m.cfg.get_static("NOPE")


def test_sync_external_env_sets_expected_defaults_for_cuda(cfgmod_factory):

    assert os.environ.get("JAX_PLATFORMS") == ""
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"
    assert os.environ.get("NETKET_EXPERIMENTAL_SHARDING") == "True"
