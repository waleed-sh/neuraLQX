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
import threading

import pytest


def _clear_related_env(monkeypatch):
    """Removes NQX_*, NETKET_*, MPI4JAX_* and a few JAX env vars."""
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
        "JAX_ENABLE_X64",
    ]:
        monkeypatch.delenv(k, raising=False)


def _import_config(monkeypatch, env=None):
    """Forces a fresh module import with an optional environment overlay."""
    _clear_related_env(monkeypatch)
    env = env or {}
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))

    sys.modules.pop("neuralqx.configs", None)
    return importlib.import_module("neuralqx.configs")


@pytest.fixture
def cfgmod_factory(monkeypatch):
    """Factory fixture that returns a fresh neuralqx.configs module."""

    def _factory(env=None):
        return _import_config(monkeypatch, env=env)

    return _factory


def _bare_manager(m):
    """
    Creates a ConfigManager instance that bypasses the singleton and has no
    registered options. Used for unit-testing the ConfigManager machinery in
    isolation from the default option set.
    """
    mgr = object.__new__(m.ConfigManager)
    object.__setattr__(mgr, "_options", {})
    object.__setattr__(mgr, "_hooks", {})
    object.__setattr__(mgr, "_global_hooks", [])
    object.__setattr__(mgr, "_runtime_locked", False)
    object.__setattr__(mgr, "_thread_local_store", threading.local())
    object.__setattr__(mgr, "_lock", threading.RLock())
    object.__setattr__(mgr, "_initialized", True)
    mgr._register_statics()
    return mgr


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


def test_new_error_content_dedents_message(cfgmod_factory):
    m = cfgmod_factory()
    msg = m.new_error_content("  indented line")
    # dedent should remove leading whitespace
    assert "indented line" in msg


def test_configerror_wraps_message_with_help_link(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigError) as ex:
        raise m.ConfigError("Bad config.")
    s = str(ex.value)
    assert "Bad config." in s
    assert "guides/errors.html" in s


def test_unknown_option_error_is_config_error(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigError):
        raise m.UnknownOptionError("nope")


def test_config_validation_error_is_config_error(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigError):
        raise m.ConfigValidationError("bad value")


def test_readonlydict_rejects_set_and_del(cfgmod_factory):
    m = cfgmod_factory()
    d = m.ReadOnlyDict({"A": 1})
    with pytest.raises(TypeError, match="read-only"):
        d["A"] = 2
    with pytest.raises(TypeError, match="read-only"):
        del d["A"]


def test_readonlydict_allows_read(cfgmod_factory):
    m = cfgmod_factory()
    d = m.ReadOnlyDict({"X": 42})
    assert d["X"] == 42
    assert "X" in d


def test_config_mutability_values(cfgmod_factory):
    m = cfgmod_factory()
    assert m.ConfigMutability.IMMUTABLE.value == "immutable"
    assert m.ConfigMutability.STARTUP.value == "startup"
    assert m.ConfigMutability.RUNTIME.value == "runtime"


def test_config_mutability_is_str_enum(cfgmod_factory):
    m = cfgmod_factory()
    assert isinstance(m.ConfigMutability.RUNTIME, str)
    assert m.ConfigMutability.RUNTIME == "runtime"


def test_config_source_values(cfgmod_factory):
    m = cfgmod_factory()
    assert m.ConfigSource.DEFAULT.value == "default"
    assert m.ConfigSource.ENV_DEFAULT.value == "env_default"
    assert m.ConfigSource.ENV_FORCE.value == "env_force"
    assert m.ConfigSource.USER.value == "user"
    assert m.ConfigSource.PATCH.value == "patch"
    assert m.ConfigSource.THREAD_LOCAL.value == "thread_local"
    assert m.ConfigSource.RESET.value == "reset"


def test_config_source_is_str_enum(cfgmod_factory):
    m = cfgmod_factory()
    assert isinstance(m.ConfigSource.USER, str)
    assert m.ConfigSource.USER == "user"


def test_config_mutation_fields(cfgmod_factory):
    m = cfgmod_factory()
    event = m.ConfigMutation(
        name="FOO",
        old_value=1,
        new_value=2,
        source=m.ConfigSource.USER,
        thread_local=False,
        mutability=m.ConfigMutability.RUNTIME,
    )
    assert event.name == "FOO"
    assert event.old_value == 1
    assert event.new_value == 2
    assert event.source is m.ConfigSource.USER
    assert event.thread_local is False
    assert event.mutability is m.ConfigMutability.RUNTIME


def test_config_mutation_is_frozen(cfgmod_factory):
    m = cfgmod_factory()
    event = m.ConfigMutation(
        name="X",
        old_value=0,
        new_value=1,
        source=m.ConfigSource.USER,
        thread_local=False,
        mutability=m.ConfigMutability.RUNTIME,
    )
    with pytest.raises((AttributeError, TypeError)):
        event.name = "Y"  # type: ignore[misc]


def test_config_option_parse_uses_parser(cfgmod_factory):
    m = cfgmod_factory()
    opt = m.ConfigOption(
        name="OPT",
        default=0,
        doc="test",
        value_type=int,
        parser=m.parse_int,
    )
    assert opt.parse("7") == 7


def test_config_option_parse_raises_on_wrong_type(cfgmod_factory):
    m = cfgmod_factory()
    # parse_int returns int, value_type=str will reject it
    opt = m.ConfigOption(
        name="OPT",
        default="hello",
        doc="test",
        value_type=str,
        parser=m.parse_int,
    )
    with pytest.raises(m.ConfigValidationError):
        opt.parse("5")  # parse_int("5") -> 5, not str


def test_config_option_parse_runs_validator(cfgmod_factory):
    m = cfgmod_factory()
    opt = m.ConfigOption(
        name="OPT",
        default=1,
        doc="test",
        value_type=int,
        parser=m.parse_int,
        validator=m.positive_int_validator,
    )
    with pytest.raises(m.ConfigValidationError):
        opt.parse("-1")


@pytest.mark.parametrize(
    "s",
    ["true", "True", "1", "yes", "on", "t", "y", "  TRUE  "],
)
def test_parse_bool_true(cfgmod_factory, s):
    m = cfgmod_factory()
    assert m.parse_bool(s) is True


@pytest.mark.parametrize(
    "s",
    ["false", "False", "0", "no", "off", "f", "n", "  0  "],
)
def test_parse_bool_false(cfgmod_factory, s):
    m = cfgmod_factory()
    assert m.parse_bool(s) is False


def test_parse_bool_accepts_bool_passthrough(cfgmod_factory):
    m = cfgmod_factory()
    assert m.parse_bool(True) is True
    assert m.parse_bool(False) is False


def test_parse_bool_accepts_int_zero_one(cfgmod_factory):
    m = cfgmod_factory()
    assert m.parse_bool(1) is True
    assert m.parse_bool(0) is False


def test_parse_bool_invalid_raises_config_validation_error(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigValidationError):
        m.parse_bool("maybe")


def test_parse_int_valid(cfgmod_factory):
    m = cfgmod_factory()
    assert m.parse_int("42") == 42
    assert m.parse_int(7) == 7


def test_parse_int_rejects_bool(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigValidationError):
        m.parse_int(True)


def test_parse_int_invalid_raises(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigValidationError):
        m.parse_int("not_a_number")


def test_parse_float_valid(cfgmod_factory):
    m = cfgmod_factory()
    assert m.parse_float("3.14") == pytest.approx(3.14)
    assert m.parse_float(2) == pytest.approx(2.0)


def test_parse_float_rejects_bool(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigValidationError):
        m.parse_float(True)


def test_parse_float_invalid_raises(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigValidationError):
        m.parse_float("xyz")


def test_parse_optional_int_none(cfgmod_factory):
    m = cfgmod_factory()
    assert m.parse_optional_int(None) is None
    assert m.parse_optional_int("none") is None
    assert m.parse_optional_int("null") is None
    assert m.parse_optional_int("") is None


def test_parse_optional_int_valid(cfgmod_factory):
    m = cfgmod_factory()
    assert m.parse_optional_int("5") == 5


def test_parse_optional_string_none(cfgmod_factory):
    m = cfgmod_factory()
    assert m.parse_optional_string(None) is None
    assert m.parse_optional_string("none") is None
    assert m.parse_optional_string("") is None


def test_parse_optional_string_valid(cfgmod_factory):
    m = cfgmod_factory()
    assert m.parse_optional_string("  hello  ") == "hello"


def test_parse_optional_string_rejects_non_string(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigValidationError):
        m.parse_optional_string(123)


def test_parse_csv_tuple_string(cfgmod_factory):
    m = cfgmod_factory()
    result = m.parse_csv_tuple("a, b, c")
    assert result == ("a", "b", "c")


def test_parse_csv_tuple_filters_empty(cfgmod_factory):
    m = cfgmod_factory()
    assert m.parse_csv_tuple("a,,b") == ("a", "b")


def test_parse_csv_tuple_none(cfgmod_factory):
    m = cfgmod_factory()
    assert m.parse_csv_tuple(None) == ()


def test_parse_csv_tuple_sequence(cfgmod_factory):
    m = cfgmod_factory()
    assert m.parse_csv_tuple(["x", "y"]) == ("x", "y")


def test_parse_csv_tuple_invalid_sequence_entries(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigValidationError):
        m.parse_csv_tuple([1, 2])


def test_positive_int_validator_passes(cfgmod_factory):
    m = cfgmod_factory()
    m.positive_int_validator(1)  # no exception


def test_positive_int_validator_rejects_zero_and_negative(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigValidationError):
        m.positive_int_validator(0)
    with pytest.raises(m.ConfigValidationError):
        m.positive_int_validator(-5)


def test_non_negative_int_validator_passes(cfgmod_factory):
    m = cfgmod_factory()
    m.non_negative_int_validator(0)
    m.non_negative_int_validator(10)


def test_non_negative_int_validator_rejects_negative(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigValidationError):
        m.non_negative_int_validator(-1)


def test_positive_float_validator_passes(cfgmod_factory):
    m = cfgmod_factory()
    m.positive_float_validator(0.1)


def test_positive_float_validator_rejects_zero_and_negative(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigValidationError):
        m.positive_float_validator(0.0)
    with pytest.raises(m.ConfigValidationError):
        m.positive_float_validator(-1.5)


def test_configmanager_is_singleton_within_module(cfgmod_factory):
    m = cfgmod_factory()
    a = m.ConfigManager()
    b = m.ConfigManager()
    assert a is b
    assert m.cfg is a


def test_register_option_stores_option_and_writes_env(cfgmod_factory, monkeypatch):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    opt = m.ConfigOption(
        name="MY_INT",
        default=42,
        doc="a test int",
        value_type=int,
        parser=m.parse_int,
        env_default=("NQX_MY_INT",),
    )
    mgr.register_option(opt)
    assert "MY_INT" in mgr.list_options()
    assert os.environ.get("NQX_MY_INT") == "42"


def test_register_option_duplicate_raises(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    opt = m.ConfigOption(
        name="DUP", default=0, doc="dup", value_type=int, parser=m.parse_int
    )
    mgr.register_option(opt)
    with pytest.raises(m.ConfigValidationError, match="already registered"):
        mgr.register_option(opt)


def test_register_option_respects_env_default(cfgmod_factory, monkeypatch):
    m = cfgmod_factory()
    monkeypatch.setenv("NQX_MY_BOOL", "true")
    mgr = _bare_manager(m)
    mgr.define_bool("MY_BOOL", default=False, doc="test", env_default="NQX_MY_BOOL")
    assert mgr.get("MY_BOOL") is True


def test_register_option_respects_env_force(cfgmod_factory, monkeypatch):
    m = cfgmod_factory()
    monkeypatch.setenv("NQX_FORCE_FLAG", "1")
    mgr = _bare_manager(m)
    mgr.define_int(
        "FORCED",
        default=0,
        doc="test",
        env_default="NQX_FORCED",
        env_force="NQX_FORCE_FLAG",
    )
    assert mgr.get("FORCED") == 1
    with pytest.raises(m.ConfigError):
        mgr.set("FORCED", 99)


def test_define_bool(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    mgr.define_bool("FLAG", default=True, doc="a flag")
    assert mgr.get("FLAG") is True


def test_define_int(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    mgr.define_int(
        "COUNT", default=5, doc="a count", validator=m.positive_int_validator
    )
    assert mgr.get("COUNT") == 5
    with pytest.raises(m.ConfigValidationError):
        mgr.set("COUNT", -1)


def test_define_float(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    mgr.define_float("RATE", default=0.5, doc="rate")
    assert mgr.get("RATE") == pytest.approx(0.5)


def test_define_string(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    mgr.define_string("LABEL", default="default_label", doc="label")
    assert mgr.get("LABEL") == "default_label"


def test_define_optional_string_none_default(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    mgr.define_optional_string("DIR", default=None, doc="dir")
    assert mgr.get("DIR") is None


def test_define_optional_int_none_default(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    mgr.define_optional_int("PORT", default=None, doc="port")
    assert mgr.get("PORT") is None
    mgr.set("PORT", "8080")
    assert mgr.get("PORT") == 8080


def test_define_enum_normalises_to_lowercase(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    mgr.define_enum(
        "LEVEL", default="info", values=("debug", "info", "warning"), doc="lvl"
    )
    assert mgr.get("LEVEL") == "info"
    mgr.set("LEVEL", "DEBUG")
    assert mgr.get("LEVEL") == "debug"


def test_define_enum_case_sensitive(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    mgr.define_enum(
        "MODE",
        default="ON",
        values=("ON", "OFF"),
        doc="mode",
        case_sensitive=True,
        mutability=m.ConfigMutability.RUNTIME,
    )
    assert mgr.get("MODE") == "ON"
    with pytest.raises(m.ConfigValidationError):
        mgr.set("MODE", "on")  # wrong case


def test_define_enum_rejects_invalid_value(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    mgr.define_enum(
        "E",
        default="a",
        values=("a", "b"),
        doc="e",
        mutability=m.ConfigMutability.RUNTIME,
    )
    with pytest.raises(m.ConfigValidationError):
        mgr.set("E", "c")


def test_get_unknown_key_raises_unknown_option_error(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.UnknownOptionError):
        m.cfg.get("DOES_NOT_EXIST")


def test_get_unknown_key_is_also_config_error(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigError):
        m.cfg.get("DOES_NOT_EXIST")


def test_set_unknown_key_raises_unknown_option_error(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.UnknownOptionError):
        m.cfg.set("DOES_NOT_EXIST", 1)


def test_set_runtime_option_succeeds(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    m.cfg.set("DEBUG", True)
    assert m.cfg.get("DEBUG") is True


def test_set_startup_option_before_lock_succeeds(cfgmod_factory):
    m = cfgmod_factory()
    # LOG_LEVEL is RUNTIME, PROFILE is STARTUP, runtime not yet locked
    original = m.cfg.get("PROFILE")
    m.cfg.set("PROFILE", 1)
    assert m.cfg.get("PROFILE") == 1
    m.cfg.set("PROFILE", original)  # restore


def test_set_immutable_option_raises(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    mgr.define_int(
        "FIXED",
        default=0,
        doc="immutable",
        mutability=m.ConfigMutability.IMMUTABLE,
    )
    with pytest.raises(m.ConfigError, match="immutable"):
        mgr.set("FIXED", 1)


def test_set_startup_option_after_lock_raises(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    mgr.define_int(
        "SCFG",
        default=0,
        doc="startup cfg",
        mutability=m.ConfigMutability.STARTUP,
    )
    mgr.lock_runtime()
    with pytest.raises(m.ConfigError, match="startup-only"):
        mgr.set("SCFG", 1)
    mgr.unlock_runtime_for_testing()


def test_set_parsing_error_raises_config_validation_error(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigValidationError):
        m.cfg.set("DEBUG", "not-a-bool")


def test_read_is_alias_for_get(cfgmod_factory):
    m = cfgmod_factory()
    assert m.cfg.read("DEBUG") == m.cfg.get("DEBUG")


def test_update_is_alias_for_set(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_VERBOSE": "true"})
    m.cfg.update("VERBOSE", False)
    assert m.cfg.get("VERBOSE") is False


def test_set_updates_os_environ(cfgmod_factory, monkeypatch):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    m.cfg.set("DEBUG", True)
    assert os.environ.get("NQX_DEBUG") == "True"


def test_set_many_applies_all_updates(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0", "NQX_VERBOSE": "false"})
    m.cfg.set_many({"DEBUG": True, "VERBOSE": True})
    assert m.cfg.get("DEBUG") is True
    assert m.cfg.get("VERBOSE") is True


def test_clear_override_restores_default(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    mgr.define_bool(
        "OV", default=False, doc="ov", mutability=m.ConfigMutability.RUNTIME
    )
    mgr.set("OV", True)
    assert mgr.get("OV") is True
    mgr.clear_override("OV")
    assert mgr.get("OV") is False


def test_getattr_reads_option(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    assert m.cfg.DEBUG is False


def test_setattr_sets_option(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    m.cfg.DEBUG = True
    assert m.cfg.get("DEBUG") is True


def test_getattr_unknown_raises_attribute_error(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(AttributeError):
        _ = m.cfg.NONEXISTENT_OPTION_XYZ


def test_setattr_unknown_raises_attribute_error(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(AttributeError):
        m.cfg.NONEXISTENT_OPTION_XYZ = 1


def test_patch_restores_value_on_exit(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    assert m.cfg.get("DEBUG") is False
    with m.cfg.patch("DEBUG", True):
        assert m.cfg.get("DEBUG") is True
    assert m.cfg.get("DEBUG") is False


def test_patch_mapping_form(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0", "NQX_VERBOSE": "true"})
    with m.cfg.patch({"DEBUG": True, "VERBOSE": False}):
        assert m.cfg.get("DEBUG") is True
        assert m.cfg.get("VERBOSE") is False
    assert m.cfg.get("DEBUG") is False
    assert m.cfg.get("VERBOSE") is True


def test_patch_kwargs_form(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    with m.cfg.patch(DEBUG=True):
        assert m.cfg.get("DEBUG") is True
    assert m.cfg.get("DEBUG") is False


def test_patch_restores_on_exception(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    try:
        with m.cfg.patch("DEBUG", True):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert m.cfg.get("DEBUG") is False


def test_patch_thread_local_only_affects_current_thread(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    other_thread_value = []

    def _reader():
        other_thread_value.append(m.cfg.get("DEBUG"))

    with m.cfg.patch("DEBUG", True, thread_local=True):
        assert m.cfg.get("DEBUG") is True
        t = threading.Thread(target=_reader)
        t.start()
        t.join()

    # Other thread sees the global value (False), not the thread-local True.
    assert other_thread_value == [False]
    # After context exit, thread-local is cleared.
    assert m.cfg.get("DEBUG") is False


def test_patch_thread_local_on_non_runtime_option_raises(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    mgr.define_int("S", default=0, doc="s", mutability=m.ConfigMutability.STARTUP)
    with pytest.raises(m.ConfigError):
        with mgr.patch("S", 1, thread_local=True):
            pass


def test_thread_local_override_context(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    with m.cfg.thread_local_override("DEBUG", True):
        assert m.cfg.get("DEBUG") is True
    assert m.cfg.get("DEBUG") is False


def test_snapshot_returns_dict_of_all_options(cfgmod_factory):
    m = cfgmod_factory()
    snap = m.cfg.snapshot()
    assert isinstance(snap, dict)
    assert "DEBUG" in snap
    assert "VERBOSE" in snap


def test_snapshot_is_a_copy(cfgmod_factory):
    m = cfgmod_factory()
    snap = m.cfg.snapshot()
    snap["DEBUG"] = "corrupted"
    assert m.cfg.get("DEBUG") in (True, False)


def test_values_property_ignores_thread_local(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    with m.cfg.thread_local_override("DEBUG", True):
        # values property uses thread_local=False
        assert m.cfg.values["DEBUG"] is False


def test_user_overrides_returns_only_overridden(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    assert "DEBUG" not in m.cfg.user_overrides()
    m.cfg.set("DEBUG", True)
    assert "DEBUG" in m.cfg.user_overrides()


def test_fingerprint_is_hex_string(cfgmod_factory):
    m = cfgmod_factory()
    fp = m.cfg.fingerprint()
    assert isinstance(fp, str)
    assert len(fp) == 64  # SHA-256 hex digest


def test_fingerprint_changes_when_option_changes(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    fp_before = m.cfg.fingerprint()
    m.cfg.set("DEBUG", True)
    fp_after = m.cfg.fingerprint()
    assert fp_before != fp_after


def test_fingerprint_excludes_non_fingerprint_options(cfgmod_factory):
    m = cfgmod_factory()
    # LOG_LEVEL has include_in_fingerprint=False
    fp_before = m.cfg.fingerprint()
    m.cfg.set("LOG_LEVEL", "DEBUG")
    fp_after = m.cfg.fingerprint()
    assert fp_before == fp_after  # LOG_LEVEL excluded


def test_describe_option_returns_metadata(cfgmod_factory):
    m = cfgmod_factory()
    desc = m.cfg.describe_option("DEBUG")
    assert desc["name"] == "DEBUG"
    assert "doc" in desc
    assert "mutability" in desc
    assert desc["mutability"] == "runtime"
    assert "default" in desc
    assert "effective_value" in desc


def test_options_by_mutability_contains_all_keys(cfgmod_factory):
    m = cfgmod_factory()
    groups = m.cfg.options_by_mutability()
    all_grouped = set()
    for names in groups.values():
        all_grouped.update(names)
    assert all_grouped == set(m.cfg.list_options())


def test_options_by_mutability_classifies_debug_as_runtime(cfgmod_factory):
    m = cfgmod_factory()
    groups = m.cfg.options_by_mutability()
    assert "DEBUG" in groups[m.ConfigMutability.RUNTIME]


def test_options_by_role_groups_correctly(cfgmod_factory):
    m = cfgmod_factory()
    by_role = m.cfg.options_by_role()
    assert "debugging" in by_role
    assert "profiling" in by_role
    assert "DEBUG" in by_role["debugging"]
    assert "PROFILE" in by_role["profiling"]


def test_is_runtime_mutable_true_for_runtime_options(cfgmod_factory):
    m = cfgmod_factory()
    assert m.cfg.is_runtime_mutable("DEBUG") is True
    assert m.cfg.is_runtime_mutable("VERBOSE") is True


def test_is_runtime_mutable_false_for_startup_options(cfgmod_factory):
    m = cfgmod_factory()
    assert m.cfg.is_runtime_mutable("PROFILE") is False
    assert m.cfg.is_runtime_mutable("ENABLE_X64") is False


def test_value_source_default(cfgmod_factory):
    m = cfgmod_factory()
    src = m.cfg.value_source("DEBUG")
    assert src is m.ConfigSource.DEFAULT or src is m.ConfigSource.ENV_DEFAULT


def test_value_source_user_after_set(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    m.cfg.set("DEBUG", True)
    assert m.cfg.value_source("DEBUG") is m.ConfigSource.USER


def test_value_source_env_force(cfgmod_factory, monkeypatch):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    monkeypatch.setenv("NQX_FORCE_X", "1")
    mgr.define_int(
        "X",
        default=0,
        doc="x",
        env_force="NQX_FORCE_X",
    )
    assert mgr.value_source("X") is m.ConfigSource.ENV_FORCE


def test_value_source_env_default(cfgmod_factory, monkeypatch):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    monkeypatch.setenv("NQX_MY_VAL", "7")
    mgr.define_int("MY_VAL", default=0, doc="v", env_default="NQX_MY_VAL")
    assert mgr.value_source("MY_VAL") is m.ConfigSource.ENV_DEFAULT


def test_lock_runtime_property(cfgmod_factory):
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    assert mgr.runtime_locked is False
    mgr.lock_runtime()
    assert mgr.runtime_locked is True
    mgr.unlock_runtime_for_testing()
    assert mgr.runtime_locked is False


def test_add_hook_fires_on_change(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    events = []

    def hook(event):
        events.append(event)

    m.cfg.add_hook("DEBUG", hook)
    m.cfg.set("DEBUG", True)

    assert len(events) == 1
    assert events[0].name == "DEBUG"
    assert events[0].old_value is False
    assert events[0].new_value is True
    assert events[0].source is m.ConfigSource.USER


def test_add_hook_run_immediately(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    events = []

    def hook(event):
        events.append(event)

    m.cfg.add_hook("DEBUG", hook, run_immediately=True)
    # Fired immediately even though value did not change
    assert len(events) == 1
    assert events[0].source is m.ConfigSource.DEFAULT


def test_add_hook_does_not_fire_when_value_unchanged(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    events = []
    m.cfg.add_hook("DEBUG", lambda e: events.append(e))
    m.cfg.set("DEBUG", False)  # same as current value
    assert events == []


def test_add_global_hook_fires_for_any_option(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0", "NQX_VERBOSE": "true"})
    events = []
    m.cfg.add_global_hook(lambda e: events.append(e.name))
    m.cfg.set("DEBUG", True)
    m.cfg.set("VERBOSE", False)
    assert "DEBUG" in events
    assert "VERBOSE" in events


def test_add_hook_receives_config_mutation_instance(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    received = []
    m.cfg.add_hook("DEBUG", lambda e: received.append(e))
    m.cfg.set("DEBUG", True)
    assert isinstance(received[0], m.ConfigMutation)
    assert received[0].mutability is m.ConfigMutability.RUNTIME


def test_default_debug_hook_refreshes_runtime_debug_module(cfgmod_factory, monkeypatch):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})

    calls = {"init": 0, "refresh": 0}
    import neuralqx.debug as dbg

    monkeypatch.setattr(
        dbg,
        "initialise",
        lambda force=False: calls.__setitem__("init", calls["init"] + 1),
        raising=True,
    )
    monkeypatch.setattr(
        dbg,
        "refresh_settings",
        lambda reinit=False: calls.__setitem__("refresh", calls["refresh"] + 1),
        raising=True,
    )

    m.cfg.set("DEBUG", True)
    m.cfg.set("DEBUG", False)

    assert calls["init"] == 1
    assert calls["refresh"] >= 1


def test_default_log_level_hook_refreshes_runtime_debug_module(
    cfgmod_factory, monkeypatch
):
    m = cfgmod_factory(env={"NQX_LOG_LEVEL": "INFO"})

    calls = {"refresh": 0}
    import neuralqx.debug as dbg

    monkeypatch.setattr(dbg, "initialise", lambda force=False: None, raising=True)
    monkeypatch.setattr(
        dbg,
        "refresh_settings",
        lambda reinit=False: calls.__setitem__("refresh", calls["refresh"] + 1),
        raising=True,
    )

    m.cfg.set("LOG_LEVEL", "DEBUG")

    assert calls["refresh"] == 1


def test_patch_emits_patch_source(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    events = []
    m.cfg.add_hook("DEBUG", lambda e: events.append(e))
    with m.cfg.patch("DEBUG", True):
        pass
    sources = [e.source for e in events]
    assert m.ConfigSource.PATCH in sources


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


def test_get_env_case_insensitive_returns_value_when_only_lowercase_key_exists(
    cfgmod_factory, monkeypatch
):
    m = cfgmod_factory()
    monkeypatch.delenv("NQX_VERBOSE", raising=False)
    monkeypatch.setenv("nqx_verbose", "true")
    assert m.ConfigManager._get_env_case_insensitive("NQX_VERBOSE") == "true"


def test_get_env_case_insensitive_returns_none_when_absent(cfgmod_factory, monkeypatch):
    m = cfgmod_factory()
    assert (
        m.ConfigManager._get_env_case_insensitive("NQX_DEFINITELY_NOT_SET_XYZ") is None
    )


def test_get_env_case_insensitive_prefers_first_inserted_when_multiple_case_variants_exist(
    cfgmod_factory, monkeypatch
):
    m = cfgmod_factory()
    monkeypatch.setenv("nqx_verbose", "true")
    # The canonical NQX_VERBOSE was written by register_option, it should be
    # found first since os.environ preserves insertion order on Python.
    result = m.ConfigManager._get_env_case_insensitive("NQX_VERBOSE")
    assert result == os.environ["NQX_VERBOSE"]


def test_profiling_deep_trace_defaults_are_registered(cfgmod_factory):
    m = cfgmod_factory()
    assert m.cfg.get("PROFILE_PY_CALLS") == 0
    assert m.cfg.get("PROFILE_PY_CALLS_INCLUDE") == "netket,neuralqx"
    assert m.cfg.get("PROFILE_PY_CALLS_EXCLUDE") == "neuralqx.profile"
    assert m.cfg.get("PROFILE_PY_CALLS_MAX_DEPTH") == 6


def test_log_level_default_is_info(cfgmod_factory):
    m = cfgmod_factory()
    assert m.cfg.get("LOG_LEVEL") == "INFO"


def test_log_level_is_runtime_mutable(cfgmod_factory):
    m = cfgmod_factory()
    assert m.cfg.is_runtime_mutable("LOG_LEVEL") is True


def test_debug_default_is_false(cfgmod_factory):
    m = cfgmod_factory()
    assert m.cfg.get("DEBUG") is False


def test_verbose_default_is_true(cfgmod_factory):
    m = cfgmod_factory()
    assert m.cfg.get("VERBOSE") is True


def test_experimental_default_is_false(cfgmod_factory):
    m = cfgmod_factory()
    assert m.cfg.get("EXPERIMENTAL") is False


def test_testing_default_is_false(cfgmod_factory):
    m = cfgmod_factory()
    assert m.cfg.get("TESTING") is False


def test_profile_defaults(cfgmod_factory):
    m = cfgmod_factory()
    assert m.cfg.get("PROFILE") == 0
    assert m.cfg.get("PROFILE_NVTX") == 0
    assert m.cfg.get("PROFILE_JAX_ANNOTATE") == 1
    assert m.cfg.get("PROFILE_JAX_TRACE") == 0
    assert m.cfg.get("PROFILE_SAMPLE_PERIOD_S") == pytest.approx(0.1)
    assert m.cfg.get("PROFILE_MAX_EVENTS") == 2_000_000
    assert m.cfg.get("PROFILE_RUN_ID") == ""


def test_enable_x64_env_default_respected(cfgmod_factory, monkeypatch):
    m = cfgmod_factory(env={"JAX_ENABLE_X64": "0"})
    assert m.cfg.get("ENABLE_X64") is False


def test_enable_x64_nqx_env_takes_priority(cfgmod_factory, monkeypatch):
    m = cfgmod_factory(env={"JAX_ENABLE_X64": "0", "NQX_ENABLE_X64": "1"})
    assert m.cfg.get("ENABLE_X64") is True


def test_log_level_rejects_invalid_value(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigValidationError):
        m.cfg.set("LOG_LEVEL", "VERBOSE")


def test_log_level_enum_accepts_valid_values(cfgmod_factory):
    m = cfgmod_factory()
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        m.cfg.set("LOG_LEVEL", level)
        assert m.cfg.get("LOG_LEVEL") == level


def test_statics_is_readonly_and_contains_expected_keys(cfgmod_factory):
    m = cfgmod_factory()
    s = m.cfg.statics
    assert isinstance(s, m.ReadOnlyDict)
    for k in [
        "Errors Directory",
        "Cache Directory",
        "Leach Directory",
        "Profiling Directory",
    ]:
        assert k in s
        assert isinstance(s[k], str)
    with pytest.raises(TypeError, match="read-only"):
        s["X"] = "y"


def test_get_static_returns_correct_value(cfgmod_factory):
    m = cfgmod_factory()
    errors_dir = m.cfg.get_static("Errors Directory")
    assert "neuralqx.readthedocs.io" in errors_dir


def test_get_static_missing_key_raises_key_error(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(KeyError, match="does not exist"):
        m.cfg.get_static("NOPE")


def test_warn_unused_prints_for_unknown_nqx_variables(
    cfgmod_factory, capsys, monkeypatch
):
    _ = cfgmod_factory(env={"NQX_SOMETHING_UNUSED": "abc"})
    out = capsys.readouterr().out
    assert "not\nrecognized by neuraLQX and will be ignored" in out
    assert "NQX_SOMETHING_UNUSED" in out
    assert "abc" in out


def test_warn_unused_silent_for_known_variables(cfgmod_factory, capsys):
    _ = cfgmod_factory()
    out = capsys.readouterr().out
    # No unknown NQX_ vars means no warning (only the regular stdout from show may appear)
    assert "not\nrecognized by neuraLQX and will be ignored" not in out


def test_sync_external_env_sets_expected_defaults(cfgmod_factory):
    # JAX_PLATFORMS, OMP_NUM_THREADS, and XLA_PYTHON_CLIENT_PREALLOCATE are
    # now set by neuralqx._boot (via the neuralqx_boot.pth site-packages hook)
    # before JAX is imported. _sync_external_envars skips those setdefault
    # calls when JAX is already in sys.modules (as it always is during tests),
    # so we verify the boot module is responsible instead.
    # NETKET_EXPERIMENTAL_SHARDING is no longer set, NetKet defaults it True.
    cfgmod_factory()
    assert os.environ.get("JAX_ENABLE_X64") is not None
    assert os.environ.get("NETKET_ENABLE_X64") is not None


def test_sync_external_env_x64_enabled(cfgmod_factory):
    cfgmod_factory(env={"NQX_ENABLE_X64": "1"})
    assert os.environ.get("NETKET_ENABLE_X64") == "1"
    assert os.environ.get("JAX_ENABLE_X64") == "1"


def test_sync_external_env_x64_disabled(cfgmod_factory):
    cfgmod_factory(env={"NQX_ENABLE_X64": "0"})
    assert os.environ.get("NETKET_ENABLE_X64") == "0"
    assert os.environ.get("JAX_ENABLE_X64") == "0"


def test_show_emits_table_with_expected_columns(cfgmod_factory, capsys):
    m = cfgmod_factory()
    m.cfg.show()
    out = capsys.readouterr().out
    assert "name" in out
    assert "mutability" in out
    assert "DEBUG" in out


def test_show_returns_table_string(cfgmod_factory):
    m = cfgmod_factory()
    import io

    buf = io.StringIO()
    result = m.cfg.show(file=buf)
    assert isinstance(result, str)
    assert "DEBUG" in result


def test_show_include_env_adds_columns(cfgmod_factory, capsys):
    m = cfgmod_factory()
    m.cfg.show(include_env=True)
    out = capsys.readouterr().out
    assert "env_default" in out


def test_repr_contains_options_count(cfgmod_factory):
    m = cfgmod_factory()
    r = repr(m.cfg)
    assert "ConfigManager" in r
    assert "options=" in r


def test_user_overrides_thread_local_returns_thread_local_overrides(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    with m.cfg.thread_local_override("DEBUG", True):
        tl_overrides = m.cfg.user_overrides(thread_local=True)
        assert "DEBUG" in tl_overrides
        assert tl_overrides["DEBUG"] is True
    # After exiting context, thread-local override is gone
    assert m.cfg.user_overrides(thread_local=True) == {}


def test_value_source_thread_local(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    with m.cfg.thread_local_override("DEBUG", True):
        src = m.cfg.value_source("DEBUG", thread_local=True)
        assert src is m.ConfigSource.THREAD_LOCAL


def test_clear_override_thread_local(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    mgr = m.cfg
    # Set a thread-local override
    overrides = mgr._thread_local_overrides(create=True)
    from neuralqx.configs import parse_bool

    overrides["DEBUG"] = True
    assert mgr.get("DEBUG") is True
    # Clear the thread-local override
    mgr.clear_override("DEBUG", thread_local=True)
    assert mgr.get("DEBUG") is False  # falls back to global default


def test_unset_repr(cfgmod_factory):
    m = cfgmod_factory()
    assert repr(m.UNSET) == "<UNSET>"


def test_format_table_cell_truncates_long_values(cfgmod_factory):
    m = cfgmod_factory()
    long_value = "x" * 100
    result = m.ConfigManager._format_table_cell(long_value, max_width=10)
    assert len(result) == 10
    assert result.endswith("...")


def test_format_table_cell_very_short_max_width(cfgmod_factory):
    m = cfgmod_factory()
    result = m.ConfigManager._format_table_cell("hello", max_width=3)
    assert len(result) == 3


def test_format_table_cell_unset(cfgmod_factory):
    m = cfgmod_factory()
    result = m.ConfigManager._format_table_cell(m.UNSET, max_width=48)
    assert result == "<unset>"


def test_patch_raises_if_none_arg1_and_arg2_given(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(TypeError):
        with m.cfg.patch(None, "value"):
            pass


def test_patch_raises_if_positional_with_kwargs(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(TypeError):
        with m.cfg.patch("DEBUG", True, VERBOSE=False):
            pass


def test_patch_raises_if_arg1_not_string_in_two_arg_form(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(TypeError):
        with m.cfg.patch(123, True):
            pass


def test_patch_raises_if_arg1_not_mapping_in_single_arg_form(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(TypeError):
        with m.cfg.patch(42):
            pass


def test_patch_raises_if_mapping_with_kwargs(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(TypeError):
        with m.cfg.patch({"DEBUG": True}, VERBOSE=False):
            pass


def test_should_init_jax_distributed_slurm(cfgmod_factory, monkeypatch):
    m = cfgmod_factory()
    monkeypatch.setenv("SLURM_NTASKS", "2")
    assert m._should_init_jax_distributed() is True


def test_should_init_jax_distributed_ompi(cfgmod_factory, monkeypatch):
    m = cfgmod_factory()
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert m._should_init_jax_distributed() is True


def test_should_init_jax_distributed_pmi(cfgmod_factory, monkeypatch):
    m = cfgmod_factory()
    monkeypatch.setenv("PMI_SIZE", "2")
    assert m._should_init_jax_distributed() is True


def test_should_init_jax_distributed_jax_process_count(cfgmod_factory, monkeypatch):
    m = cfgmod_factory()
    monkeypatch.setenv("JAX_PROCESS_COUNT", "2")
    assert m._should_init_jax_distributed() is True


def test_should_init_jax_distributed_coordinator_address(cfgmod_factory, monkeypatch):
    m = cfgmod_factory()
    monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "localhost:1234")
    assert m._should_init_jax_distributed() is True


def test_should_init_jax_distributed_false_when_single_process(
    cfgmod_factory, monkeypatch
):
    m = cfgmod_factory()
    for k in (
        "SLURM_NTASKS",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "JAX_PROCESS_COUNT",
        "JAX_COORDINATOR_ADDRESS",
    ):
        monkeypatch.delenv(k, raising=False)
    assert m._should_init_jax_distributed() is False


def test_env_default_tuple_first_present_wins(cfgmod_factory, monkeypatch):
    m = cfgmod_factory()
    monkeypatch.setenv("NQX_FIRST", "10")
    monkeypatch.setenv("NQX_SECOND", "20")
    mgr = _bare_manager(m)
    mgr.define_int(
        "MULTI",
        default=0,
        doc="multi env",
        env_default=("NQX_FIRST", "NQX_SECOND"),
    )
    assert mgr.get("MULTI") == 10  # NQX_FIRST wins


def test_env_default_tuple_second_used_when_first_absent(cfgmod_factory, monkeypatch):
    m = cfgmod_factory()
    monkeypatch.delenv("NQX_FIRST", raising=False)
    monkeypatch.setenv("NQX_SECOND", "20")
    mgr = _bare_manager(m)
    mgr.define_int(
        "MULTI",
        default=0,
        doc="multi env",
        env_default=("NQX_FIRST", "NQX_SECOND"),
    )
    assert mgr.get("MULTI") == 20  # NQX_SECOND used as fallback


def test_fingerprint_with_thread_local(cfgmod_factory):
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    fp_global = m.cfg.fingerprint(thread_local=False)
    with m.cfg.thread_local_override("DEBUG", True):
        fp_tl = m.cfg.fingerprint(thread_local=True)
    assert fp_global != fp_tl


def test_patch_restores_pre_existing_global_override(cfgmod_factory):
    """Covers the patch() branch that restores a pre-existing user_override."""
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    # Establish a pre-existing override before the patch
    m.cfg.set("DEBUG", True)
    assert m.cfg.get("DEBUG") is True

    with m.cfg.patch("DEBUG", False):
        assert m.cfg.get("DEBUG") is False

    # After the patch, the pre-existing True override should be restored
    assert m.cfg.get("DEBUG") is True


def test_patch_thread_local_restores_pre_existing_thread_local_override(cfgmod_factory):
    """Covers the thread-local patch() branch that restores an existing TL override."""
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})

    # Establish a thread-local override, then patch over it
    tl_overrides = m.cfg._thread_local_overrides(create=True)
    tl_overrides["DEBUG"] = True

    with m.cfg.patch("DEBUG", False, thread_local=True):
        assert m.cfg.get("DEBUG") is False

    # The pre-existing thread-local True should be restored
    assert m.cfg.get("DEBUG") is True

    # Clean up
    m.cfg.clear_override("DEBUG", thread_local=True)


def test_show_without_current_columns(cfgmod_factory, capsys):
    m = cfgmod_factory()
    result = m.cfg.show(include_current=False)
    assert "effective" not in result
    assert "source" not in result
    assert "DEBUG" in result


def test_clear_override_thread_local_when_no_overrides_dict(cfgmod_factory):
    """Covers the branch where overrides is None in clear_override(thread_local=True)."""
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    # Don't create any thread-local overrides first, clearing should be a no-op
    result = m.cfg.clear_override("DEBUG", thread_local=True)
    assert result is False  # falls back to default value


def test_parse_bool_non_string_non_bool_raises(cfgmod_factory):
    m = cfgmod_factory()
    with pytest.raises(m.ConfigValidationError):
        m.parse_bool(3.14)


def test_parse_csv_tuple_sequence_with_empty_string_entries(cfgmod_factory):
    """Empty stripped entries are skipped."""
    m = cfgmod_factory()
    result = m.parse_csv_tuple(["  ", "hello", ""])
    assert result == ("hello",)


def test_parse_csv_tuple_non_sequence_non_string_raises(cfgmod_factory):
    """parse_csv_tuple raises for non-string, non-sequence values."""
    m = cfgmod_factory()
    with pytest.raises(m.ConfigValidationError):
        m.parse_csv_tuple(42)


def test_register_option_empty_name_raises(cfgmod_factory):
    """Covers the empty option name guard at line 707."""
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    with pytest.raises(m.ConfigValidationError, match="non-empty string"):
        mgr.register_option(
            m.ConfigOption(
                name="", default=0, doc="bad", value_type=int, parser=m.parse_int
            )
        )


def test_register_option_normalises_default_via_parser(cfgmod_factory):
    """
    Covers the replace(option, default=canonical_default) branch.
    When parse(default) returns a different value from default, the option is
    updated.  define_enum always normalises to lower-case, so define_enum with
    an upper-case default will trigger this branch.
    """
    m = cfgmod_factory()
    mgr = _bare_manager(m)
    # The enum parser normalises to lower-case, passing 'INFO' causes replace()
    mgr.define_enum(
        "NORM",
        default="INFO",
        values=("info", "debug"),
        doc="normalisation test",
        mutability=m.ConfigMutability.RUNTIME,
    )
    assert mgr.get("NORM") == "info"


def test_get_returns_user_override_when_set(cfgmod_factory):
    """Covers the `return state.user_override` branch in get()."""
    m = cfgmod_factory(env={"NQX_DEBUG": "0"})
    mgr = m.cfg
    mgr.set("DEBUG", True)
    # value_source must be USER and get() must return the override
    assert mgr.value_source("DEBUG") is m.ConfigSource.USER
    assert mgr.get("DEBUG", thread_local=False) is True
