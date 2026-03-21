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


import ast
import importlib.util
import types
from pathlib import Path
from dataclasses import FrozenInstanceError

import pytest


def _load_dependency_check_shim():

    spec = importlib.util.find_spec("neuralqx.utils.module.dependency_check")
    if spec is None or spec.origin is None:
        pytest.skip("Could not locate neuralqx.utils.module.dependency_check")

    src_path = Path(spec.origin)
    src = src_path.read_text(encoding="utf-8")

    tree = ast.parse(src, filename=str(src_path))

    if tree.body:
        last = tree.body[-1]
        if (
            isinstance(last, ast.Expr)
            and isinstance(last.value, ast.Call)
            and isinstance(last.value.func, ast.Name)
            and last.value.func.id == "enforce_default_dependencies"
            and not last.value.args
            and not last.value.keywords
        ):
            tree.body = tree.body[:-1]

    code = compile(tree, filename=str(src_path), mode="exec")

    mod = types.ModuleType("neuralqx.utils.module._dependency_check_testshim")
    mod.__file__ = str(src_path)
    mod.__package__ = "neuralqx.utils.module"

    exec(code, mod.__dict__)
    return mod


@pytest.fixture
def dchk():
    return _load_dependency_check_shim()


def test_dependency_policy_is_frozen_dataclass(dchk):
    Policy = dchk.DependencyPolicy
    p = Policy(name="x", minimum=(1, 2, 3), maximum=(2, 0, 0), rationale="because")

    with pytest.raises(FrozenInstanceError):
        p.name = "y"


def test_dependency_violation_is_runtimeerror_subclass(dchk):
    assert issubclass(dchk.DependencyViolation, RuntimeError)


def test_format_missing_dependency_includes_name_rationale_and_action(dchk):
    Policy = dchk.DependencyPolicy
    p = Policy(name="missing_pkg", rationale="Needed for correctness.")

    msg = dchk._format_missing_dependency(p)
    assert "Dependency check failed" in msg
    assert "missing_pkg" in msg
    assert "Needed for correctness." in msg
    assert "Install the missing package" in msg


def test_format_missing_dependency_uses_default_rationale_when_empty(dchk):
    Policy = dchk.DependencyPolicy
    p = Policy(name="missing_pkg", rationale="")

    msg = dchk._format_missing_dependency(p)
    assert "missing_pkg" in msg
    assert "This package is required for correct execution of neuraLQX." in msg


def test_format_version_error_contains_constraints_and_versions(dchk, monkeypatch):
    Policy = dchk.DependencyPolicy
    p = Policy(name="foo", minimum=(1, 2, 3), maximum=(2, 0, 0), rationale="R")

    monkeypatch.setattr(dchk, "get_module_version_string", lambda name: "1.9.0+abc")

    msg = dchk._format_version_error(
        p, installed=dchk.Version("1.9.0"), relation="below the supported minimum"
    )

    assert "Dependency version incompatibility detected." in msg
    assert "foo" in msg
    assert "1.9.0+abc" in msg
    assert "parsed as 1.9.0" in msg

    assert ">= 1.2.3 and <= 2.0.0" in msg
    assert "below the supported minimum" in msg
    assert "Why this matters" in msg
    assert "R" in msg


def test_format_version_error_works_with_only_minimum(dchk, monkeypatch):
    Policy = dchk.DependencyPolicy
    p = Policy(name="foo", minimum=(1, 2, 3), maximum=None)

    monkeypatch.setattr(dchk, "get_module_version_string", lambda name: "0.0.1")
    msg = dchk._format_version_error(
        p, installed=dchk.Version("0.0.1"), relation="below the supported minimum"
    )
    assert ">= 1.2.3" in msg
    assert "<=" not in msg


def test_format_version_error_works_with_only_maximum(dchk, monkeypatch):
    Policy = dchk.DependencyPolicy
    p = Policy(name="foo", minimum=None, maximum=(3, 0, 0))

    monkeypatch.setattr(dchk, "get_module_version_string", lambda name: "4.0.0")
    msg = dchk._format_version_error(
        p, installed=dchk.Version("4.0.0"), relation="above the supported maximum"
    )
    assert "<= 3.0.0" in msg
    assert ">=" not in msg


def test_enforce_policy_missing_module_raises_dependency_violation(dchk, monkeypatch):
    Policy = dchk.DependencyPolicy
    p = Policy(name="does_not_exist", minimum=(0, 0, 1), rationale="Need it.")

    def _import_module(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(dchk.importlib, "import_module", _import_module)

    called = {"vstr": False}

    def _get_module_version_string(_):
        called["vstr"] = True
        return "999.999.999"

    monkeypatch.setattr(dchk, "get_module_version_string", _get_module_version_string)

    with pytest.raises(dchk.DependencyViolation) as exc:
        dchk.enforce_policy(p)

    assert "required package not found" in str(exc.value)
    assert "does_not_exist" in str(exc.value)
    assert "Need it." in str(exc.value)
    assert called["vstr"] is False


def test_enforce_policy_below_minimum_raises_dependency_violation(dchk, monkeypatch):
    Policy = dchk.DependencyPolicy
    p = Policy(name="dummy", minimum=(1, 0, 0), rationale="min rationale")

    monkeypatch.setattr(dchk.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(dchk, "get_module_version_string", lambda name: "0.9.0")

    with pytest.raises(dchk.DependencyViolation) as exc:
        dchk.enforce_policy(p)

    msg = str(exc.value)
    assert "below the supported minimum" in msg
    assert "dummy" in msg
    assert "0.9.0 (parsed as 0.9.0)" in msg
    assert ">= 1.0.0" in msg
    assert "min rationale" in msg


def test_enforce_policy_above_maximum_raises_dependency_violation(dchk, monkeypatch):
    Policy = dchk.DependencyPolicy
    p = Policy(name="dummy", maximum=(1, 0, 0), rationale="max rationale")

    monkeypatch.setattr(dchk.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(dchk, "get_module_version_string", lambda name: "1.2.0")

    with pytest.raises(dchk.DependencyViolation) as exc:
        dchk.enforce_policy(p)

    msg = str(exc.value)
    assert "above the supported maximum" in msg
    assert "dummy" in msg
    assert "1.2.0 (parsed as 1.2.0)" in msg
    assert "<= 1.0.0" in msg
    assert "max rationale" in msg


def test_enforce_policy_passes_when_in_range(dchk, monkeypatch):
    Policy = dchk.DependencyPolicy
    p = Policy(name="dummy", minimum=(1, 0, 0), maximum=(2, 0, 0), rationale="ok")

    monkeypatch.setattr(dchk.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(dchk, "get_module_version_string", lambda name: "1.5.0")

    dchk.enforce_policy(p)


def test_enforce_policy_with_no_min_no_max_only_checks_import(dchk, monkeypatch):
    Policy = dchk.DependencyPolicy
    p = Policy(name="dummy", minimum=None, maximum=None)

    monkeypatch.setattr(dchk.importlib, "import_module", lambda name: object())

    called = {"vstr": False}

    def _get_module_version_string(_):
        called["vstr"] = True
        return "0.0.0"

    monkeypatch.setattr(dchk, "get_module_version_string", _get_module_version_string)

    dchk.enforce_policy(p)
    assert called["vstr"] is True


def test_enforce_policies_stops_at_first_violation(dchk, monkeypatch):
    Policy = dchk.DependencyPolicy
    p1 = Policy(name="p1", minimum=(1, 0, 0))
    p2 = Policy(name="p2", minimum=(1, 0, 0))

    calls = []

    def _enforce_policy(policy):
        calls.append(policy.name)
        if policy.name == "p1":
            raise dchk.DependencyViolation("boom")

    monkeypatch.setattr(dchk, "enforce_policy", _enforce_policy)

    with pytest.raises(dchk.DependencyViolation):
        dchk.enforce_policies([p1, p2])

    assert calls == ["p1"]


def test_enforce_default_dependencies_calls_enforce_policies_with_defaults(
    dchk, monkeypatch
):
    captured = {}

    def _enforce_policies(policies):
        captured["policies"] = policies

    monkeypatch.setattr(dchk, "enforce_policies", _enforce_policies)

    dchk.enforce_default_dependencies()
    assert "policies" in captured
    assert captured["policies"] == dchk._DEFAULT_POLICIES
    assert len(captured["policies"]) >= 1


def test_dependency_policy_rejects_invalid_bound(dchk):
    with pytest.raises(ValueError):
        dchk.DependencyPolicy(name="x", minimum="not-a-version")


def test_dependency_policy_rejects_inverted_bounds(dchk):
    with pytest.raises(ValueError):
        dchk.DependencyPolicy(name="x", minimum="2.0.0", maximum="1.0.0")


def test_enforce_policy_unparseable_installed_version_raises_dependency_violation(
    dchk, monkeypatch
):
    Policy = dchk.DependencyPolicy
    p = Policy(name="dummy", minimum=(1, 0, 0), rationale="strict check")

    monkeypatch.setattr(dchk.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(
        dchk, "get_module_version_string", lambda name: "nightly-local-build"
    )

    with pytest.raises(dchk.DependencyViolation) as exc:
        dchk.enforce_policy(p)

    msg = str(exc.value)
    assert "unparseable" in msg
    assert "nightly-local-build" in msg
    assert "strict check" in msg
