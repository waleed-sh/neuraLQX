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
import pytest


@pytest.fixture
def emod():
    return importlib.import_module("neuralqx.utils.experimental")


@pytest.fixture
def DeniedExperimentalFeatureError(emod):
    return emod.DeniedExperimentalFeatureError


def test_experimental_function_denied_raises_and_blocks_execution(
    emod, DeniedExperimentalFeatureError, monkeypatch
):
    monkeypatch.setattr(emod.cfg, "get", lambda key: False)

    called = {"hit": False}

    @emod.experimental
    def foo(x):
        """doc"""
        called["hit"] = True
        return x + 1

    with pytest.raises(DeniedExperimentalFeatureError) as exc:
        foo(10)

    assert called["hit"] is False

    assert "foo()" in str(exc.value)


def test_experimental_function_allowed_executes(emod, monkeypatch):
    monkeypatch.setattr(emod.cfg, "get", lambda key: True)

    @emod.experimental
    def add(a, b=2):
        return a + b

    assert add(3) == 5
    assert add(3, b=7) == 10


def test_experimental_function_preserves_wraps_metadata(emod, monkeypatch):
    monkeypatch.setattr(emod.cfg, "get", lambda key: True)

    @emod.experimental
    def bar():
        """hello there"""
        return 1

    assert bar.__name__ == "bar"
    assert "hello there" in (bar.__doc__ or "")
    assert hasattr(bar, "__wrapped__")
    assert bar.__wrapped__() == 1


def test_experimental_class_denied_raises_and_blocks_init(
    emod, DeniedExperimentalFeatureError, monkeypatch
):
    monkeypatch.setattr(emod.cfg, "get", lambda key: False)

    called = {"init": False}

    @emod.experimental
    class C:
        def __init__(self, x):
            called["init"] = True
            self.x = x

    with pytest.raises(DeniedExperimentalFeatureError) as exc:
        C(123)

    assert called["init"] is False
    assert "C" in str(exc.value)
    assert "()" not in str(exc.value)


def test_experimental_class_allowed_instantiates_and_init_runs(emod, monkeypatch):
    monkeypatch.setattr(emod.cfg, "get", lambda key: True)

    @emod.experimental
    class D:
        def __init__(self, x, *, y=0):
            self.x = x
            self.y = y

    obj = D(5, y=9)
    assert obj.x == 5
    assert obj.y == 9


def test_experimental_class_init_wraps_metadata(emod, monkeypatch):
    monkeypatch.setattr(emod.cfg, "get", lambda key: True)

    @emod.experimental
    class E:
        def __init__(self):
            """init doc"""
            self.ok = True

    assert E.__init__.__name__ == "__init__"
    assert hasattr(E.__init__, "__wrapped__")
    e = E()
    assert e.ok is True


def test_experimental_rejects_non_callable_non_class(emod):
    with pytest.raises(
        TypeError,
        match=r"@experimental decorator can only be applied to functions and classes",
    ):
        emod.experimental(123)
