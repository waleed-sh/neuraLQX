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
import inspect
import warnings
import pytest


@pytest.fixture
def dmod():
    return importlib.import_module("neuralqx.utils.deprecation")


def _assert_futurewarning_with_text(wrec, must_contain):

    assert len(wrec) == 1, f"Expected exactly 1 warning, got {len(wrec)}"
    w = wrec[0]
    assert issubclass(w.category, FutureWarning)
    text = str(w.message)
    if isinstance(must_contain, str):
        must_contain = [must_contain]
    for s in must_contain:
        assert s in text, f"Expected warning to contain {s!r}, got:\n{text}"


def test_deprecated_function_warns_and_returns_value_and_preserves_wraps(dmod):
    @dmod.deprecated()
    def f(x):
        """docstring"""
        return x + 1

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        out = f(3)

    assert out == 4
    _assert_futurewarning_with_text(
        wrec,
        [
            "Call to deprecated function 'f'",
            "Function 'f' is now deprecated",
            "removed in the next version release",
            "Please update your code",
        ],
    )

    assert f.__name__ == "f"
    assert "docstring" in (f.__doc__ or "")
    assert hasattr(f, "__wrapped__")
    assert f.__wrapped__(3) == 4


def test_deprecated_function_with_func_name_override(dmod):
    @dmod.deprecated(func_name="new_fancy_name")
    def original(x):
        return x

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        original(1)

    _assert_futurewarning_with_text(
        wrec,
        [
            "deprecated function 'new_fancy_name'",
            "Function 'new_fancy_name' is now deprecated",
        ],
    )
    assert original.__name__ == "original"


def test_deprecated_includes_reason_and_dedent(dmod):
    reason = """
        Line 1
            Line 2 indented
    """

    @dmod.deprecated(reason=reason)
    def f():
        return "ok"

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        f()

    msg = str(wrec[0].message)
    assert "Notes:" in msg
    assert "Line 1" in msg
    assert "Line 2 indented" in msg


def test_deprecated_class_warns_on_instantiation(dmod):
    @dmod.deprecated()
    class C:
        def __init__(self, x):
            self.x = x

    assert not inspect.isclass(C)
    assert callable(C)

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        c = C(7)

    assert getattr(c, "x") == 7
    assert c.__class__.__name__ == "C"

    _assert_futurewarning_with_text(
        wrec,
        [
            "Call to deprecated class 'C'",
            "Class 'C' is now deprecated",
        ],
    )


def test_deprecation_warning_dedents_and_warns_futurewarning(dmod):
    msg = """
        Hello
            World
    """
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        dmod.deprecation_warning(msg)

    _assert_futurewarning_with_text(wrec, ["Hello", "World"])
    assert str(wrec[0].message).lstrip().startswith("Hello")


def test_old_name_deprecation_warns_contains_old_new_and_reason_and_preserves_wraps(
    dmod,
):
    reason = """
        Use the new API.
    """

    @dmod.old_name_deprecation("new_name", reason=reason)
    def old_name(x):
        """doc"""
        return x * 2

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        out = old_name(3)

    assert out == 6
    _assert_futurewarning_with_text(
        wrec,
        [
            "old_name has been renamed to new_name",
            "old_name",
            "new_name",
            "Use the new API.",
        ],
    )

    assert old_name.__name__ == "old_name"
    assert "doc" in (old_name.__doc__ or "")
    assert hasattr(old_name, "__wrapped__")
    assert old_name.__wrapped__(3) == 6


def test_parameter_name_deprecation_warns_only_if_old_param_in_kwargs_and_does_not_mutate_kwargs(
    dmod,
):
    @dmod.parameter_name_deprecation("old", "new", reason="Because reasons.")
    def f(**kwargs):
        return dict(kwargs)

    kwargs = {"old": 1, "other": 2}
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        out = f(**kwargs)

    assert out == {"old": 1, "other": 2}
    _assert_futurewarning_with_text(
        wrec,
        [
            "The parameter `old` in function `f` is deprecated",
            "`new`",
            "Because reasons.",
        ],
    )

    assert "old" in out and "new" not in out

    with warnings.catch_warnings(record=True) as wrec2:
        warnings.simplefilter("always")
        out2 = f(new=1)

    assert out2 == {"new": 1}
    assert len(wrec2) == 0


def test_deprecated_stacklevel_points_to_caller(dmod):
    @dmod.deprecated()
    def f():
        return 1

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        f()

    assert wrec[0].filename.endswith("test_deprecation.py")
