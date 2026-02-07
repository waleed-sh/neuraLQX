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
import importlib.util
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest


@pytest.fixture
def parsing(project_root: Path):

    try:
        import neuralqx.utils.parsing as mod

        return mod
    except ModuleNotFoundError:
        mp = pytest.MonkeyPatch()

        flax_mod = types.ModuleType("flax")
        linen_mod = types.ModuleType("flax.linen")

        class StubFlaxModule:
            pass

        linen_mod.Module = StubFlaxModule
        flax_mod.linen = linen_mod

        mp.setitem(sys.modules, "flax", flax_mod)
        mp.setitem(sys.modules, "flax.linen", linen_mod)

        optax_mod = types.ModuleType("optax")
        mp.setitem(sys.modules, "optax", optax_mod)

        netket_mod = types.ModuleType("netket")
        nkopt_mod = types.ModuleType("netket.optimizer")

        class Adam:
            pass

        class AdaGrad:
            pass

        class Momentum:
            pass

        class Sgd:
            pass

        class RmsProp:
            pass

        nkopt_mod.Adam = Adam
        nkopt_mod.AdaGrad = AdaGrad
        nkopt_mod.Momentum = Momentum
        nkopt_mod.Sgd = Sgd
        nkopt_mod.RmsProp = RmsProp

        netket_mod.optimizer = nkopt_mod
        mp.setitem(sys.modules, "netket", netket_mod)
        mp.setitem(sys.modules, "netket.optimizer", nkopt_mod)

        path = project_root / "neuralqx" / "utils" / "parsing.py"
        assert path.exists(), f"Expected parsing.py at {path}"

        name = "neuralqx.utils._parsing_testload"
        loader = SourceFileLoader(name, str(path))
        spec = importlib.util.spec_from_loader(name, loader)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        loader.exec_module(mod)

        mp.undo()
        return mod


def test_required_kwargs_missing_unconditional(parsing):
    @parsing.required_kwargs("a", "b")
    def f(**kwargs):
        return kwargs["a"] + kwargs["b"]

    with pytest.raises(ValueError, match=r"Missing required argument b for 'f\(\)'"):
        f(a=1)

    with pytest.raises(ValueError, match=r"Missing required argument a for 'f\(\)'"):
        f(b=1)

    with pytest.raises(ValueError, match=r"Missing required argument a, b for 'f\(\)'"):
        f()


def test_required_kwargs_conditional_missing(parsing):
    @parsing.required_kwargs(
        "sampler_type",
        conditional_args={"sampler_type": {"Weighted": ["rules", "probabilities"]}},
    )
    def f(**kwargs):
        return "ok"

    with pytest.raises(
        ValueError, match=r"'rules' is required when 'sampler_type' = 'Weighted'"
    ):
        f(sampler_type="Weighted", probabilities=[0.5, 0.5])

    with pytest.raises(
        ValueError,
        match=r"'probabilities' is required when 'sampler_type' = 'Weighted'",
    ):
        f(sampler_type="Weighted", rules=["x"])


def test_required_kwargs_conditional_not_triggered(parsing):
    @parsing.required_kwargs(
        "sampler_type",
        conditional_args={"sampler_type": {"Weighted": ["rules", "probabilities"]}},
    )
    def f(**kwargs):
        return "ok"

    assert f(sampler_type="Other") == "ok"

    @parsing.required_kwargs("x", conditional_args={"mode": {"A": ["y"]}})
    def g(**kwargs):
        return "ok"

    assert g(x=1) == "ok"


def test_required_kwargs_positional_does_not_satisfy_kwargs_regression(parsing):
    @parsing.required_kwargs("a")
    def f(a, **kwargs):
        return a

    with pytest.raises(ValueError, match=r"Missing required argument a .*f\(\)"):
        f(123)


def test_required_kwargs_preserves_wraps_metadata(parsing):
    def original_doc():
        """docstring here"""

    @parsing.required_kwargs("x")
    def f(**kwargs):
        """docstring here"""
        return kwargs["x"]

    assert f.__name__ == "f"
    assert f.__doc__ and "docstring here" in f.__doc__
    assert hasattr(f, "__wrapped__")


def test_get_attr_val_module_class_and_callables(parsing):
    import math

    assert parsing.get_attr_val(math) == "Module 'math'"

    class C:
        pass

    assert parsing.get_attr_val(C) == "C"

    def fn():
        return 1

    assert parsing.get_attr_val(fn) == "fn"

    class Functor:
        def __call__(self):
            return 0

    assert parsing.get_attr_val(Functor()) == "Functor"


def test_get_attr_val_containers_and_partial(parsing):
    import math
    from functools import partial

    assert parsing.get_attr_val([1, 2]) == "['1', '2']"
    assert parsing.get_attr_val((1, 2)) == "('1', '2')"
    assert parsing.get_attr_val({"a": 1}) == "{'a': '1'}"

    p = partial(math.sin, 0.1)
    assert parsing.get_attr_val(p) == "partial(sin)"


def test_get_attr_val_str_fallback_and_str_failure(parsing):
    class X:
        def __str__(self):
            return "hello"

    assert parsing.get_attr_val(X()) == "hello"

    class BadStr:
        def __str__(self):
            raise RuntimeError("nope")

    assert parsing.get_attr_val(BadStr()) == "'BadStr'"


def test_get_attr_val_flax_module_instance_branch_without_real_flax(
    parsing, monkeypatch
):

    flax_mod = types.ModuleType("flax")
    linen_mod = types.ModuleType("flax.linen")

    class M:
        pass

    linen_mod.Module = M
    flax_mod.linen = linen_mod

    monkeypatch.setitem(sys.modules, "flax", flax_mod)
    monkeypatch.setitem(sys.modules, "flax.linen", linen_mod)

    class MyModule(M):
        pass

    assert parsing.get_attr_val(MyModule()) == "MyModule"


class DummyLogger:
    def __init__(self):
        self.added = []
        self.logged = []

    def add_field(self, group, field_name):
        self.added.append((group, field_name))

    def log(self, field_name, value):
        self.logged.append((field_name, value))


def test_log_module_attributes_uses_solver_logger_and_recurses(parsing, monkeypatch):
    class DummyModule:
        pass

    monkeypatch.setattr(parsing.fln, "Module", DummyModule, raising=False)

    class Leaf(DummyModule):
        def __init__(self):
            self.weight = 7
            self._private = "skip"
            self.name = "skip"

        def __call__(self):
            return None

    class Parent(DummyModule):
        def __init__(self):
            self.alpha = 0.25
            self.child = Leaf()
            self.name = "skip"

        def __call__(self):
            return None

    logger = DummyLogger()

    class Solver:
        def __init__(self, logger):
            self.logger = logger

    parsing.log_module_attributes(Solver(logger), Parent())

    logged_fields = {k for (k, _) in logger.logged}

    assert "Parent.alpha" in logged_fields
    assert "Parent.child" in logged_fields

    assert "Parent.child.weight" in logged_fields

    assert "Parent._private" not in logged_fields
    assert "Parent.name" not in logged_fields
    assert "Parent.child._private" not in logged_fields
    assert "Parent.child.name" not in logged_fields

    assert all(group == "Network Configs" for (group, _) in logger.added)


def test_log_module_attributes_falls_back_to_solver__logger_and_prefix_and_base(
    parsing, monkeypatch
):
    class DummyModule:
        pass

    monkeypatch.setattr(parsing.fln, "Module", DummyModule, raising=False)

    class Base(DummyModule):
        def __init__(self):
            self.base_param = 123

        def __call__(self):
            return None

    class Wrapper(DummyModule):
        def __init__(self):
            self.base = Base()
            self.extra = "x"

        def __call__(self):
            return None

    logger = DummyLogger()

    class Solver:
        def __init__(self, logger):
            self._logger = logger

    parsing.log_module_attributes(Solver(logger), Wrapper(), prefix="MyNet")

    logged_fields = {k for (k, _) in logger.logged}

    assert "MyNet.extra" in logged_fields
    assert "MyNet.base" in logged_fields

    assert "MyNet.base.base_param" in logged_fields


def test_str_edge_parser_rejects_missing_parentheses(parsing):
    with pytest.raises(
        ValueError,
        match=r"Invalid edge format\. The input edge must be a tuple of integers",
    ):
        parsing.str_edge_parser("[1, 2]")


def test_str_edge_parser_accepts_planar_tuple(parsing):
    assert parsing.str_edge_parser("(1, 2)") == (1, 2)


def test_str_edge_parser_accepts_tuple_of_tuples(parsing):
    s = "((1, 2, 3), (4, 5, 6))"
    assert parsing.str_edge_parser(s) == ((1, 2, 3), (4, 5, 6))


def test_str_edge_parser_non_tuple_or_bad_literal_raises_valueerror(parsing):
    with pytest.raises(ValueError, match=r"^Invalid edge format\.$"):
        parsing.str_edge_parser("(1)")

    with pytest.raises(ValueError, match=r"^Invalid edge format\.$"):
        parsing.str_edge_parser("(1, two)")

    with pytest.raises((ValueError, SyntaxError)):
        parsing.str_edge_parser("(1,")


def test_calculate_total_terms_empty(parsing):
    assert parsing.calculate_total_terms({}) == 0


def test_calculate_total_terms_sums_over_vertices_and_triplets(parsing):
    current_level_terms = {
        "v0": {"triplets": [{"terms": [1, 2, 3]}, {"terms": []}]},
        "v1": {"triplets": [{"terms": ["a"]}]},
        "v2": {"triplets": [{"terms": [None, None]}]},
    }
    assert parsing.calculate_total_terms(current_level_terms) == 6
