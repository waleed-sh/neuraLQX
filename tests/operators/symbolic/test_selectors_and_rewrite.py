#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import pytest


def test_site_selector_construction_and_namespace(symbolic):
    i = symbolic.site("i")
    e = symbolic.emitted("i")

    assert i.label == "i"
    assert i.namespace == "site"
    assert e.namespace == "emit"
    assert i.value.op == "symbol"
    assert e.value.op == "symbol"
    assert i.as_site_ref().args[0] == "site:i:index"

    with pytest.raises(ValueError, match="non-empty string"):
        symbolic.site("  ")


def test_site_selector_predicate_overloads(symbolic):
    i = symbolic.site("i")
    assert (i < 1).op == "lt"
    assert (i <= 1).op == "le"
    assert (i > 1).op == "gt"
    assert (i >= 1).op == "ge"
    assert (i == 1).op == "eq"
    assert (i != 1).op == "ne"
    assert i.abs().op == "abs_"


def test_expression_context_boolean_helpers(symbolic):
    ctx = symbolic.ExpressionContext()
    p0 = ctx.lt(ctx.const(0), ctx.const(1))
    p1 = ctx.eq(ctx.const(2), ctx.const(2))
    assert ctx.all_of(p0, p1).op == "and"
    assert ctx.any_of(p0, p1).op == "or"
    assert ctx.not_(p0).op == "not"


def test_expression_context_scalar_helpers_and_coercions(symbolic):
    ctx = symbolic.ExpressionContext()
    assert ctx.symbol("lambda").op == "symbol"
    assert ctx.sqrt(4).op == "sqrt"
    assert ctx.conj(1 + 2j).op == "conj"
    assert ctx.neg(3).op == "neg"
    assert ctx.site("i").namespace == "site"
    assert ctx.emitted("i").namespace == "emit"
    assert ctx.eq(1, 1).op == "eq"
    assert ctx.ne(1, 2).op == "ne"
    assert ctx.lt(1, 2).op == "lt"
    assert ctx.le(1, 2).op == "le"
    assert ctx.gt(2, 1).op == "gt"
    assert ctx.ge(2, 1).op == "ge"
    assert ctx.coerce_amplitude(7).op == "const"
    assert ctx.coerce_predicate(True).op == "const"
    assert ctx.pow(2, 3).op == "pow"
    assert ctx.abs_(-3).op == "abs_"
    assert ctx.wrap_mod(5).op == "wrap_mod"


def test_update_builder_immutability(symbolic):
    u0 = symbolic.identity()
    u1 = u0.shift("i", 1)
    u2 = u1.write("i", 0)

    assert u0.to_program().op_count == 0
    assert u1.to_program().op_count == 1
    assert u2.to_program().op_count == 2


def test_update_scatter_requires_matching_lengths(symbolic):
    with pytest.raises(ValueError, match="same length"):
        symbolic.scatter([0, 1], [1])


def test_update_cond_defaults_else_identity(symbolic):
    pred = symbolic.site("i") > 0
    c = symbolic.Update.cond(pred, if_true=symbolic.write("i", 1))
    op = c.to_program().ops[0]
    assert op.kind == "cond_branch"
    assert op.get("predicate").op == "gt"
    assert op.get("else_ops") == ()


def test_module_level_update_factories(symbolic):
    assert symbolic.shift("i", 1).to_program().ops[0].kind == "shift_site"
    assert symbolic.shift_mod("i", 1).to_program().ops[0].kind == "shift_mod_site"
    assert symbolic.write("i", 0).to_program().ops[0].kind == "write_site"
    assert symbolic.swap("i", "j").to_program().ops[0].kind == "swap_sites"
    assert symbolic.permute("i", "j").to_program().ops[0].kind == "permute_sites"
    assert (
        symbolic.affine("i", scale=2, bias=-1).to_program().ops[0].kind == "affine_site"
    )


def test_site_ref_coercion_paths_and_errors(symbolic):
    s = symbolic.site("i")
    a = symbolic.AmplitudeExpr.constant(0)
    assert symbolic.shift(s, 1).to_program().ops[0].get("site").op == "symbol"
    assert symbolic.shift(a, 1).to_program().ops[0].get("site").op == "const"
    with pytest.raises(TypeError, match="site reference must be"):
        symbolic.shift(object(), 1)


def test_symbol_factory_non_empty(symbolic):
    s = symbolic.symbol("kappa")
    assert s.op == "symbol"
    assert s.args[0] == "kappa"
    with pytest.raises(ValueError, match="non-empty"):
        symbolic.symbol("   ")


def test_site_selector_attr_guards_and_repr(symbolic):
    s = symbolic.site("i")
    assert "SiteSelector" in repr(s)
    with pytest.raises(ValueError, match="attribute name must be non-empty"):
        s.attr("   ")


def test_update_eq_with_non_update_returns_notimplemented(symbolic):
    u = symbolic.identity()
    assert u.__eq__(object()) is NotImplemented
