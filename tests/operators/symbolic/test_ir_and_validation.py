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

import numpy as np
import pytest

from neuralqx.experimental.operators.symbolic.ir.term import _scale_ir_term
from neuralqx.experimental.operators.symbolic.ir.validate import validate_symbolic_ir


def _base_term(symbolic):
    return symbolic.SymbolicIRTerm.create(
        name="t0",
        iterator=symbolic.KBodyIteratorSpec(labels=("i",), index_sets=((0,),)),
        predicate=symbolic.site("i") > -10,
        update_program=symbolic.identity().to_program(),
        amplitude=symbolic.site("i").value + symbolic.symbol("alpha"),
    )


def test_amplitude_expr_guards_and_coercion(symbolic):
    A = symbolic.AmplitudeExpr

    with pytest.raises(TypeError, match="Boolean values are not valid"):
        A.constant(True)
    with pytest.raises(ValueError, match="non-empty"):
        A.symbol("   ")
    with pytest.raises(ValueError, match="non-negative"):
        A.static_index(-1)
    with pytest.raises(ValueError, match="non-negative"):
        A.static_emitted_index(-1)
    with pytest.raises(TypeError, match="Cannot coerce"):
        symbolic.coerce_amplitude_expr(object())

    assert symbolic.coerce_amplitude_expr(2).op == "const"
    assert symbolic.coerce_amplitude_expr(3.5).op == "const"
    assert symbolic.coerce_amplitude_expr(2 + 3j).op == "const"
    assert symbolic.coerce_amplitude_expr("kappa").op == "symbol"


def test_predicate_expr_guards_and_coercion(symbolic):
    P = symbolic.PredicateExpr

    assert P.and_().op == "const" and P.and_().args[0] is True
    assert P.or_().op == "const" and P.or_().args[0] is False
    assert symbolic.coerce_predicate_expr(True).op == "const"

    with pytest.raises(TypeError, match="explicit comparison"):
        symbolic.coerce_predicate_expr(symbolic.AmplitudeExpr.constant(1))
    with pytest.raises(TypeError, match="Cannot coerce"):
        symbolic.coerce_predicate_expr(123)


def test_update_program_helpers_and_rendering(symbolic):
    up = (
        symbolic.shift("i", 1)
        .write("i", 0)
        .swap("i", 0)
        .invalidate(reason="boundary")
        .to_program()
    )
    assert up.op_count == 4
    assert up.has_invalidate()
    text = str(up)
    assert "x'[i] = (x[i] + 1)" in text
    assert "invalidate  ; boundary" in text


def test_term_free_symbols_and_effective_emissions(symbolic):
    t = _base_term(symbolic)
    assert t.free_symbols == frozenset({"alpha"})
    em = t.effective_emissions
    assert len(em) == 1
    assert em[0].amplitude.op == "add"


def test_symbolic_operator_ir_fingerprint_stability(symbolic):
    t0 = _base_term(symbolic)
    ir0 = symbolic.SymbolicOperatorIR.from_terms(
        operator_name="op",
        hilbert_size=4,
        dtype_str="complex64",
        is_hermitian=False,
        terms=(t0,),
    )
    ir1 = symbolic.SymbolicOperatorIR.from_terms(
        operator_name="op",
        hilbert_size=4,
        dtype_str="complex64",
        is_hermitian=False,
        terms=(t0,),
    )
    t2 = symbolic.SymbolicIRTerm.create(
        name="t1",
        iterator=t0.iterator,
        predicate=t0.predicate,
        update_program=t0.update_program,
        amplitude=t0.amplitude,
    )
    ir2 = symbolic.SymbolicOperatorIR.from_terms(
        operator_name="op",
        hilbert_size=4,
        dtype_str="complex64",
        is_hermitian=False,
        terms=(t2,),
    )

    assert ir0.static_fingerprint() == ir1.static_fingerprint()
    assert ir0.static_fingerprint() != ir2.static_fingerprint()
    d = ir0.as_dict()
    assert d["operator_name"] == "op"
    assert len(d["terms"]) == 1


def test_validate_symbolic_ir_accepts_well_scoped_symbols(symbolic):
    ir = symbolic.SymbolicOperatorIR.from_terms(
        operator_name="valid",
        hilbert_size=2,
        dtype_str="complex64",
        is_hermitian=False,
        terms=(_base_term(symbolic),),
    )
    summary = validate_symbolic_ir(ir)
    assert summary["mode"] == "symbolic"
    assert summary["term_count"] == 1


def test_validate_symbolic_ir_rejects_unbound_site_label(symbolic):
    bad_term = symbolic.SymbolicIRTerm.create(
        name="bad_label",
        iterator=symbolic.KBodyIteratorSpec(labels=("i",), index_sets=((0,),)),
        predicate=symbolic.AmplitudeExpr.symbol("site:j:value") > 0,
        update_program=symbolic.identity().to_program(),
        amplitude=1.0,
    )
    ir = symbolic.SymbolicOperatorIR.from_terms(
        operator_name="invalid",
        hilbert_size=2,
        dtype_str="complex64",
        is_hermitian=False,
        terms=(bad_term,),
    )
    with pytest.raises(ValueError, match="not bound by its iterator"):
        validate_symbolic_ir(ir)


def test_validate_symbolic_ir_rejects_unknown_namespace(symbolic):
    bad_term = symbolic.SymbolicIRTerm.create(
        name="bad_ns",
        iterator=symbolic.KBodyIteratorSpec(labels=("i",), index_sets=((0,),)),
        predicate=True,
        update_program=symbolic.identity().to_program(),
        amplitude=symbolic.AmplitudeExpr.symbol("foo:i:value"),
    )
    ir = symbolic.SymbolicOperatorIR.from_terms(
        operator_name="invalid",
        hilbert_size=2,
        dtype_str="complex64",
        is_hermitian=False,
        terms=(bad_term,),
    )
    with pytest.raises(ValueError, match="unknown namespace"):
        validate_symbolic_ir(ir)


def test_validate_symbolic_ir_rejects_site_symbol_in_global_term(symbolic):
    bad_term = symbolic.SymbolicIRTerm.create(
        name="bad_global",
        iterator=symbolic.KBodyIteratorSpec(labels=(), index_sets=((),)),
        predicate=True,
        update_program=symbolic.identity().to_program(),
        amplitude=symbolic.AmplitudeExpr.symbol("site:i:value"),
    )
    ir = symbolic.SymbolicOperatorIR.from_terms(
        operator_name="invalid",
        hilbert_size=2,
        dtype_str="complex64",
        is_hermitian=False,
        terms=(bad_term,),
    )
    with pytest.raises(ValueError, match="not bound by its iterator|global iterator"):
        validate_symbolic_ir(ir)


def test_validate_symbolic_ir_rejects_missing_update_parameters(symbolic):
    bad_update = symbolic.UpdateProgram(
        ops=(
            symbolic.UpdateOp.from_mapping(
                kind="shift_site", params={"site": symbolic.AmplitudeExpr.constant(0)}
            ),
        )
    )
    bad_term = symbolic.SymbolicIRTerm.create(
        name="bad_update",
        iterator=symbolic.KBodyIteratorSpec(labels=("i",), index_sets=((0,),)),
        predicate=True,
        update_program=bad_update,
        amplitude=1.0,
    )
    ir = symbolic.SymbolicOperatorIR.from_terms(
        operator_name="invalid",
        hilbert_size=2,
        dtype_str="complex64",
        is_hermitian=False,
        terms=(bad_term,),
    )
    with pytest.raises(ValueError, match="requires a 'delta' parameter"):
        validate_symbolic_ir(ir)


def test_scale_ir_term_scales_all_emissions(symbolic):
    e0 = symbolic.EmissionSpec(
        update_program=symbolic.identity().to_program(),
        amplitude=symbolic.AmplitudeExpr.constant(1.0),
        branch_tag="a",
    )
    e1 = symbolic.EmissionSpec(
        update_program=symbolic.identity().to_program(),
        amplitude=symbolic.AmplitudeExpr.constant(2.0),
        branch_tag="b",
    )
    term = symbolic.SymbolicIRTerm.create(
        name="multi",
        iterator=symbolic.KBodyIteratorSpec(labels=("i",), index_sets=((0,),)),
        predicate=True,
        update_program=e0.update_program,
        amplitude=e0.amplitude,
        emissions=(e0, e1),
    )
    scaled = _scale_ir_term(term, symbolic.AmplitudeExpr.constant(3.0))
    amps = [em.amplitude for em in scaled.effective_emissions]
    assert all(a.op == "mul" for a in amps)
    assert len(amps) == 2


def test_ir_string_dump_contains_terms_and_free_symbols(symbolic):
    ir = symbolic.SymbolicOperatorIR.from_terms(
        operator_name="dump",
        hilbert_size=4,
        dtype_str="complex64",
        is_hermitian=False,
        terms=(_base_term(symbolic),),
    )
    s = str(ir)
    assert 'symbolic.operator @"dump"' in s
    assert "term #0" in s
    assert "%alpha" in s


def test_ir_requires_positive_hilbert_and_non_empty_terms(symbolic):
    with pytest.raises(ValueError, match="at least one term"):
        symbolic.SymbolicOperatorIR.from_terms(
            operator_name="x",
            hilbert_size=2,
            dtype_str="complex64",
            is_hermitian=False,
            terms=(),
        )

    with pytest.raises(ValueError, match="positive integer"):
        symbolic.SymbolicOperatorIR(
            operator_name="x",
            mode="symbolic",
            hilbert_size=0,
            dtype_str="complex64",
            is_hermitian=False,
            terms=(),
        )
