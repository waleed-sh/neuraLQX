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

from tests.operators.symbolic.helpers import compile_no_cache


def test_operator_requires_open_term_before_term_methods(symbolic, hilbert_tiny):
    op = symbolic.DOperator(hilbert_tiny, "must_open")

    with pytest.raises(ValueError, match="before any iterator method"):
        op.where(True)
    with pytest.raises(ValueError, match="before any iterator method"):
        op.emit(symbolic.identity(), amplitude=1.0)
    with pytest.raises(ValueError, match="before any iterator method"):
        op.named("name")
    with pytest.raises(ValueError, match="before any iterator method"):
        op.fanout(1)


def test_operator_build_requires_terms_and_emissions(symbolic, hilbert_tiny):
    with pytest.raises(ValueError, match="zero terms"):
        symbolic.DOperator(hilbert_tiny, "empty").build()

    op = symbolic.DOperator(hilbert_tiny, "no_emit").for_each_site("i")
    with pytest.raises(ValueError, match="has no emissions"):
        op.build()


def test_for_each_and_annotation_validations(symbolic, hilbert_tiny):
    with pytest.raises(ValueError, match="must not be empty"):
        symbolic.DOperator(hilbert_tiny, "bad").for_each(("i",), over=())

    with pytest.raises(ValueError, match="must have length 2"):
        symbolic.DOperator(hilbert_tiny, "bad").for_each(("i", "j"), over=((0,),))

    with pytest.raises(ValueError, match="at least 2"):
        symbolic.permute("i")

    with pytest.raises(ValueError, match="positive integer"):
        (symbolic.DOperator(hilbert_tiny, "bad_fanout").for_each_site("i").fanout(0))

    with pytest.raises(ValueError, match="non-empty string"):
        (symbolic.DOperator(hilbert_tiny, "bad_name").for_each_site("i").named("   "))


def test_where_composes_with_logical_and(symbolic, hilbert_tiny):
    op = (
        symbolic.DOperator(hilbert_tiny, "where_and")
        .for_each_pair("i", "j")
        .where(symbolic.site("i") > -1)
        .where(symbolic.site("j") < 1)
        .emit(symbolic.identity(), amplitude=1.0)
        .build()
    )
    pred = op.to_ir().terms[0].predicate
    assert pred.op == "and"
    assert len(pred.args) == 2


def test_emit_accepts_updateprogram_and_callable_amplitude(symbolic, hilbert_tiny):
    update_program = symbolic.shift("i", 0).to_program()
    op = (
        symbolic.DOperator(hilbert_tiny, "callable_amp")
        .for_each_site("i")
        .emit(
            update_program,
            amplitude=lambda ctx: ctx.sqrt(ctx.site("i").value + 1),
            tag="branch0",
        )
        .build()
    )
    term = op.to_ir().terms[0]
    assert term.effective_emissions[0].amplitude.op == "sqrt"
    assert term.effective_emissions[0].branch_tag == "branch0"


def test_emit_rejects_invalid_update_type(symbolic, hilbert_tiny):
    with pytest.raises(TypeError, match="Expected Update or UpdateProgram"):
        (
            symbolic.DOperator(hilbert_tiny, "bad_emit")
            .for_each_site("i")
            .emit(1234, amplitude=1.0)
            .build()
        )


def test_compile_shortcut_returns_executable_operator(symbolic, hilbert_tiny):
    compiled = (
        symbolic.DOperator(hilbert_tiny, "shortcut", hermitian=True)
        .for_each_site("i")
        .emit(symbolic.identity(), amplitude=1.0)
        .compile(cache=False)
    )

    x = np.asarray(hilbert_tiny.all_states(), dtype=np.int64)
    xp, mels = compiled.get_conn_padded(x)
    assert xp.shape == (x.shape[0], hilbert_tiny.size, hilbert_tiny.size)
    assert mels.shape == (x.shape[0], hilbert_tiny.size)


def test_shift_mod_metadata_inferred_for_contiguous_local_states(
    symbolic, hilbert_fock_one
):
    op = (
        symbolic.DOperator(hilbert_fock_one, "shift_mod_meta")
        .for_each_site("i")
        .emit(symbolic.shift_mod("i", 1), amplitude=1.0)
        .build()
    )
    meta = op.metadata
    assert "shift_mod_spec" in meta
    spec = meta["shift_mod_spec"]
    assert spec["version"] == "uniform_integer_wrap_v1"
    assert spec["state_min"] == 0
    assert spec["mod_span"] == 3
    assert spec["local_states"] == (0, 1, 2)


def test_shift_mod_build_rejects_non_contiguous_local_states(symbolic, hilbert_tiny):
    with pytest.raises(ValueError, match="contiguous unit-spaced local_states"):
        (
            symbolic.DOperator(hilbert_tiny, "shift_mod_bad")
            .for_each_site("i")
            .emit(symbolic.shift_mod("i", 1), amplitude=1.0)
            .build()
        )


def test_expression_context_extended_helpers(symbolic):
    ctx = symbolic.ExpressionContext()

    with pytest.raises(ValueError, match="at least one component"):
        ctx.sq_norm()

    sq = ctx.sq_norm(ctx.const(1), ctx.const(2), ctx.const(3))
    n2 = ctx.norm2(ctx.const(1), ctx.const(2), ctx.const(2))
    edge = ctx.edge_value(edge_idx=1, gauge_copy=2, n_edges_per_copy=4)
    emitted_edge = ctx.emitted_edge_value(edge_idx=1, gauge_copy=2, n_edges_per_copy=4)

    assert sq.op == "add"
    assert n2.op == "sqrt"
    assert edge.op == "static_index" and edge.args[0] == 9
    assert emitted_edge.op == "static_emitted_index" and emitted_edge.args[0] == 9


def test_update_identity_singleton_hash_and_repr(symbolic):
    u0 = symbolic.identity()
    u1 = symbolic.identity()
    assert u0 is u1
    assert u0 == u1
    assert hash(u0) == hash(u1)
    assert "Update" in repr(u0)


def test_compiled_operator_name_dtype_and_repr(symbolic, hilbert_tiny):
    op = (
        symbolic.DOperator(hilbert_tiny, "repr_test", dtype="complex128")
        .globally()
        .emit(symbolic.identity(), amplitude=1.0)
        .build()
    )
    compiled = compile_no_cache(op)
    assert compiled.name == "repr_test"
    assert compiled.dtype == np.dtype("complex128")
    assert "CompiledOperator" in repr(compiled)
