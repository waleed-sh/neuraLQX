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

from tests.operators.helpers import dense_from_get_conn_padded


def _term_op(symbolic, hilbert, *, name, amp=1.0, hermitian=True, fanout=None):
    b = symbolic.DOperator(
        hilbert, name, dtype="float64", hermitian=hermitian
    ).for_each_site("i")
    if fanout is not None:
        b = b.fanout(fanout)
    return b.emit(symbolic.identity(), amplitude=amp).build()


def test_symbolic_operator_addition_merges_terms_and_flags(symbolic, hilbert_tiny):
    a = _term_op(symbolic, hilbert_tiny, name="a", amp=1.0, hermitian=True)
    b = _term_op(symbolic, hilbert_tiny, name="b", amp=2.0, hermitian=False)
    c = a + b

    assert c.term_count == a.term_count + b.term_count
    assert c.is_hermitian is False
    assert "(a + b)" in c.name


def test_symbolic_operator_addition_rejects_conflicting_metadata(
    symbolic, hilbert_tiny
):
    term = _term_op(symbolic, hilbert_tiny, name="base").to_ir().terms[0]
    a = symbolic.SymbolicOperator(hilbert_tiny, "a", (term,), metadata={"k": 1})
    b = symbolic.SymbolicOperator(hilbert_tiny, "b", (term,), metadata={"k": 2})

    with pytest.raises(ValueError, match="Cannot merge symbolic operator metadata"):
        _ = a + b


def test_symbolic_operator_scalar_multiplication_dtype_and_hermiticity(
    symbolic, hilbert_tiny
):
    op = _term_op(symbolic, hilbert_tiny, name="h", amp=1.0, hermitian=True)

    r = 2.0 * op
    assert r.is_hermitian is True
    assert r.dtype == np.dtype("float64")

    c = (1.0 + 2.0j) * op
    assert c.is_hermitian is False
    assert c.dtype == np.dtype("complex128")
    assert "(1+2j" in c.name or "1+2j" in c.name


def test_estimate_max_conn_size_uses_hints_and_emissions(symbolic, hilbert_tiny):
    a = (
        symbolic.DOperator(hilbert_tiny, "hinted")
        .for_each_site("i")
        .fanout(5)
        .emit(symbolic.identity(), amplitude=1.0)
        .build()
    )
    b = (
        symbolic.DOperator(hilbert_tiny, "plain")
        .for_each_site("i")
        .emit(symbolic.identity(), amplitude=1.0)
        .emit(symbolic.identity(), amplitude=2.0)
        .build()
    )
    c = a + b
    assert c.estimate_max_conn_size() == 5 + (hilbert_tiny.size * 2)


def test_symbolic_operator_sum_flattens_and_aggregates_ir(symbolic, hilbert_tiny):
    a = _term_op(symbolic, hilbert_tiny, name="a")
    b = _term_op(symbolic, hilbert_tiny, name="b")
    c = _term_op(symbolic, hilbert_tiny, name="c")

    nested = symbolic.SymbolicOperatorSum(
        hilbert_tiny,
        terms=(a, symbolic.SymbolicOperatorSum(hilbert_tiny, terms=(b, c))),
    )
    assert len(nested.terms) == 3
    ir = nested.to_ir()
    md = ir.metadata_dict()
    assert "child_ir_fingerprints" in md
    assert len(md["child_ir_fingerprints"]) == 3


def test_symbolic_operator_sum_compile_matches_sum_of_parts(symbolic, hilbert_tiny):
    a = _term_op(symbolic, hilbert_tiny, name="a", amp=1.5)
    b = _term_op(symbolic, hilbert_tiny, name="b", amp=-0.25)
    s = symbolic.SymbolicOperatorSum(hilbert_tiny, terms=(a, b))

    ca = a.compile(cache=False)
    cb = b.compile(cache=False)
    cs = symbolic.SymbolicCompiler(
        options=symbolic.SymbolicCompilerOptions(cache_enabled=False)
    ).compile_operator(s)

    ma = dense_from_get_conn_padded(ca)
    mb = dense_from_get_conn_padded(cb)
    ms = dense_from_get_conn_padded(cs)
    assert np.allclose(ms, ma + mb)


def test_uncompiled_symbolic_operator_execution_guard(symbolic, hilbert_tiny):
    op = _term_op(symbolic, hilbert_tiny, name="guard")
    x = np.asarray(hilbert_tiny.all_states(), dtype=np.int64)
    with pytest.raises(Exception, match="cannot execute before compilation"):
        op.get_conn_padded(x)


def test_compiled_operator_tree_flatten_roundtrip(symbolic, hilbert_tiny):
    import jax

    op = _term_op(symbolic, hilbert_tiny, name="tree").compile(cache=False)
    leaves, treedef = jax.tree_util.tree_flatten(op)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.name == op.name
    assert rebuilt.dtype == op.dtype
    assert rebuilt.is_hermitian == op.is_hermitian


def test_symbolic_operator_repr_and_free_symbols(symbolic, hilbert_tiny):
    op = (
        symbolic.DOperator(hilbert_tiny, "free")
        .globally()
        .emit(symbolic.identity(), amplitude=symbolic.symbol("lambda"))
        .build()
    )
    assert op.free_symbols == frozenset({"lambda"})
    r = repr(op)
    assert "SymbolicOperator" in r
    assert "terms=1" in r
