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

from collections.abc import Mapping
from typing import Any

import pytest

from neuralqx.experimental.operators.symbolic.compiler.compiler import (
    SymbolicCompiler,
    compile_symbolic_operator,
)
from neuralqx.experimental.operators.symbolic.compiler.core.pipeline import (
    SymbolicPassPipeline,
)
from neuralqx.experimental.operators.symbolic.compiler.lowering.base import (
    AbstractSymbolicLowerer,
)
from neuralqx.experimental.operators.symbolic.compiler.lowering.registry import (
    SymbolicLowererRegistry,
)
from neuralqx.experimental.operators.symbolic.compiler.passes.base import (
    AbstractSymbolicPass,
)
from neuralqx.experimental.operators.symbolic.core.base import AbstractSymbolicOperator
from neuralqx.experimental.operators.symbolic.core.operator import SymbolicOperator
from neuralqx.experimental.operators.symbolic.core.sum import SymbolicOperatorSum
from neuralqx.experimental.operators.symbolic.ir.expressions import (
    AmplitudeExpr,
    _collect_free_symbols,
)
from neuralqx.experimental.operators.symbolic.ir.predicates import (
    PredicateExpr,
    _collect_free_symbols_pred,
    _render_predicate,
)
from neuralqx.experimental.operators.symbolic.ir.program import SymbolicOperatorIR
from neuralqx.experimental.operators.symbolic.ir.term import (
    KBodyIteratorSpec,
    SymbolicIRTerm,
)
from neuralqx.experimental.operators.symbolic.ir.update import (
    UpdateOp,
    UpdateProgram,
    _collect_free_symbols_from_ops,
)
from neuralqx.utils.errors import SymbolicCompilerError


class _NoopPass(AbstractSymbolicPass):
    @property
    def name(self) -> str:
        return "noop"

    def run(self, context) -> Mapping[str, Any] | None:
        return {"ok": True}


class _BoomPass(AbstractSymbolicPass):
    def __init__(self, pass_name: str) -> None:
        self._name = pass_name

    @property
    def name(self) -> str:
        return self._name

    def run(self, context) -> Mapping[str, Any] | None:
        raise RuntimeError("boom")


class _BoomLowerer(AbstractSymbolicLowerer):
    @property
    def name(self) -> str:
        return "boom_lowerer"

    @property
    def backend(self) -> str:
        return "jax"

    def supports(self, context) -> bool:
        return True

    def lower(self, context):
        raise RuntimeError("lower boom")


class _DummyNonSymbolic(AbstractSymbolicOperator):
    def __init__(self, hilbert):
        super().__init__(
            hilbert, name="dummy_non_symbolic", dtype_str="float64", is_hermitian=True
        )

    def to_ir(self):
        return SymbolicOperatorIR(
            operator_name="dummy_non_symbolic",
            mode="jax_kernel",
            hilbert_size=int(self.hilbert.size),
            dtype_str="float64",
            is_hermitian=True,
            terms=(),
        )

    def _apply_scalar(self, scalar):
        return self


def _simple_op(symbolic, hilbert_tiny, *, name="simple"):
    return (
        symbolic.DOperator(hilbert_tiny, name, hermitian=True)
        .globally()
        .emit(symbolic.identity(), amplitude=1.0)
        .build()
    )


def test_update_ir_rendering_and_free_symbol_collection(symbolic):
    p = PredicateExpr.gt(AmplitudeExpr.symbol("kappa"), 0)
    cond_op = UpdateOp.from_mapping(
        kind="cond_branch",
        params={
            "predicate": p,
            "then_ops": (
                UpdateOp.from_mapping(
                    kind="write_site",
                    params={
                        "site": AmplitudeExpr.constant(0),
                        "value": AmplitudeExpr.symbol("theta"),
                    },
                ),
            ),
            "else_ops": (),
        },
    )
    ops = (
        UpdateOp.from_mapping(
            kind="shift_mod_site",
            params={"site": AmplitudeExpr.constant(0), "delta": 1},
        ),
        UpdateOp.from_mapping(
            kind="affine_site",
            params={"site": AmplitudeExpr.constant(1), "scale": 2, "bias": 3},
        ),
        UpdateOp.from_mapping(
            kind="permute_sites",
            params={"sites": (AmplitudeExpr.constant(0), AmplitudeExpr.constant(1))},
        ),
        UpdateOp.from_mapping(
            kind="scatter",
            params={
                "flat_indices": (0, 1),
                "values": (AmplitudeExpr.symbol("phi"), AmplitudeExpr.constant(0)),
            },
        ),
        cond_op,
    )
    program = UpdateProgram(ops=ops)
    s = str(program)
    assert "wrap" in s
    assert "cond(" in s
    assert "scatter" not in s  # rendered as assignment list
    assert "UpdateProgram(op_count=5)" in repr(program)
    assert "UpdateOp(" in repr(ops[0])

    merged = UpdateProgram().extend(program)
    assert merged.op_count == 5

    free: set[str] = set()
    _collect_free_symbols_from_ops(ops, free)
    assert {"kappa", "theta", "phi"} <= free

    with pytest.raises(ValueError, match="Unsupported update operation kind"):
        UpdateOp(kind="invalid_kind")


def test_amplitude_expr_dunders_render_and_symbol_collection():
    with pytest.raises(ValueError, match="Unsupported amplitude-expression op"):
        AmplitudeExpr(op="not_an_op")

    # _freeze(list) path
    list_const = AmplitudeExpr.constant([1, 2, 3])
    assert list_const.args[0] == (1, 2, 3)

    x = AmplitudeExpr.symbol("site:i:foo")
    y = AmplitudeExpr.symbol("emit:j:bar")
    z = AmplitudeExpr.symbol("free_symbol")

    expr = -((2 + x) - (3 - y)) * (4 / z)
    # Trigger all dunders/renders
    _ = x <= 1
    _ = x >= 1
    s = str(expr)
    assert "x[i].foo" in s
    assert "x'[j].bar" in s

    c = AmplitudeExpr.constant(1 + 2j)
    assert str(c) == repr(1 + 2j)
    assert "AmplitudeExpr(" in repr(c)

    free: set[str] = set()
    nested = AmplitudeExpr(
        op="add", args=(AmplitudeExpr.constant(1), (AmplitudeExpr.symbol("alpha"),))
    )
    _collect_free_symbols(nested, free)
    assert "alpha" in free


def test_predicate_dunders_and_rendering():
    with pytest.raises(ValueError, match="Unsupported predicate-expression op"):
        PredicateExpr(op="bad")

    p0 = PredicateExpr.lt(AmplitudeExpr.symbol("free"), 2)
    p1 = PredicateExpr.ge(AmplitudeExpr.symbol("site:i:value"), -1)
    tree = (~p0) | (p1 & PredicateExpr.constant(True))
    txt = _render_predicate(tree)
    assert "||" in txt or "&&" in txt
    assert "PredicateExpr(" in repr(tree)

    free: set[str] = set()
    _collect_free_symbols_pred(tree, free)
    assert "free" in free


def test_symbolic_sum_internal_branches(symbolic, hilbert_tiny, nk):
    a = _simple_op(symbolic, hilbert_tiny, name="a")
    b = _simple_op(symbolic, hilbert_tiny, name="b")

    with pytest.raises(ValueError, match="at least one term"):
        SymbolicOperatorSum(hilbert_tiny, terms=())

    other_h = nk.hilbert.Spin(s=1 / 2, N=2)
    with pytest.raises(ValueError, match="same Hilbert space"):
        SymbolicOperatorSum(a.hilbert, terms=(a, _DummyNonSymbolic(other_h)))

    s = SymbolicOperatorSum(
        hilbert_tiny,
        terms=(a, b),
        dtype_str="complex64",
        name="sum",
        is_hermitian=False,
    )
    assert len(s) == 2
    assert list(iter(s)) == [a, b]
    assert s.free_symbols == frozenset()
    assert "SymbolicOperatorSum(" in repr(s)

    scaled = 2j * s
    assert scaled.is_hermitian is False
    assert scaled.dtype.name.startswith("complex")

    mixed = SymbolicOperatorSum(
        hilbert_tiny, terms=(a, _DummyNonSymbolic(hilbert_tiny))
    )
    with pytest.raises(ValueError, match="mixed modes"):
        mixed.to_ir()


def test_symbolic_operator_estimate_max_conn_size_type_guard(symbolic, hilbert_tiny):
    term = SymbolicIRTerm.create(
        name="bad_it",
        iterator=object(),  # not a KBodyIteratorSpec
        predicate=True,
        update_program=symbolic.identity().to_program(),
        amplitude=1.0,
    )
    op = SymbolicOperator(hilbert_tiny, "bad_it_op", (term,))
    with pytest.raises(TypeError, match="Unsupported iterator type"):
        op.estimate_max_conn_size()


def test_compiler_error_paths(symbolic, hilbert_tiny):
    op = _simple_op(symbolic, hilbert_tiny, name="compiler_errors")

    class _BadIROp:
        def to_ir(self):
            raise RuntimeError("bad ir")

    with pytest.raises(SymbolicCompilerError, match="Failed to extract IR"):
        SymbolicCompiler().compile(_BadIROp())

    pre_boom = SymbolicCompiler(
        pipeline=SymbolicPassPipeline(
            pre_cache_passes=[_BoomPass("pre_boom")],
            post_cache_passes=[],
        )
    )
    with pytest.raises(SymbolicCompilerError, match="Pre-cache pass failed"):
        pre_boom.compile(op)

    post_boom = SymbolicCompiler(
        pipeline=SymbolicPassPipeline(
            pre_cache_passes=[_NoopPass()],
            post_cache_passes=[_BoomPass("post_boom")],
        ),
        options=symbolic.SymbolicCompilerOptions(cache_enabled=False),
    )
    with pytest.raises(SymbolicCompilerError, match="Post-cache pass failed"):
        post_boom.compile(op)

    reg = SymbolicLowererRegistry()
    reg.register(_BoomLowerer())
    low_boom = SymbolicCompiler(
        lowerer_registry=reg,
        options=symbolic.SymbolicCompilerOptions(cache_enabled=False),
    )
    with pytest.raises(SymbolicCompilerError, match="Lowering failed"):
        low_boom.compile(op)


def test_compile_symbolic_operator_convenience_function(symbolic, hilbert_tiny):
    op = _simple_op(symbolic, hilbert_tiny, name="conv")
    compiled = compile_symbolic_operator(
        op,
        options=symbolic.SymbolicCompilerOptions(cache_enabled=False),
    )
    assert compiled.name == "conv"
