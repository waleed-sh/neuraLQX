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

import numpy as np
import pytest

from neuralqx.experimental.operators.symbolic.compiler.cache.store import (
    InMemorySymbolicArtifactStore,
)
from neuralqx.experimental.operators.symbolic.compiler.compiler import SymbolicCompiler
from neuralqx.experimental.operators.symbolic.compiler.core.artifact import (
    SymbolicCompiledArtifact,
)
from neuralqx.experimental.operators.symbolic.compiler.core.context import (
    SymbolicCompilationContext,
)
from neuralqx.experimental.operators.symbolic.compiler.core.options import (
    SymbolicCompilerOptions,
)
from neuralqx.experimental.operators.symbolic.compiler.core.pass_report import (
    SymbolicPassReport,
)
from neuralqx.experimental.operators.symbolic.compiler.core.pipeline import (
    SymbolicPassPipeline,
)
from neuralqx.experimental.operators.symbolic.compiler.core.signature import (
    SymbolicCompilationSignature,
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
from neuralqx.experimental.operators.symbolic.compiler.passes.fusion import (
    SymbolicFusionPass,
)
from neuralqx.experimental.operators.symbolic.core.base import AbstractSymbolicOperator
from neuralqx.experimental.operators.symbolic.core.operator import (
    SymbolicOperator,
    _merge_metadata as _op_merge_metadata,
    _promote_dtype_for_scalar as _op_promote_dtype_for_scalar,
    _resolve_dtype as _op_resolve_dtype,
    _scalar_str as _op_scalar_str,
)
from neuralqx.experimental.operators.symbolic.core.sum import (
    SymbolicOperatorSum,
    _resolve_dtype as _sum_resolve_dtype,
)
from neuralqx.experimental.operators.symbolic.dsl import op as op_mod
from neuralqx.experimental.operators.symbolic.dsl.selectors import SiteSelector
from neuralqx.experimental.operators.symbolic.ir.expressions import (
    AmplitudeExpr,
    _render_amplitude,
)
from neuralqx.experimental.operators.symbolic.ir.predicates import (
    PredicateExpr,
    _render_predicate,
)
from neuralqx.experimental.operators.symbolic.ir.program import SymbolicOperatorIR
from neuralqx.experimental.operators.symbolic.ir.term import (
    KBodyIteratorSpec,
    SymbolicIRTerm,
)
from neuralqx.experimental.operators.symbolic.ir.update import (
    UpdateOp,
    _render_update_op,
)
from neuralqx.experimental.operators.symbolic.ir.validate import validate_symbolic_ir


def _simple_op(symbolic, hilbert_tiny, *, name: str = "edge_case"):
    return (
        symbolic.DOperator(hilbert_tiny, name, hermitian=True)
        .globally()
        .emit(symbolic.identity(), amplitude=1.0)
        .build()
    )


class _NoopPass(AbstractSymbolicPass):
    def __init__(self, pass_name: str) -> None:
        self._name = pass_name

    @property
    def name(self) -> str:
        return self._name

    def run(self, context) -> Mapping[str, Any] | None:
        return None


class _DummyLowerer(AbstractSymbolicLowerer):
    @property
    def name(self) -> str:
        return "dummy_lowerer"

    @property
    def backend(self) -> str:
        return "jax"

    def supports(self, context: SymbolicCompilationContext) -> bool:
        return True

    def lower(self, context: SymbolicCompilationContext):
        return SymbolicCompiledArtifact.create(
            operator_name=context.ir.operator_name,
            backend="jax",
            lowerer_name=self.name,
            compiled_operator=object(),
        )


class _MiniSym(AbstractSymbolicOperator):
    def __init__(
        self, hilbert, name: str = "mini", *, dtype_str: str = "float64"
    ) -> None:
        super().__init__(
            hilbert,
            name=name,
            dtype_str=dtype_str,
            is_hermitian=True,
        )

    def to_ir(self):
        from neuralqx.experimental.operators.symbolic.ir.update import UpdateProgram

        term = SymbolicIRTerm.create(
            name="t",
            iterator=KBodyIteratorSpec(labels=("i",), index_sets=((0,),)),
            predicate=True,
            update_program=UpdateProgram(),
            amplitude=1.0,
        )
        return SymbolicOperatorIR.from_terms(
            operator_name=self.operator_name,
            hilbert_size=int(self.hilbert.size),
            dtype_str="float64",
            is_hermitian=True,
            terms=(term,),
        )

    def _apply_scalar(self, scalar):
        return self


class _NoEstimateSym(_MiniSym):
    pass


def test_options_and_pass_report_edge_branches():
    with pytest.raises(ValueError, match="Unsupported backend_preference"):
        SymbolicCompilerOptions(backend_preference="numba")
    with pytest.raises(ValueError, match="cache_namespace"):
        SymbolicCompilerOptions(cache_namespace="   ")

    opts = SymbolicCompilerOptions.from_mapping(debug_flags=None)
    assert opts.debug_flags == ()
    assert opts.debug_flag_map() == {}

    rep = SymbolicPassReport.create(pass_name="p", duration_ms=1.25, metadata=None)
    assert rep.metadata_map() == {}
    assert "SymbolicPassReport(" in repr(rep)


def test_pipeline_registry_and_store_error_repr_paths():
    with pytest.raises(ValueError, match="at least one pre-cache pass"):
        SymbolicPassPipeline(pre_cache_passes=(), post_cache_passes=())

    pipeline = SymbolicPassPipeline(
        pre_cache_passes=(_NoopPass("pre"),),
        post_cache_passes=(_NoopPass("post"),),
    )
    assert pipeline.pre_cache_passes[0].name == "pre"
    assert pipeline.post_cache_passes[0].name == "post"
    assert "SymbolicPassPipeline(" in repr(pipeline)

    reg = SymbolicLowererRegistry()
    with pytest.raises(TypeError, match="Expected AbstractSymbolicLowerer"):
        reg.register(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Expected AbstractSymbolicLowerer"):
        reg.register_first(object())  # type: ignore[arg-type]
    reg.register(_DummyLowerer())
    assert "SymbolicLowererRegistry(" in repr(reg)

    with pytest.raises(ValueError, match="positive integer"):
        InMemorySymbolicArtifactStore(max_entries=0)
    store = InMemorySymbolicArtifactStore(max_entries=1)
    assert "InMemorySymbolicArtifactStore(" in repr(store)


def test_context_signature_artifact_and_compiler_properties(symbolic, hilbert_tiny):
    op = _simple_op(symbolic, hilbert_tiny, name="ctxsig")
    ctx = SymbolicCompilationContext(
        operator=op,
        ir=op.to_ir(),
        options=SymbolicCompilerOptions(),
    )
    ctx.set_metadata("alpha", 1)
    ctx.set_analysis("fanout", 3)
    ctx.set_selected_backend("jax")
    ctx.set_selected_lowerer("jax_symbolic_v1")

    assert ctx.metadata["alpha"] == 1
    assert ctx.selected_lowerer == "jax_symbolic_v1"
    assert ctx.analyses["fanout"] == 3
    with pytest.raises(ValueError, match="Required compiler analysis"):
        ctx.require_analysis("missing")

    summary = ctx.summary()
    assert summary["operator_name"] == "ctxsig"
    assert summary["pass_count"] == 0

    sig = SymbolicCompilationSignature.from_context(ctx)
    key = sig.build_cache_key(namespace="nqx_symbolic_v1", extension_context={"x": 1})
    assert str(key) == key.token
    assert sig.as_dict()["backend_target"] == "jax"
    assert "SymbolicCompilationSignature(" in repr(sig)

    art = SymbolicCompiledArtifact.create(
        operator_name="x",
        backend="jax",
        lowerer_name="lx",
        compiled_operator=object(),
        metadata=None,
    )
    assert art.cache_token() is None
    assert art.metadata_map() == {}
    assert "SymbolicCompiledArtifact(" in repr(art)

    compiler = SymbolicCompiler()
    assert isinstance(compiler.pass_names, tuple)
    assert isinstance(compiler.lowerer_names, tuple)
    assert "SymbolicCompiler(" in repr(compiler)


def test_fusion_pass_disabled_singletons(symbolic, hilbert_tiny):
    op = (
        symbolic.DOperator(hilbert_tiny, "fusion_off")
        .for_each_site("i")
        .emit(symbolic.identity(), amplitude=1.0)
        .emit(symbolic.identity(), amplitude=2.0)
        .build()
    )
    ctx = SymbolicCompilationContext(
        operator=op,
        ir=op.to_ir(),
        options=SymbolicCompilerOptions(enable_fusion=False),
    )
    meta = SymbolicFusionPass().run(ctx)
    assert meta["fusion_enabled"] is False
    assert ctx.analysis("fusion_groups") == [[t.name] for t in op.to_ir().terms]


def test_core_base_and_operator_internal_branches(symbolic, hilbert_tiny):
    with pytest.raises(ValueError, match="non-empty string"):
        _MiniSym(hilbert_tiny, name="  ")

    a = _MiniSym(hilbert_tiny, name="a")
    b = _MiniSym(hilbert_tiny, name="b")
    assert a.operator_name == "a"
    assert "name='a'" in repr(a)

    summed = a + b
    assert isinstance(summed, SymbolicOperatorSum)

    with pytest.raises(NotImplementedError, match="does not implement _apply_scalar"):
        AbstractSymbolicOperator._apply_scalar(a, 2)

    assert a.__mul__("x") is NotImplemented
    assert (-a) is a

    assert _op_resolve_dtype("int32", "int16") == "int32"
    assert _op_scalar_str(1.25) == repr(1.25)
    assert _op_promote_dtype_for_scalar("float32", 1j) == "complex64"
    assert _op_merge_metadata({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    term = SymbolicIRTerm.create(
        name="kbody_no_hint",
        iterator=KBodyIteratorSpec(labels=("i",), index_sets=((0,), (1,), (2,))),
        predicate=True,
        update_program=symbolic.identity().to_program(),
        amplitude=1.0,
        fanout_hint=None,
    )
    op = SymbolicOperator(hilbert_tiny, "kbody", (term,))
    assert op.estimate_max_conn_size() == 3
    assert op.__add__(object()) is NotImplemented

    other = SymbolicOperator(hilbert_tiny, "other", (term,))
    assert op.__radd__(other).term_count == 2
    with pytest.raises(Exception):
        _ = op.__radd__(0)


def test_symbolic_sum_fallback_dtype_and_fanout_branch(hilbert_tiny):
    x = _NoEstimateSym(hilbert_tiny, name="x", dtype_str="int32")
    y = _NoEstimateSym(hilbert_tiny, name="y", dtype_str="int16")
    assert _sum_resolve_dtype((x, y), explicit=None) == "complex64"

    total = SymbolicOperatorSum(hilbert_tiny, terms=(x,))
    assert total.estimate_max_conn_size() == int(hilbert_tiny.size)


def test_dsl_operator_private_paths_and_repr(symbolic, hilbert_tiny, nk):
    wrapped_amp_term = SymbolicIRTerm.create(
        name="wrap_amp",
        iterator=KBodyIteratorSpec(labels=(), index_sets=((),)),
        predicate=True,
        update_program=symbolic.identity().to_program(),
        amplitude=AmplitudeExpr.wrap_mod(AmplitudeExpr.constant(1)),
    )
    assert op_mod._terms_use_shift_mod((wrapped_amp_term,)) is True

    class _NoStates:
        local_states = None

    class _BadShape:
        local_states = np.asarray([[0, 1]])

    class _BadType:
        local_states = np.asarray([0.0, 0.5])

    with pytest.raises(ValueError, match="local_states=None"):
        op_mod._infer_shift_mod_spec_from_hilbert(_NoStates())
    with pytest.raises(ValueError, match="non-empty 1D sequence"):
        op_mod._infer_shift_mod_spec_from_hilbert(_BadShape())
    with pytest.raises(ValueError, match="integer local_states"):
        op_mod._infer_shift_mod_spec_from_hilbert(_BadType())

    with pytest.raises(ValueError, match="non-empty string"):
        symbolic.DOperator(hilbert_tiny, "   ")

    tri = (
        symbolic.DOperator(hilbert_tiny, "tri")
        .for_each_triplet("a", "b", "c", over=((0, 1, 2),))
        .emit()
        .build()
    )
    plaq = (
        symbolic.DOperator(hilbert_tiny, "plaq")
        .for_each_plaquette("a", "b", "c", "d", over=((0, 1, 2, 0),))
        .emit()
        .build()
    )
    assert tri.to_ir().terms[0].iterator.labels == ("a", "b", "c")
    assert plaq.to_ir().terms[0].iterator.labels == ("a", "b", "c", "d")

    builder = symbolic.DOperator(
        nk.hilbert.Spin(s=1 / 2, N=2), "repr_case"
    ).for_each_site("i")
    r_open = repr(builder)
    assert "term_open=True" in r_open
    _ = builder.emit().build()
    r_closed = repr(builder)
    assert "terms_sealed=1" in r_closed


def test_selectors_and_ir_rendering_fallbacks():
    with pytest.raises(ValueError, match="namespace must be a non-empty string"):
        SiteSelector("i", namespace=" ")

    sel = SiteSelector("i")
    assert sel.index.args[0] == "site:i:index"
    assert sel.attr("custom").args[0] == "site:i:custom"
    assert sel.custom.args[0] == "site:i:custom"

    a = AmplitudeExpr.symbol("emit:i:value")
    b = AmplitudeExpr.symbol("emit:i:index")
    assert str(a) == "x'[i]"
    assert str(b) == "i"
    assert "sqrt(" in str(AmplitudeExpr.sqrt(4))
    assert "conj(" in str(AmplitudeExpr.conj(1 + 2j))
    assert "|" in str(AmplitudeExpr.abs_(3))
    assert "wrap(" in str(AmplitudeExpr.wrap_mod(5))
    assert str(AmplitudeExpr.static_emitted_index(2)) == "x'[2]"
    assert "^" in str(AmplitudeExpr.pow(2, 3))
    _ = 2 * AmplitudeExpr.symbol("alpha")
    assert _render_amplitude(AmplitudeExpr.constant(2 + 0j)) in {"2.0", "2"}

    fake_amp = object.__new__(AmplitudeExpr)
    object.__setattr__(fake_amp, "op", "mystery")
    object.__setattr__(fake_amp, "args", (AmplitudeExpr.constant(1),))
    assert _render_amplitude(fake_amp).startswith("mystery(")

    p = PredicateExpr.eq(AmplitudeExpr.symbol("x"), 0)
    assert PredicateExpr.__rand__(p, PredicateExpr.constant(True)).op == "and"
    assert PredicateExpr.__ror__(p, PredicateExpr.constant(False)).op == "or"

    fake_pred = object.__new__(PredicateExpr)
    object.__setattr__(fake_pred, "op", "mystery")
    object.__setattr__(fake_pred, "args", (PredicateExpr.constant(True),))
    assert _render_predicate(fake_pred).startswith("mystery(")


def test_ir_program_term_update_and_validation_edges(symbolic, hilbert_tiny):
    from neuralqx.experimental.operators.symbolic.ir import program as program_mod
    from neuralqx.experimental.operators.symbolic.ir.update import UpdateProgram

    assert isinstance(
        program_mod._serialize_predicate((PredicateExpr.constant(True),)), list
    )
    assert isinstance(program_mod._serialize_update(object()), str)

    term = SymbolicIRTerm.create(
        name="no_free",
        iterator=KBodyIteratorSpec(labels=("i",), index_sets=((0,),)),
        predicate=True,
        update_program=symbolic.identity().to_program(),
        amplitude=1.0,
    )
    ir = SymbolicOperatorIR.from_terms(
        operator_name="ir_dump",
        hilbert_size=2,
        dtype_str="float64",
        is_hermitian=True,
        terms=(term,),
    )
    assert "; 1 term(s)" in str(ir)
    assert "SymbolicOperatorIR(" in repr(ir)

    with pytest.raises(ValueError, match="non-empty string"):
        SymbolicOperatorIR(
            operator_name=" ",
            mode="symbolic",
            hilbert_size=1,
            dtype_str="float64",
            is_hermitian=True,
            terms=(),
        )
    with pytest.raises(ValueError, match="Unsupported IR mode"):
        SymbolicOperatorIR(
            operator_name="bad",
            mode="bad_mode",
            hilbert_size=1,
            dtype_str="float64",
            is_hermitian=True,
            terms=(),
        )

    g = KBodyIteratorSpec(labels=(), index_sets=((),))
    p = KBodyIteratorSpec(
        labels=("i", "j"), index_sets=((0, 1), (1, 2), (2, 0), (0, 2))
    )
    assert g._format_iterate_line() == "globally"
    assert "+1 more" in p._format_iterate_line(max_shown=3)
    assert "for (i, j) in" in p._format_iterate_line(max_shown=10)
    assert "KBodyIteratorSpec(" in repr(p)

    with pytest.raises(ValueError, match="non-empty string"):
        SymbolicIRTerm.create(
            name=" ",
            iterator=p,
            predicate=True,
            update_program=UpdateProgram(),
            amplitude=1.0,
        )
    with pytest.raises(ValueError, match="positive integer"):
        SymbolicIRTerm.create(
            name="bad_fanout",
            iterator=p,
            predicate=True,
            update_program=UpdateProgram(),
            amplitude=1.0,
            fanout_hint=0,
        )

    meta_term = SymbolicIRTerm.create(
        name="meta",
        iterator=p,
        predicate=True,
        update_program=UpdateProgram(),
        amplitude=1.0,
        metadata={"k": 1},
    )
    assert meta_term.metadata_dict() == {"k": 1}
    assert "term " in str(meta_term)
    assert "SymbolicIRTerm(" in repr(meta_term)

    cond = UpdateOp.from_mapping(
        kind="cond_branch",
        params={"predicate": True, "then_ops": (), "else_ops": ()},
    )
    assert "cond(" in str(cond)
    fake_update = object.__new__(UpdateOp)
    object.__setattr__(fake_update, "kind", "mystery")
    object.__setattr__(fake_update, "params", ())
    assert "UpdateOp(kind='mystery'" in _render_update_op(fake_update)

    nonsym = SymbolicOperatorIR(
        operator_name="kernel",
        mode="jax_kernel",
        hilbert_size=1,
        dtype_str="float64",
        is_hermitian=True,
        terms=(),
    )
    assert validate_symbolic_ir(nonsym)["term_count"] == 0

    empty_symbolic = SymbolicOperatorIR(
        operator_name="empty",
        mode="symbolic",
        hilbert_size=1,
        dtype_str="float64",
        is_hermitian=False,
        terms=(),
    )
    with pytest.raises(ValueError, match="has no terms"):
        validate_symbolic_ir(empty_symbolic)

    class _PairLike:
        kind = "pair"
        label_a = "i"
        label_b = "j"
        index_sets = ((0, 1),)

    pair_term = SymbolicIRTerm.create(
        name="pair_scope",
        iterator=_PairLike(),
        predicate=PredicateExpr.and_(
            PredicateExpr.constant(True),
            PredicateExpr.eq(AmplitudeExpr.symbol("site:j:value"), 0),
        ),
        update_program=symbolic.identity().to_program(),
        amplitude=1.0,
    )
    pair_ir = SymbolicOperatorIR.from_terms(
        operator_name="pair_scope_ir",
        hilbert_size=int(hilbert_tiny.size),
        dtype_str="float64",
        is_hermitian=True,
        terms=(pair_term,),
    )
    summary = validate_symbolic_ir(pair_ir)
    assert summary["term_symbols"]["pair_scope"] == ("site:j:value",)
