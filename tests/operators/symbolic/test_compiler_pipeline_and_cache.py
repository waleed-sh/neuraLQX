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


def _simple_symbolic_op(symbolic, hilbert_tiny, *, name="simple"):
    return (
        symbolic.DOperator(hilbert_tiny, name, hermitian=True)
        .globally()
        .emit(symbolic.identity(), amplitude=1.0)
        .build()
    )


def test_default_pipeline_and_registry_contract(symbolic):
    pipeline = symbolic.default_symbolic_pass_pipeline()
    assert pipeline.pass_names() == (
        "symbolic_validation",
        "symbolic_normalization",
        "symbolic_fanout_analysis",
        "symbolic_fusion_planning",
    )

    registry = symbolic.default_symbolic_lowerer_registry()
    assert "jax_symbolic_v1" in registry.lowerer_names
    assert len(registry) >= 1


def test_symbolic_compiler_cache_hit_reuses_artifact(symbolic, hilbert_tiny):
    op = _simple_symbolic_op(symbolic, hilbert_tiny, name="cache_hit")
    compiler = symbolic.SymbolicCompiler()
    compiler.clear_cache()

    a0 = compiler.compile(op)
    assert a0.cache_key is not None
    assert compiler.cache_size == 1
    assert len(a0.pass_reports) == 4

    a1 = compiler.compile(op)
    assert a1 is a0
    assert compiler.cache_size == 1


def test_compile_with_cache_disabled_has_no_cache_key(symbolic, hilbert_tiny):
    op = _simple_symbolic_op(symbolic, hilbert_tiny, name="cache_off")
    compiler = symbolic.SymbolicCompiler(
        options=symbolic.SymbolicCompilerOptions(cache_enabled=False)
    )
    artifact = compiler.compile(op)
    assert artifact.cache_key is None
    assert artifact.backend == "jax"
    assert artifact.lowerer_name == "jax_symbolic_v1"


def test_inmemory_store_eviction_and_invalidation(symbolic, hilbert_tiny):
    store = symbolic.compiler.cache.store.InMemorySymbolicArtifactStore(max_entries=1)
    op = _simple_symbolic_op(symbolic, hilbert_tiny, name="store")
    compiler = symbolic.SymbolicCompiler(
        options=symbolic.SymbolicCompilerOptions(cache_enabled=False)
    )
    artifact = compiler.compile(op)

    k0 = symbolic.SymbolicCacheKey(token="0", namespace="n")
    k1 = symbolic.SymbolicCacheKey(token="1", namespace="n")
    store.put(k0, artifact)
    assert len(store) == 1
    assert store.get(k0) is artifact

    store.put(k1, artifact)
    assert len(store) == 1
    assert store.get(k0) is None
    assert store.get(k1) is artifact

    assert store.invalidate(k0) is False
    assert store.invalidate(k1) is True
    assert len(store) == 0


def test_lowerer_registry_resolution_and_failure(symbolic, hilbert_tiny):
    op = _simple_symbolic_op(symbolic, hilbert_tiny, name="resolve")
    ctx = symbolic.SymbolicCompilationContext(
        operator=op,
        ir=op.to_ir(),
        options=symbolic.SymbolicCompilerOptions(),
    )
    symbolic.compiler.passes.normalization.SymbolicNormalizationPass().run(ctx)

    reg = symbolic.default_symbolic_lowerer_registry()
    lowerer = reg.resolve(ctx)
    assert lowerer.name == "jax_symbolic_v1"

    empty = symbolic.compiler.lowering.registry.SymbolicLowererRegistry()
    with pytest.raises(RuntimeError, match="No registered symbolic lowerer"):
        empty.resolve(ctx)


def test_compilation_signature_changes_with_options(symbolic, hilbert_tiny):
    op = _simple_symbolic_op(symbolic, hilbert_tiny, name="sig")
    ctx0 = symbolic.SymbolicCompilationContext(
        operator=op,
        ir=op.to_ir(),
        options=symbolic.SymbolicCompilerOptions(enable_fusion=True),
    )
    ctx1 = symbolic.SymbolicCompilationContext(
        operator=op,
        ir=op.to_ir(),
        options=symbolic.SymbolicCompilerOptions(enable_fusion=False),
    )
    symbolic.compiler.passes.normalization.SymbolicNormalizationPass().run(ctx0)
    symbolic.compiler.passes.normalization.SymbolicNormalizationPass().run(ctx1)

    s0 = symbolic.SymbolicCompilationSignature.from_context(ctx0)
    s1 = symbolic.SymbolicCompilationSignature.from_context(ctx1)
    k0 = s0.build_cache_key(namespace="nqx_symbolic_v1")
    k1 = s1.build_cache_key(namespace="nqx_symbolic_v1")
    assert k0.token != k1.token


def test_validation_pass_respects_strict_flag(symbolic):
    bad_term = symbolic.SymbolicIRTerm.create(
        name="bad",
        iterator=symbolic.KBodyIteratorSpec(labels=("i",), index_sets=((0,),)),
        predicate=True,
        update_program=symbolic.identity().to_program(),
        amplitude=symbolic.AmplitudeExpr.symbol("foo:i:value"),
    )
    ir = symbolic.SymbolicOperatorIR.from_terms(
        operator_name="bad_ir",
        hilbert_size=2,
        dtype_str="complex64",
        is_hermitian=False,
        terms=(bad_term,),
    )

    p = symbolic.compiler.passes.validation.SymbolicValidationPass()

    strict_ctx = symbolic.SymbolicCompilationContext(
        operator=object(),
        ir=ir,
        options=symbolic.SymbolicCompilerOptions(strict_validation=True),
    )
    with pytest.raises(ValueError, match="unknown namespace"):
        p.run(strict_ctx)

    non_strict_ctx = symbolic.SymbolicCompilationContext(
        operator=object(),
        ir=ir,
        options=symbolic.SymbolicCompilerOptions(strict_validation=False),
    )
    meta = p.run(non_strict_ctx)
    assert meta["valid"] is False
    summary = non_strict_ctx.analysis("validation_summary")
    assert summary["valid"] is False
    assert "error" in summary


def test_analysis_and_fusion_passes_write_expected_payloads(symbolic, hilbert_tiny):
    t0 = (
        symbolic.DOperator(hilbert_tiny, "a")
        .for_each_site("i")
        .fanout(5)
        .emit(symbolic.identity(), amplitude=1.0)
        .build()
        .to_ir()
        .terms[0]
    )
    t1 = (
        symbolic.DOperator(hilbert_tiny, "b")
        .for_each_site("i")
        .emit(symbolic.identity(), amplitude=1.0)
        .build()
        .to_ir()
        .terms[0]
    )
    t2 = symbolic.SymbolicIRTerm.create(
        name="c",
        iterator=t1.iterator,
        predicate=True,
        update_program=symbolic.identity().invalidate().to_program(),
        amplitude=1.0,
    )

    ir = symbolic.SymbolicOperatorIR.from_terms(
        operator_name="fusion_case",
        hilbert_size=hilbert_tiny.size,
        dtype_str="complex64",
        is_hermitian=False,
        terms=(t0, t1, t2),
    )
    ctx = symbolic.SymbolicCompilationContext(
        operator=object(),
        ir=ir,
        options=symbolic.SymbolicCompilerOptions(enable_fusion=True),
    )

    fanout_meta = symbolic.compiler.passes.analysis.SymbolicFanoutAnalysisPass().run(
        ctx
    )
    assert fanout_meta["total_fanout"] == 5 + hilbert_tiny.size + hilbert_tiny.size

    fusion_meta = symbolic.compiler.passes.fusion.SymbolicFusionPass().run(ctx)
    assert fusion_meta["fusion_enabled"] is True
    groups = ctx.analysis("fusion_groups")
    assert sum(len(g) for g in groups) == 3


def test_compiler_options_and_signature_repr(symbolic):
    opts = symbolic.SymbolicCompilerOptions.from_mapping(
        backend_preference="auto",
        enable_fusion=False,
        strict_validation=True,
        cache_enabled=True,
        cache_namespace="ns",
        debug_flags={"trace": 1, "dump": True},
    )
    sig = opts.static_signature()
    assert ("cache_namespace", "ns") in sig
    assert "SymbolicCompilerOptions" in repr(opts)
