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


from __future__ import annotations

import re
import warnings
from contextlib import contextmanager

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import neuralqx as nqx

from neuralqx.utils.errors import ExpectationValueError

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:.*ConcretizationTypeError is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:.*linear_util\\.wrap_init is missing a DebugInfo object.*:Warning"
    ),
    pytest.mark.filterwarnings(
        "ignore:.*For performance reasons, we suggest to use a power-of-two chunk size.*:Warning"
    ),
]


@contextmanager
def _set_chunk_size(vstate, chunk_size):
    old = getattr(vstate, "chunk_size", None)
    vstate.chunk_size = chunk_size
    try:
        yield
    finally:
        vstate.chunk_size = old


def _flattened_n_samples(vstate) -> int:
    σ = vstate.samples
    if σ is None:
        raise RuntimeError("Tests assume samples are cached (mcstate fixture).")
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)
    return int(σ.shape[0])


def _valid_chunk_sizes(vstate) -> list[int]:

    n = _flattened_n_samples(vstate)
    candidates = [n // 4, n // 2, n]
    out = []
    for c in candidates:
        if c > 0 and n % c == 0:
            out.append(int(c))
    # de-dup preserving order
    seen = set()
    out2 = []
    for c in out:
        if c not in seen:
            out2.append(c)
            seen.add(c)
    return out2


def _stats_mean(stats):
    return jnp.asarray(stats.Mean)


def _assert_stats_close(a, b, *, rtol=1e-6, atol=1e-7):
    assert jnp.allclose(_stats_mean(a), _stats_mean(b), rtol=rtol, atol=atol)
    assert jnp.allclose(
        jnp.asarray(a.Sigma), jnp.asarray(b.Sigma), rtol=rtol, atol=atol
    )


def _tree_allclose(a, b, *, rtol=1e-6, atol=1e-7):
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb, "Pytree structure mismatch"
    for xa, xb in zip(la, lb):
        assert jnp.allclose(xa, xb, rtol=rtol, atol=atol), (xa, xb)


_CHUNK_IGNORE_RE = re.compile(r"Ignoring chunk_size=")


def _capture_warnings(
    fn, *args, **kwargs
) -> tuple[object, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, w


def _relevant_messages(w) -> list[str]:
    return [str(ww.message) for ww in w]


def _assert_no_chunk_ignored(w):
    msgs = _relevant_messages(w)
    assert not any(
        _CHUNK_IGNORE_RE.search(m) for m in msgs
    ), f"Unexpected chunk-ignore warnings: {msgs}"


def _assert_chunk_ignored(w, *, match: str | None = None):
    msgs = _relevant_messages(w)
    hit = [m for m in msgs if _CHUNK_IGNORE_RE.search(m)]
    assert len(hit) >= 1, f"Expected a chunk-ignore warning, got: {msgs}"
    if match is not None:
        assert any(
            re.search(match, m) for m in hit
        ), f"Chunk-ignore warning did not match /{match}/. Got: {hit}"


def test_chunk_size_non_divisor_fails_at_runtime_expect(mcstate, ops_spin_2):

    sx0, _, _ = ops_spin_2
    n = _flattened_n_samples(mcstate)

    bad = n - 1
    assert n % bad != 0

    with pytest.raises(ValueError):
        with _set_chunk_size(mcstate, bad):
            mcstate.expect(sx0)


@pytest.mark.parametrize("chunk_size", [None, "VALID"])
def test_expect_chunked_matches_unchunked_and_dispatches(
    chunk_size, mcstate, ops_spin_2, monkeypatch
):
    sx0, sz0, _ = ops_spin_2
    op = sx0

    if chunk_size == "VALID":
        chunk_size = _valid_chunk_sizes(mcstate)[0]

    import neuralqx.vqs.mc.mc_state.expect_chunked as mod

    called = {"count": 0}
    orig = mod._expect_chunking

    def wrapped(*args, **kwargs):
        called["count"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(mod, "_expect_chunking", wrapped)

    with _set_chunk_size(mcstate, None):
        stats_ref, w_ref = _capture_warnings(mcstate.expect, op)

    with _set_chunk_size(mcstate, chunk_size):
        stats_out, w_out = _capture_warnings(mcstate.expect, op)

    _assert_stats_close(stats_out, stats_ref)

    if chunk_size is None:
        assert called["count"] == 0
    else:
        assert called["count"] >= 1, "Chunked expect entrypoint was not called."
        _assert_no_chunk_ignored(w_out)


def test_expect_sequence_chunked_dispatches_sequence_kernel(
    mcstate, ops_spin_2, monkeypatch
):
    sx0, sz0, _ = ops_spin_2
    ops = [sx0, sz0]
    cs = _valid_chunk_sizes(mcstate)[0]

    import neuralqx.vqs.mc.mc_state.expect_chunked as mod

    called = {"count": 0}
    orig = mod._expect_sequence_chunked

    def wrapped(*args, **kwargs):
        called["count"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(mod, "_expect_sequence_chunked", wrapped)

    with _set_chunk_size(mcstate, None):
        stats_ref, _ = _capture_warnings(mcstate.expect, ops)

    with _set_chunk_size(mcstate, cs):
        stats_chk, w = _capture_warnings(mcstate.expect, ops)

    assert called["count"] >= 1, "Chunked expect(sequence) kernel was not called."
    _assert_no_chunk_ignored(w)
    _assert_stats_close(stats_chk, stats_ref)


def test_expect_penaltycost_chunked_supported(mcstate, ops_spin_2):
    sx0, sz0, _ = ops_spin_2
    pc = nqx.operators.PenaltyCost(sx0, factor=0.5)
    cs = _valid_chunk_sizes(mcstate)[0]

    with _set_chunk_size(mcstate, None):
        stats_ref, _ = _capture_warnings(mcstate.expect, [pc, sz0])

    with _set_chunk_size(mcstate, cs):
        stats_chk, w = _capture_warnings(mcstate.expect, [pc, sz0])

    _assert_no_chunk_ignored(w)
    _assert_stats_close(stats_chk, stats_ref)


def test_expect_inverse_expectation_cost_is_not_chunked_warns_and_matches(
    mcstate, ops_spin_2
):
    _, sz0, _ = ops_spin_2
    iec = nqx.operators.InverseExpectationCost(sz0, factor=0.7, alpha=0.2)
    cs = _valid_chunk_sizes(mcstate)[0]

    with _set_chunk_size(mcstate, None):
        stats_ref, _ = _capture_warnings(mcstate.expect, iec)

    with _set_chunk_size(mcstate, cs):
        stats_chk, w = _capture_warnings(mcstate.expect, iec)

    _assert_chunk_ignored(w, match=r"chunking is not supported")
    _assert_stats_close(stats_chk, stats_ref)


def test_expect_and_forces_sequence_chunked_uses_chunked_vjp_and_matches(
    mcstate, ops_spin_2, monkeypatch
):

    sx0, _, _ = ops_spin_2
    ops = [sx0]
    cs = _valid_chunk_sizes(mcstate)[0]

    import neuralqx.vqs.mc.mc_state.expect_forces_chunked as mod_forces
    import netket.jax as nkjax

    called = {"seq_chunked": 0, "vjp_chunked": 0}

    orig_seq = mod_forces.forces_expect_hermitian_sequence_chunked

    def wrapped_seq(*args, **kwargs):
        called["seq_chunked"] += 1
        return orig_seq(*args, **kwargs)

    monkeypatch.setattr(
        mod_forces, "forces_expect_hermitian_sequence_chunked", wrapped_seq
    )

    orig_vjp = nkjax.vjp_chunked

    def wrapped_vjp(*args, **kwargs):
        called["vjp_chunked"] += 1
        return orig_vjp(*args, **kwargs)

    monkeypatch.setattr(nkjax, "vjp_chunked", wrapped_vjp)

    with _set_chunk_size(mcstate, None):
        (stats_ref, f_ref), _ = _capture_warnings(
            mcstate.expect_and_forces, ops, mutable=False
        )

    with _set_chunk_size(mcstate, cs):
        with jax.disable_jit():
            (stats_chk, f_chk), w = _capture_warnings(
                mcstate.expect_and_forces, ops, mutable=False
            )

    assert called["seq_chunked"] >= 1, "Chunked sequence forces kernel was not called."
    assert called["vjp_chunked"] >= 1, "nkjax.vjp_chunked was not called."
    _assert_no_chunk_ignored(w)
    _assert_stats_close(stats_chk, stats_ref)
    _tree_allclose(f_chk, f_ref)


def test_expect_and_forces_sequence_chunked_mixed_ops_matches(mcstate, ops_spin_2):
    sx0, sz0, _ = ops_spin_2
    ops = [sx0, nqx.operators.PenaltyCost(sz0, factor=0.25)]
    cs = _valid_chunk_sizes(mcstate)[0]

    with _set_chunk_size(mcstate, None):
        (stats_ref, f_ref), _ = _capture_warnings(
            mcstate.expect_and_forces, ops, mutable=False
        )

    with _set_chunk_size(mcstate, cs):
        (stats_chk, f_chk), w = _capture_warnings(
            mcstate.expect_and_forces, ops, mutable=False
        )

    _assert_no_chunk_ignored(w)
    _assert_stats_close(stats_chk, stats_ref)
    _tree_allclose(f_chk, f_ref)


def test_expect_and_forces_sequence_chunked_iec_is_not_chunked_warns_and_matches(
    mcstate, ops_spin_2, monkeypatch
):

    _, sz0, _ = ops_spin_2
    ops = [nqx.operators.InverseExpectationCost(sz0, factor=0.7, alpha=0.2)]
    cs = _valid_chunk_sizes(mcstate)[0]

    import neuralqx.vqs.mc.mc_state.expect_forces_chunked as mod_forces
    import netket.jax as nkjax

    called = {"seq_chunked": 0, "vjp_chunked": 0}

    orig_seq = mod_forces.forces_expect_hermitian_sequence_chunked

    def wrapped_seq(*args, **kwargs):
        called["seq_chunked"] += 1
        return orig_seq(*args, **kwargs)

    monkeypatch.setattr(
        mod_forces, "forces_expect_hermitian_sequence_chunked", wrapped_seq
    )

    orig_vjp = nkjax.vjp_chunked

    def wrapped_vjp(*args, **kwargs):
        called["vjp_chunked"] += 1
        return orig_vjp(*args, **kwargs)

    monkeypatch.setattr(nkjax, "vjp_chunked", wrapped_vjp)

    with _set_chunk_size(mcstate, None):
        (stats_ref, f_ref), _ = _capture_warnings(
            mcstate.expect_and_forces, ops, mutable=False
        )

    with _set_chunk_size(mcstate, cs):
        with jax.disable_jit():
            (stats_chk, f_chk), w = _capture_warnings(
                mcstate.expect_and_forces, ops, mutable=False
            )

    _assert_chunk_ignored(w, match=r"chunking is not supported")
    assert (
        called["seq_chunked"] == 0
    ), "IEC forces should not call chunked sequence kernel."
    assert called["vjp_chunked"] == 0, "IEC forces should not use vjp_chunked."
    _assert_stats_close(stats_chk, stats_ref)
    _tree_allclose(f_chk, f_ref)


def test_expect_and_grad_sequence_chunked_uses_chunked_forces_under_the_hood(
    mcstate, ops_spin_2, monkeypatch
):
    sx0, sz0, _ = ops_spin_2
    ops = [sx0, sz0]
    cs = _valid_chunk_sizes(mcstate)[0]

    import neuralqx.vqs.mc.mc_state.expect_forces_chunked as mod_forces

    called = {"seq_chunked": 0}
    orig_seq = mod_forces.forces_expect_hermitian_sequence_chunked

    def wrapped_seq(*args, **kwargs):
        called["seq_chunked"] += 1
        return orig_seq(*args, **kwargs)

    monkeypatch.setattr(
        mod_forces, "forces_expect_hermitian_sequence_chunked", wrapped_seq
    )

    with _set_chunk_size(mcstate, None):
        (stats_ref, g_ref), _ = _capture_warnings(
            mcstate.expect_and_grad, ops, mutable=False
        )

    with _set_chunk_size(mcstate, cs):
        with jax.disable_jit():
            (stats_chk, g_chk), w = _capture_warnings(
                mcstate.expect_and_grad, ops, mutable=False
            )

    assert (
        called["seq_chunked"] >= 1
    ), "Chunked sequence forces was not used inside expect_and_grad."
    _assert_no_chunk_ignored(w)
    _assert_stats_close(stats_chk, stats_ref)
    _tree_allclose(g_chk, g_ref)


def test_expect_and_grad_squared_operator_chunking_is_ignored_warns_and_matches(
    mcstate, ops_spin_2, nk
):

    sx0, _, _ = ops_spin_2
    sq = nk.operator.Squared(sx0)
    cs = _valid_chunk_sizes(mcstate)[0]

    with _set_chunk_size(mcstate, None):
        (stats_ref, g_ref), _ = _capture_warnings(
            mcstate.expect_and_grad, sq, mutable=False
        )

    with _set_chunk_size(mcstate, cs):
        (stats_chk, g_chk), w = _capture_warnings(
            mcstate.expect_and_grad, sq, mutable=False
        )

    _assert_chunk_ignored(w, match=r"expect_and_grad_nonhermitian|expect_and_grad")
    _assert_stats_close(stats_chk, stats_ref)
    _tree_allclose(g_chk, g_ref)


def test_chunked_expect_empty_sequence_raises(mcstate):
    cs = _valid_chunk_sizes(mcstate)[0]
    with _set_chunk_size(mcstate, cs):
        with pytest.raises(ExpectationValueError):
            mcstate.expect([])


def test_chunked_expect_and_forces_empty_sequence_raises(mcstate):
    cs = _valid_chunk_sizes(mcstate)[0]
    with _set_chunk_size(mcstate, cs):
        with pytest.raises(ExpectationValueError):
            mcstate.expect_and_forces([], mutable=False)


def test_chunked_expect_and_grad_empty_sequence_raises(mcstate):
    cs = _valid_chunk_sizes(mcstate)[0]
    with _set_chunk_size(mcstate, cs):
        with pytest.raises(ExpectationValueError):
            mcstate.expect_and_grad([], mutable=False)
