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


def _compile_global(
    symbolic, hilbert, update, *, amplitude=1.0, name="op", hermitian=False
):
    return (
        symbolic.DOperator(hilbert, name, hermitian=hermitian)
        .globally()
        .emit(update, amplitude=amplitude)
        .compile(cache=False)
    )


@pytest.mark.parametrize(
    ("update_factory", "x", "expected"),
    [
        (
            lambda s: s.write(0, 1),
            np.array([-1, 0, 1], dtype=np.int64),
            np.array([1, 0, 1], dtype=np.int64),
        ),
        (
            lambda s: s.shift(1, +1),
            np.array([-1, 0, 1], dtype=np.int64),
            np.array([-1, 1, 1], dtype=np.int64),
        ),
        (
            lambda s: s.swap(0, 2),
            np.array([-1, 0, 1], dtype=np.int64),
            np.array([1, 0, -1], dtype=np.int64),
        ),
        (
            lambda s: s.permute(0, 1, 2),
            np.array([-1, 0, 1], dtype=np.int64),
            np.array([0, 1, -1], dtype=np.int64),
        ),
        (
            lambda s: s.affine(2, scale=-1, bias=0),
            np.array([-1, 0, 1], dtype=np.int64),
            np.array([-1, 0, -1], dtype=np.int64),
        ),
        (
            lambda s: s.scatter([0, 2], [1, -1]),
            np.array([-1, 0, 1], dtype=np.int64),
            np.array([1, 0, -1], dtype=np.int64),
        ),
    ],
)
def test_update_primitives_runtime_semantics(
    symbolic, hilbert_spin1, update_factory, x, expected
):
    compiled = _compile_global(
        symbolic, hilbert_spin1, update_factory(symbolic), amplitude=1.0
    )
    xp, mels = compiled.get_conn_padded(x[None, :])
    assert np.array_equal(np.asarray(xp[0, 0]), expected)
    assert np.asarray(mels[0, 0]) == pytest.approx(1.0)


def test_conditional_update_runtime_branching(symbolic, hilbert_fock_one):
    pred = symbolic.AmplitudeExpr.static_index(0) > 0
    update = symbolic.Update.cond(
        pred,
        if_true=symbolic.write(0, 2),
        if_false=symbolic.write(0, 0),
    )
    compiled = _compile_global(
        symbolic, hilbert_fock_one, update, amplitude=1.0, name="cond"
    )

    x = np.asarray([[0], [1], [2]], dtype=np.int64)
    xp, mels = compiled.get_conn_padded(x)
    assert np.array_equal(np.asarray(xp[:, 0, 0]), np.array([0, 2, 2], dtype=np.int64))
    assert np.allclose(np.asarray(mels[:, 0]), 1.0)


def test_emitted_selector_reads_connected_state(symbolic, hilbert_fock_one):
    pred = symbolic.site("i") < 2
    update = symbolic.Update.cond(
        pred,
        if_true=symbolic.shift("i", +1),
        if_false=symbolic.identity(),
    )
    compiled = (
        symbolic.DOperator(hilbert_fock_one, "emit_amp")
        .for_each_site("i")
        .where(pred)
        .emit(update, amplitude=symbolic.emitted("i").value)
        .compile(cache=False)
    )

    x = np.asarray([[0], [1]], dtype=np.int64)
    xp, mels = compiled.get_conn_padded(x)
    assert np.array_equal(np.asarray(xp[:, 0, 0]), np.array([1, 2], dtype=np.int64))
    assert np.array_equal(np.asarray(mels[:, 0]), np.array([1, 2], dtype=np.complex64))


def test_multi_emission_preserves_branch_multiset_duplicates(
    symbolic, hilbert_fock_one
):
    compiled = (
        symbolic.DOperator(hilbert_fock_one, "dups")
        .globally()
        .emit(symbolic.identity(), amplitude=1.0, tag="a")
        .emit(symbolic.identity(), amplitude=2.0, tag="b")
        .compile(cache=False)
    )

    x = np.asarray([[1]], dtype=np.int64)
    xp, mels = compiled.get_conn_padded(x)
    xp = np.asarray(xp[0])
    mels = np.asarray(mels[0])
    assert xp.shape == (2, 1)
    assert np.array_equal(xp[0], xp[1])
    assert np.array_equal(mels, np.asarray([1.0, 2.0], dtype=mels.dtype))


def test_invalidate_branch_zeroes_matrix_elements(symbolic, hilbert_fock_one):
    compiled = (
        symbolic.DOperator(hilbert_fock_one, "invalidate")
        .globally()
        .emit(symbolic.identity().invalidate(reason="test"), amplitude=7.0)
        .emit(symbolic.identity(), amplitude=3.0)
        .compile(cache=False)
    )
    x = np.asarray([[2]], dtype=np.int64)
    xp, mels = compiled.get_conn_padded(x)
    xp = np.asarray(xp[0])
    mels = np.asarray(mels[0])
    assert xp.shape[0] == 2
    assert mels[0] == 0
    assert mels[1] == pytest.approx(3.0)


def test_fanout_controls_padding_and_truncation(symbolic, hilbert_tiny):
    x = np.asarray(hilbert_tiny.all_states(), dtype=np.int64)

    natural = (
        symbolic.DOperator(hilbert_tiny, "fanout_natural")
        .for_each_site("i")
        .emit(symbolic.identity(), amplitude=1.0)
        .emit(symbolic.identity(), amplitude=2.0)
        .compile(cache=False)
    )
    xp_n, m_n = natural.get_conn_padded(x)
    assert xp_n.shape[1] == hilbert_tiny.size * 2

    truncated = (
        symbolic.DOperator(hilbert_tiny, "fanout_trunc")
        .for_each_site("i")
        .fanout(2)
        .emit(symbolic.identity(), amplitude=1.0)
        .emit(symbolic.identity(), amplitude=2.0)
        .compile(cache=False)
    )
    xp_t, m_t = truncated.get_conn_padded(x)
    assert xp_t.shape[1] == 2
    assert m_t.shape[1] == 2

    padded = (
        symbolic.DOperator(hilbert_tiny, "fanout_pad")
        .for_each_site("i")
        .fanout(8)
        .emit(symbolic.identity(), amplitude=1.0)
        .emit(symbolic.identity(), amplitude=2.0)
        .compile(cache=False)
    )
    xp_p, m_p = padded.get_conn_padded(x)
    assert xp_p.shape[1] == 8
    m_p = np.asarray(m_p)
    assert np.allclose(m_p[:, 6:], 0.0)


def test_shift_mod_runtime_wraps_on_fock_hilbert(symbolic, hilbert_fock_one):
    compiled = (
        symbolic.DOperator(hilbert_fock_one, "shift_mod")
        .for_each_site("i")
        .emit(symbolic.shift_mod("i", 1), amplitude=1.0)
        .compile(cache=False)
    )
    x = np.asarray(hilbert_fock_one.all_states(), dtype=np.int64)
    xp, mels = compiled.get_conn_padded(x)
    assert np.array_equal(np.asarray(xp[:, 0, 0]), np.array([1, 2, 0], dtype=np.int64))
    assert np.allclose(np.asarray(mels[:, 0]), 1.0)


def test_get_conn_padded_accepts_single_and_batched(symbolic, hilbert_fock_one):
    compiled = _compile_global(
        symbolic,
        hilbert_fock_one,
        symbolic.identity(),
        amplitude=2.0,
        name="single_vs_batch",
    )
    x_single = np.asarray([1], dtype=np.int64)
    x_batch = np.asarray([[1], [2]], dtype=np.int64)

    xp_s, m_s = compiled.get_conn_padded(x_single)
    xp_b, m_b = compiled.get_conn_padded(x_batch)

    assert xp_s.shape == (1, 1)
    assert m_s.shape == (1,)
    assert xp_b.shape == (2, 1, 1)
    assert m_b.shape == (2, 1)
