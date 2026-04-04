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

from tests.operators.helpers import dense_from_get_conn_padded


def build_symbolic_sigmaz(symbolic, hilbert, site_idx: int):
    i = int(site_idx)
    return (
        symbolic.DOperator(
            hilbert, f"symbolic_sigmaz_{i}", dtype="float64", hermitian=True
        )
        .for_each(("i",), over=((i,),))
        .named("z")
        .fanout(1)
        .emit(symbolic.identity(), amplitude=symbolic.site("i").value)
        .build()
    )


def _safe_spin_flip_update(symbolic, *, label: str, delta: int):
    pred = symbolic.site(label) < 0 if delta > 0 else symbolic.site(label) > 0
    update = symbolic.Update.cond(
        pred,
        if_true=symbolic.shift(label, delta),
        if_false=symbolic.identity(),
    )
    return pred, update


def build_symbolic_sigmap(symbolic, hilbert, site_idx: int):
    i = int(site_idx)
    pred, update = _safe_spin_flip_update(symbolic, label="i", delta=2)
    return (
        symbolic.DOperator(
            hilbert, f"symbolic_sigmap_{i}", dtype="complex128", hermitian=False
        )
        .for_each(("i",), over=((i,),))
        .named("sp")
        .where(pred)
        .fanout(1)
        .emit(update, amplitude=1.0)
        .build()
    )


def build_symbolic_sigmam(symbolic, hilbert, site_idx: int):
    i = int(site_idx)
    pred, update = _safe_spin_flip_update(symbolic, label="i", delta=-2)
    return (
        symbolic.DOperator(
            hilbert, f"symbolic_sigmam_{i}", dtype="complex128", hermitian=False
        )
        .for_each(("i",), over=((i,),))
        .named("sm")
        .where(pred)
        .fanout(1)
        .emit(update, amplitude=1.0)
        .build()
    )


def compile_no_cache(op):
    return op.compile(cache=False)


def dense_ket_action(op) -> np.ndarray:
    return np.asarray(dense_from_get_conn_padded(op))


def dense_as_local(op) -> np.ndarray:
    return dense_ket_action(op).T


def nonzero_conn_multiset(hilbert, xp_row: np.ndarray, mel_row: np.ndarray):
    xp_row = np.asarray(xp_row)
    mel_row = np.asarray(mel_row)
    nz = np.abs(mel_row) > 1e-12
    if not np.any(nz):
        return []
    nums = np.asarray(hilbert.states_to_numbers(xp_row[nz]), dtype=np.int64)
    mels = mel_row[nz]
    out = [(int(n), complex(m)) for n, m in zip(nums, mels)]
    return sorted(out, key=lambda t: (t[0], round(t[1].real, 12), round(t[1].imag, 12)))
