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

from neuralqx.experimental.operators.symbolic.compiler.lowering.jax_lowerer import (
    _apply_update_program,
    _augment_env_with_emitted_state,
    _build_kbody_env,
    _eval_amplitude,
    _eval_predicate,
    _infer_shift_mod_spec_from_hilbert,
    _ir_uses_shift_mod,
    _resolve_shift_mod_spec,
)


def test_eval_amplitude_all_supported_ops(symbolic, jnp):
    env = {
        "__x__": jnp.asarray([2.0, 3.0]),
        "__x_prime__": jnp.asarray([4.0, 5.0]),
        "kappa": jnp.asarray(2.0),
    }
    A = symbolic.AmplitudeExpr
    expr = A.conj(
        A.sqrt(
            A.abs_(
                A.wrap_mod(
                    A.pow(
                        A.add(
                            A.static_index(0),
                            A.sub(A.static_emitted_index(1), A.symbol("kappa")),
                        ),
                        A.constant(1),
                    )
                )
            )
        )
    ) / A.constant(1)
    out = _eval_amplitude(expr, env, shift_mod_state_min=0, shift_mod_mod_span=3)
    # ((2 + (5-2))^1)=5 -> wrap mod 3 => 2 -> abs=2 -> sqrt=~1.414 -> conj same.
    assert np.allclose(np.asarray(out), np.sqrt(2.0))


def test_eval_amplitude_wrap_mod_requires_spec(symbolic, jnp):
    expr = symbolic.AmplitudeExpr.wrap_mod(symbolic.AmplitudeExpr.constant(1))
    with pytest.raises(ValueError, match="requires a resolved shift_mod_spec"):
        _eval_amplitude(expr, {"__x__": jnp.asarray([0])})


def test_eval_predicate_supports_boolean_tree(symbolic, jnp):
    env = {
        "__x__": jnp.asarray([1.0]),
        "site:i:value": jnp.asarray(1.0),
    }
    p = symbolic.PredicateExpr.and_(
        symbolic.PredicateExpr.constant(True),
        symbolic.PredicateExpr.or_(
            symbolic.PredicateExpr.lt(symbolic.AmplitudeExpr.constant(0), 1),
            symbolic.PredicateExpr.constant(False),
        ),
    )
    assert bool(np.asarray(_eval_predicate(p, env)))


def test_apply_update_program_cond_and_invalidate(symbolic, jnp):
    pred = symbolic.PredicateExpr.constant(True)
    cond = symbolic.Update.cond(
        pred,
        if_true=symbolic.write(0, 3),
        if_false=symbolic.write(0, 0),
    ).to_program()
    program = cond.append(symbolic.UpdateOp.from_mapping(kind="invalidate_branch"))
    x = jnp.asarray([1, 2], dtype=jnp.int32)
    env = {"__x__": x}
    xp, valid = _apply_update_program(x, program, env, hilbert_size=2)
    assert np.array_equal(np.asarray(xp), np.asarray([3, 2]))
    assert valid is False


def test_kbody_env_build_and_emitted_augmentation(symbolic, jnp):
    x = jnp.asarray([10, 20, 30], dtype=jnp.int32)
    env = _build_kbody_env(
        x, labels=("i", "j"), index_row=jnp.asarray([2, 0], dtype=jnp.int32)
    )
    assert int(np.asarray(env["site:i:index"])) == 2
    assert int(np.asarray(env["site:j:value"])) == 10

    xp = jnp.asarray([100, 200, 300], dtype=jnp.int32)
    env2 = _augment_env_with_emitted_state(env, xp, labels=("i", "j"))
    assert int(np.asarray(env2["emit:i:value"])) == 300
    assert int(np.asarray(env2["emit:j:value"])) == 100


def test_shift_mod_spec_inference_and_resolution(symbolic, hilbert_fock_one):
    spec = _infer_shift_mod_spec_from_hilbert(hilbert_fock_one)
    assert spec == (0, 3)

    op = (
        symbolic.DOperator(hilbert_fock_one, "shift_mod_resolve")
        .for_each_site("i")
        .emit(symbolic.shift_mod("i", 1), amplitude=1.0)
        .build()
    )
    ctx = symbolic.SymbolicCompilationContext(
        operator=op,
        ir=op.to_ir(),
        options=symbolic.SymbolicCompilerOptions(),
    )
    assert _ir_uses_shift_mod(ctx.ir) is True
    assert _resolve_shift_mod_spec(ctx) == (0, 3)


def test_shift_mod_spec_inference_rejects_invalid_local_states():
    class DummyHilbert:
        local_states = np.asarray([0.0, 0.5, 1.0])

    with pytest.raises(ValueError, match="integer local_states"):
        _infer_shift_mod_spec_from_hilbert(DummyHilbert())

    class DummyHilbert2:
        local_states = np.asarray([0, 2, 3])

    with pytest.raises(ValueError, match="contiguous unit-spaced"):
        _infer_shift_mod_spec_from_hilbert(DummyHilbert2())
