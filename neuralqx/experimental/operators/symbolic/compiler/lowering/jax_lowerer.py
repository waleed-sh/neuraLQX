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


"""
JAX backend symbolic operator lowerer.

Converts a :class:`~neuralqx.experimental.operators.symbolic.ir.program.SymbolicOperatorIR`
into a concrete :class:`~neuralqx.experimental.operators.symbolic.core.compiled.CompiledOperator`
whose ``_get_conn_padded`` kernel is built by interpreting the
:class:`~neuralqx.experimental.operators.symbolic.ir.expressions.AmplitudeExpr` /
:class:`~neuralqx.experimental.operators.symbolic.ir.predicates.PredicateExpr` /
:class:`~neuralqx.experimental.operators.symbolic.ir.update.UpdateProgram` expression trees
as JAX operations at trace time.

Architecture
-------------
For each IR term the lowerer generates a *term runner*: a Python function that,
given a single input configuration ``x`` (shape ``[hilbert_size]``), returns a
tuple ``(x_primes, mels, valids)`` of shape
``(fanout, hilbert_size)``, ``(fanout,)``, ``(fanout,)`` respectively.

K-body terms use a static ``index_array`` (shape ``[M, K]``) and ``jax.vmap``
over its rows. Each row instantiates the iterator-label environment and
evaluates *all* emissions, producing ``E`` branches per row (total ``M * E``).

Branch-multiset note
---------------------
Duplicate ``x'`` values are **not** coalesced. If multiple terms or emissions
produce the same connected state, they appear as separate rows in the padded
output.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from neuralqx.experimental.operators.symbolic.compiler.core.artifact import (
    SymbolicCompiledArtifact,
)
from neuralqx.experimental.operators.symbolic.compiler.core.context import (
    SymbolicCompilationContext,
)
from neuralqx.experimental.operators.symbolic.compiler.lowering.base import (
    AbstractSymbolicLowerer,
)
from neuralqx.experimental.operators.symbolic.core.compiled import CompiledOperator
from neuralqx.experimental.operators.symbolic.ir.expressions import AmplitudeExpr
from neuralqx.experimental.operators.symbolic.ir.predicates import PredicateExpr
from neuralqx.experimental.operators.symbolic.ir.term import (
    SymbolicIRTerm,
)
from neuralqx.experimental.operators.symbolic.ir.term import (
    KBodyIteratorSpec,
)
from neuralqx.experimental.operators.symbolic.ir.update import UpdateOp
from neuralqx.experimental.operators.symbolic.ir.update import UpdateProgram


def _amplitude_uses_wrap_mod(expr: AmplitudeExpr) -> bool:
    if expr.op == "wrap_mod":
        return True
    return any(
        isinstance(arg, AmplitudeExpr) and _amplitude_uses_wrap_mod(arg)
        for arg in expr.args
    )


def _ir_uses_shift_mod(ir: Any) -> bool:
    def op_uses_shift_mod(op: Any) -> bool:
        if op.kind == "shift_mod_site":
            return True
        if op.kind == "cond_branch":
            then_ops = op.get("then_ops") or ()
            else_ops = op.get("else_ops") or ()
            return any(op_uses_shift_mod(sub) for sub in then_ops) or any(
                op_uses_shift_mod(sub) for sub in else_ops
            )
        return False

    for term in ir.terms:
        for em in term.effective_emissions:
            if any(op_uses_shift_mod(op) for op in em.update_program.ops):
                return True
            if _amplitude_uses_wrap_mod(em.amplitude):
                return True
    return False


def _infer_shift_mod_spec_from_hilbert(hilbert: Any) -> tuple[int, int]:
    local_states = getattr(hilbert, "local_states", None)
    if local_states is None:
        raise ValueError(
            "shift_mod requires a discrete Hilbert with finite local_states."
        )

    states = np.asarray(local_states)
    if states.ndim != 1 or states.size == 0:
        raise ValueError(
            "shift_mod requires hilbert.local_states to be a non-empty 1D sequence."
        )

    states_i = states.astype(np.int64)
    if not np.array_equal(states, states_i):
        raise ValueError("shift_mod currently requires integer local_states.")

    state_min = int(states_i[0])
    expected = np.arange(state_min, state_min + len(states_i), dtype=np.int64)
    if not np.array_equal(states_i, expected):
        raise ValueError(
            "shift_mod currently requires contiguous unit-spaced local_states."
        )

    return state_min, int(len(states_i))


def _resolve_shift_mod_spec(
    context: SymbolicCompilationContext,
) -> tuple[int, int] | None:
    if not _ir_uses_shift_mod(context.ir):
        return None

    meta = context.ir.metadata_dict()
    spec = meta.get("shift_mod_spec")
    if spec is not None:
        if spec.get("version") != "uniform_integer_wrap_v1":
            raise ValueError(
                f"Unsupported shift_mod_spec version: {spec.get('version')!r}."
            )
        return int(spec["state_min"]), int(spec["mod_span"])

    # Fallback for older operators built before metadata was added
    return _infer_shift_mod_spec_from_hilbert(context.operator.hilbert)


#
#
#   Amplitude / predicate / update interpreters


def _eval_amplitude(
    expr: AmplitudeExpr,
    env: dict[str, Any],
    *,
    shift_mod_state_min: int | None = None,
    shift_mod_mod_span: int | None = None,
) -> Any:
    op = expr.op

    if op == "const":
        return jnp.array(expr.args[0])

    if op == "symbol":
        name = expr.args[0]
        if name not in env:
            raise KeyError(
                f"Symbol {name!r} not found in evaluation environment. "
                f"Available: {sorted(env.keys())!r}."
            )
        return env[name]

    if op == "neg":
        return -_eval_amplitude(
            expr.args[0],
            env,
            shift_mod_state_min=shift_mod_state_min,
            shift_mod_mod_span=shift_mod_mod_span,
        )

    if op == "sqrt":
        return jnp.sqrt(
            _eval_amplitude(
                expr.args[0],
                env,
                shift_mod_state_min=shift_mod_state_min,
                shift_mod_mod_span=shift_mod_mod_span,
            )
        )

    if op == "conj":
        return jnp.conj(
            _eval_amplitude(
                expr.args[0],
                env,
                shift_mod_state_min=shift_mod_state_min,
                shift_mod_mod_span=shift_mod_mod_span,
            )
        )

    if op == "add":
        return _eval_amplitude(
            expr.args[0],
            env,
            shift_mod_state_min=shift_mod_state_min,
            shift_mod_mod_span=shift_mod_mod_span,
        ) + _eval_amplitude(
            expr.args[1],
            env,
            shift_mod_state_min=shift_mod_state_min,
            shift_mod_mod_span=shift_mod_mod_span,
        )

    if op == "sub":
        return _eval_amplitude(
            expr.args[0],
            env,
            shift_mod_state_min=shift_mod_state_min,
            shift_mod_mod_span=shift_mod_mod_span,
        ) - _eval_amplitude(
            expr.args[1],
            env,
            shift_mod_state_min=shift_mod_state_min,
            shift_mod_mod_span=shift_mod_mod_span,
        )

    if op == "mul":
        return _eval_amplitude(
            expr.args[0],
            env,
            shift_mod_state_min=shift_mod_state_min,
            shift_mod_mod_span=shift_mod_mod_span,
        ) * _eval_amplitude(
            expr.args[1],
            env,
            shift_mod_state_min=shift_mod_state_min,
            shift_mod_mod_span=shift_mod_mod_span,
        )

    if op == "div":
        return _eval_amplitude(
            expr.args[0],
            env,
            shift_mod_state_min=shift_mod_state_min,
            shift_mod_mod_span=shift_mod_mod_span,
        ) / _eval_amplitude(
            expr.args[1],
            env,
            shift_mod_state_min=shift_mod_state_min,
            shift_mod_mod_span=shift_mod_mod_span,
        )

    if op == "pow":
        return jnp.power(
            _eval_amplitude(
                expr.args[0],
                env,
                shift_mod_state_min=shift_mod_state_min,
                shift_mod_mod_span=shift_mod_mod_span,
            ),
            _eval_amplitude(
                expr.args[1],
                env,
                shift_mod_state_min=shift_mod_state_min,
                shift_mod_mod_span=shift_mod_mod_span,
            ),
        )

    if op == "abs_":
        return jnp.abs(
            _eval_amplitude(
                expr.args[0],
                env,
                shift_mod_state_min=shift_mod_state_min,
                shift_mod_mod_span=shift_mod_mod_span,
            )
        )

    if op == "static_index":
        x_arr = env.get("__x__")
        if x_arr is None:
            raise KeyError("static_index requires env['__x__'] to be set.")
        return x_arr[int(expr.args[0])]

    if op == "static_emitted_index":
        x_arr = env.get("__x_prime__")
        if x_arr is None:
            raise KeyError(
                "static_emitted_index requires env['__x_prime__'] to be set."
            )
        return x_arr[int(expr.args[0])]

    if op == "wrap_mod":
        if shift_mod_state_min is None or shift_mod_mod_span is None:
            raise ValueError(
                "wrap_mod requires a resolved shift_mod_spec, but none was provided."
            )
        value = _eval_amplitude(
            expr.args[0],
            env,
            shift_mod_state_min=shift_mod_state_min,
            shift_mod_mod_span=shift_mod_mod_span,
        )
        state_min = jnp.asarray(shift_mod_state_min, dtype=jnp.asarray(value).dtype)
        mod_span = jnp.asarray(shift_mod_mod_span, dtype=jnp.asarray(value).dtype)
        return ((value - state_min) % mod_span) + state_min

    raise ValueError(f"Unknown amplitude op: {op!r}")


def _eval_predicate(expr: PredicateExpr, env: dict[str, Any]) -> Any:
    op = expr.op
    if op == "const":
        return jnp.bool_(expr.args[0])
    if op == "not":
        return ~_eval_predicate(expr.args[0], env)
    if op == "and":
        return _eval_predicate(expr.args[0], env) & _eval_predicate(expr.args[1], env)
    if op == "or":
        return _eval_predicate(expr.args[0], env) | _eval_predicate(expr.args[1], env)
    left = _eval_amplitude(expr.args[0], env)
    right = _eval_amplitude(expr.args[1], env)
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op == "lt":
        return left < right
    if op == "le":
        return left <= right
    if op == "gt":
        return left > right
    if op == "ge":
        return left >= right
    raise ValueError(f"Unknown predicate op: {op!r}")


def _apply_single_update_op(
    op: UpdateOp,
    x_prime: Any,
    env: dict[str, Any],
    hilbert_size: int,
    *,
    shift_mod_state_min: int | None = None,
    shift_mod_mod_span: int | None = None,
) -> Any:
    """Applies one update op to x_prime, returns updated x_prime."""
    if op.kind == "write_site":
        idx = jnp.int32(_eval_amplitude(op.get("site"), env))
        val = _eval_amplitude(op.get("value"), env)
        return x_prime.at[idx].set(val)

    if op.kind == "shift_site":
        idx = jnp.int32(_eval_amplitude(op.get("site"), env))
        delta = _eval_amplitude(op.get("delta"), env)
        return x_prime.at[idx].set(x_prime[idx] + delta)

    if op.kind == "shift_mod_site":
        if shift_mod_state_min is None or shift_mod_mod_span is None:
            raise ValueError(
                "Encountered shift_mod_site without a resolved shift_mod_spec."
            )

        idx = jnp.int32(_eval_amplitude(op.get("site"), env))
        delta = _eval_amplitude(op.get("delta"), env)

        cur = x_prime[idx]
        state_min = jnp.asarray(shift_mod_state_min, dtype=cur.dtype)
        mod_span = jnp.asarray(shift_mod_mod_span, dtype=cur.dtype)

        wrapped = ((cur + delta - state_min) % mod_span) + state_min
        return x_prime.at[idx].set(wrapped)

    if op.kind == "swap_sites":
        idx_a = jnp.int32(_eval_amplitude(op.get("site_a"), env))
        idx_b = jnp.int32(_eval_amplitude(op.get("site_b"), env))
        val_a, val_b = x_prime[idx_a], x_prime[idx_b]
        return x_prime.at[idx_a].set(val_b).at[idx_b].set(val_a)

    if op.kind == "affine_site":
        idx = jnp.int32(_eval_amplitude(op.get("site"), env))
        scale = _eval_amplitude(op.get("scale"), env)
        bias = _eval_amplitude(op.get("bias"), env)
        return x_prime.at[idx].set(scale * x_prime[idx] + bias)

    if op.kind == "permute_sites":
        site_exprs = op.get("sites")
        K = len(site_exprs)
        indices = [jnp.int32(_eval_amplitude(s, env)) for s in site_exprs]
        # Read all original values first (atomic rotation)
        orig_vals = [x_prime[idx] for idx in indices]
        # Write cyclically: slot k receives original value from slot (k+1)%K
        for k in range(K):
            x_prime = x_prime.at[indices[k]].set(orig_vals[(k + 1) % K])
        return x_prime

    if op.kind == "scatter":
        flat_indices = op.get("flat_indices")
        value_exprs = op.get("values")
        for flat_idx, val_expr in zip(flat_indices, value_exprs):
            val = _eval_amplitude(val_expr, env)
            x_prime = x_prime.at[int(flat_idx)].set(val)
        return x_prime

    if op.kind == "cond_branch":
        pred = _eval_predicate(op.get("predicate"), env)
        then_ops = op.get("then_ops") or ()
        else_ops = op.get("else_ops") or ()

        def true_fn(xw: Any) -> Any:
            for sub in then_ops:
                xw = _apply_single_update_op(
                    sub,
                    xw,
                    env,
                    hilbert_size,
                    shift_mod_state_min=shift_mod_state_min,
                    shift_mod_mod_span=shift_mod_mod_span,
                )
            return xw

        def false_fn(xw: Any) -> Any:
            for sub in else_ops:
                xw = _apply_single_update_op(
                    sub,
                    xw,
                    env,
                    hilbert_size,
                    shift_mod_state_min=shift_mod_state_min,
                    shift_mod_mod_span=shift_mod_mod_span,
                )
            return xw

        return jax.lax.cond(pred, true_fn, false_fn, x_prime)

    # invalidate_branch does not affect x_prime (validity is tracked separately)
    return x_prime


def _apply_update_program(
    x: Any,
    program: UpdateProgram,
    env: dict[str, Any],
    hilbert_size: int,
    *,
    shift_mod_state_min: int | None = None,
    shift_mod_mod_span: int | None = None,
) -> tuple[Any, bool]:
    """
    Applies all ops in *program* to a working copy of *x*.

    Returns ``(x_prime, valid)`` where ``valid`` is ``False`` when an
    ``invalidate_branch`` op is present.
    """
    x_prime = x
    valid = True
    for op in program.ops:
        if op.kind == "invalidate_branch":
            valid = False
        else:
            x_prime = _apply_single_update_op(
                op,
                x_prime,
                env,
                hilbert_size,
                shift_mod_state_min=shift_mod_state_min,
                shift_mod_mod_span=shift_mod_mod_span,
            )
    return x_prime, valid


#
#
#   Environment builders


def _build_kbody_env(x: Any, labels: tuple, index_row: Any) -> dict[str, Any]:
    """Builds env for a K-body kernel given one row of site indices."""
    env: dict[str, Any] = {"__x__": x}
    for k, label in enumerate(labels):
        idx = index_row[k]
        env[f"site:{label}:index"] = idx
        env[f"site:{label}:value"] = x[idx]
    return env


def _augment_env_with_emitted_state(
    env: dict[str, Any],
    x_prime: Any,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    """Returns a copy of env augmented with emitted-state symbols."""
    amp_env = dict(env)
    amp_env["__x_prime__"] = x_prime
    for label in labels:
        idx_key = f"site:{label}:index"
        if idx_key in env:
            idx = env[idx_key]
            amp_env[f"emit:{label}:index"] = idx
            amp_env[f"emit:{label}:value"] = x_prime[idx]
    return amp_env


#
#
#   K-body runner


def _make_kbody_runner(
    term: SymbolicIRTerm,
    hilbert_size: int,
    output_dtype: Any,
    *,
    shift_mod_state_min=None,
    shift_mod_mod_span=None,
) -> Any:
    """
    Runner for KBodyIteratorSpec terms with multi-emission support.

    Produces ``(x_primes, mels, valids)`` of shape
    ``(M*E, H)``, ``(M*E,)``, ``(M*E,)`` where M = len(index_sets)
    and E = len(effective_emissions).
    """
    spec: KBodyIteratorSpec = term.iterator
    labels = spec.labels
    emissions = term.effective_emissions
    E = len(emissions)

    # global (K=0): single evaluation, no vmap
    if len(labels) == 0:
        predicate_ir = term.predicate

        def runner_global(x: Any) -> tuple:
            env: dict[str, Any] = {"__x__": x}
            pred = _eval_predicate(predicate_ir, env)
            xp_list, mel_list, valid_list = [], [], []
            for em in emissions:
                x_prime, uv = _apply_update_program(
                    x,
                    em.update_program,
                    env,
                    hilbert_size,
                    shift_mod_state_min=shift_mod_state_min,
                    shift_mod_mod_span=shift_mod_mod_span,
                )
                amp_env = _augment_env_with_emitted_state(env, x_prime, ())
                mel = _eval_amplitude(
                    em.amplitude,
                    amp_env,
                    shift_mod_state_min=shift_mod_state_min,
                    shift_mod_mod_span=shift_mod_mod_span,
                )
                valid = pred & uv
                mel_cast = jnp.where(
                    valid,
                    jnp.asarray(mel, dtype=output_dtype),
                    jnp.zeros((), dtype=output_dtype),
                )
                xp_list.append(x_prime)
                mel_list.append(mel_cast)
                valid_list.append(jnp.bool_(valid))
            if E == 1:
                return (
                    jnp.expand_dims(xp_list[0], 0),
                    jnp.expand_dims(mel_list[0], 0),
                    jnp.expand_dims(valid_list[0], 0),
                )
            return (
                jnp.stack(xp_list, axis=0),
                jnp.stack(mel_list, axis=0),
                jnp.stack(valid_list, axis=0),
            )

        return runner_global

    # K >= 1: vmap over index_array (M, K)
    index_array = jnp.array(spec.index_sets, dtype=jnp.int32)
    predicate_ir = term.predicate

    def runner_kbody(x: Any) -> tuple:
        def kernel(index_row: Any) -> tuple:
            env = _build_kbody_env(x, labels, index_row)
            pred = _eval_predicate(predicate_ir, env)
            xp_list, mel_list, valid_list = [], [], []
            for em in emissions:
                x_prime, uv = _apply_update_program(
                    x,
                    em.update_program,
                    env,
                    hilbert_size,
                    shift_mod_state_min=shift_mod_state_min,
                    shift_mod_mod_span=shift_mod_mod_span,
                )
                amp_env = _augment_env_with_emitted_state(env, x_prime, labels)
                mel = _eval_amplitude(
                    em.amplitude,
                    amp_env,
                    shift_mod_state_min=shift_mod_state_min,
                    shift_mod_mod_span=shift_mod_mod_span,
                )
                valid = pred & uv
                mel_cast = jnp.where(
                    valid,
                    jnp.asarray(mel, dtype=output_dtype),
                    jnp.zeros((), dtype=output_dtype),
                )
                xp_list.append(x_prime)
                mel_list.append(mel_cast)
                valid_list.append(jnp.bool_(valid))
            if E == 1:
                return xp_list[0], mel_list[0], valid_list[0]
            xp_stacked = jnp.stack(xp_list, axis=0)
            mel_stacked = jnp.stack(mel_list, axis=0)
            valid_stacked = jnp.stack(valid_list, axis=0)
            return xp_stacked, mel_stacked, valid_stacked

        xp_v, mel_v, valid_v = jax.vmap(kernel)(index_array)
        # xp_v: (M, H) or (M, E, H)
        # mel_v: (M,) or (M, E)

        if E == 1:
            # (M, H), (M,), (M,)
            return xp_v, mel_v, valid_v

        # Flatten M*E
        M = index_array.shape[0]
        return (
            xp_v.reshape(M * E, hilbert_size),
            mel_v.reshape(M * E),
            valid_v.reshape(M * E),
        )

    return runner_kbody


#
#
#   Compiled operator builder


def _build_compiled_operator(
    hilbert: Any,
    operator_name: str,
    is_hermitian: bool,
    output_dtype: Any,
    term_runners: list[Any],
    total_padded_size: int,
) -> CompiledOperator:
    """Composes term runners into a single ``CompiledOperator``."""
    _runners = tuple(term_runners)
    _padded = int(total_padded_size)

    def _single_sample(x: Any) -> tuple:
        all_xp, all_mel, all_valid = [], [], []
        for runner in _runners:
            xp, mel, valid = runner(x)
            all_xp.append(xp)
            all_mel.append(mel)
            all_valid.append(valid)

        all_xp = jnp.concatenate(all_xp, axis=0)
        all_mel = jnp.concatenate(all_mel, axis=0)
        all_valid = jnp.concatenate(all_valid, axis=0)
        all_mel = jnp.where(all_valid, all_mel, jnp.zeros_like(all_mel))

        current = all_xp.shape[0]
        if current < _padded:
            pad_n = _padded - current
            xp_pad = jnp.zeros((_padded, x.shape[-1]), dtype=all_xp.dtype)
            m_pad = jnp.zeros((_padded,), dtype=all_mel.dtype)
            all_xp = jnp.concatenate([all_xp, xp_pad[:pad_n]], axis=0)
            all_mel = jnp.concatenate([all_mel, m_pad[:pad_n]], axis=0)
        elif current > _padded:
            all_xp = all_xp[:_padded]
            all_mel = all_mel[:_padded]
        return all_xp, all_mel

    @jax.jit
    def _get_conn_padded_fn(x: Any) -> tuple:
        if x.ndim == 1:
            return _single_sample(x)
        batch_shape = x.shape[:-1]
        hilbert_size = x.shape[-1]
        x_2d = x.reshape(-1, hilbert_size)
        xp_2d, m_2d = jax.vmap(_single_sample)(x_2d)
        return (
            xp_2d.reshape(batch_shape + xp_2d.shape[-2:]),
            m_2d.reshape(batch_shape + m_2d.shape[-1:]),
        )

    return CompiledOperator(
        hilbert,
        name=operator_name,
        fn=_get_conn_padded_fn,
        is_hermitian=is_hermitian,
        dtype=output_dtype,
    )


#
#
#   JAX symbolic lowerer


class JAXSymbolicLowerer(AbstractSymbolicLowerer):
    """JAX-backend symbolic operator lowerer."""

    _LOWERER_NAME = "jax_symbolic_v1"
    _BACKEND = "jax"

    @property
    def name(self) -> str:
        return self._LOWERER_NAME

    @property
    def backend(self) -> str:
        return self._BACKEND

    def supports(self, context: SymbolicCompilationContext) -> bool:
        selected = context.selected_backend or context.options.backend_preference
        return context.ir.mode == "symbolic" and selected in {"jax", "auto"}

    def lower(self, context: SymbolicCompilationContext) -> SymbolicCompiledArtifact:
        ir = context.ir
        try:
            output_dtype = np.dtype(ir.dtype_str)
        except TypeError:
            output_dtype = np.complex64

        total_fanout: int = context.analysis("total_fanout")
        if total_fanout is None:
            from neuralqx.experimental.operators.symbolic.compiler.passes.analysis import (
                _compute_term_fanout,
            )

            total_fanout = sum(
                _compute_term_fanout(t, ir.hilbert_size) for t in ir.terms
            )
        total_fanout = max(1, int(total_fanout))

        shift_mod_spec = _resolve_shift_mod_spec(context)
        if shift_mod_spec is None:
            shift_mod_state_min = None
            shift_mod_mod_span = None
        else:
            shift_mod_state_min, shift_mod_mod_span = shift_mod_spec

        # Build per-term runners
        term_runners = []
        for term in ir.terms:
            runner = _make_kbody_runner(
                term,
                ir.hilbert_size,
                output_dtype,
                shift_mod_state_min=shift_mod_state_min,
                shift_mod_mod_span=shift_mod_mod_span,
            )
            term_runners.append(runner)

        compiled_op = _build_compiled_operator(
            hilbert=context.operator.hilbert,
            operator_name=ir.operator_name,
            is_hermitian=ir.is_hermitian,
            output_dtype=output_dtype,
            term_runners=term_runners,
            total_padded_size=total_fanout,
        )

        context.set_selected_lowerer(self._LOWERER_NAME)

        return SymbolicCompiledArtifact.create(
            operator_name=ir.operator_name,
            backend=self._BACKEND,
            lowerer_name=self._LOWERER_NAME,
            compiled_operator=compiled_op,
            cache_key=None,
            pass_reports=context.pass_reports,
            metadata={
                "hilbert_size": ir.hilbert_size,
                "dtype_str": ir.dtype_str,
                "term_count": ir.term_count,
                "total_padded_size": total_fanout,
                "is_hermitian": ir.is_hermitian,
            },
        )


__all__ = ["JAXSymbolicLowerer"]
