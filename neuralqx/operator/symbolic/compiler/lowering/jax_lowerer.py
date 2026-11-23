# Copyright (c) 2026 The neuraLQX and nkDSL Authors - All rights reserved.
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

Converts a :class:`~neuralqx.operator.symbolic.ir.program.SymbolicOperatorIR`
into a concrete compiled operator instance whose connectivity
kernel is built by interpreting the
:class:`~neuralqx.operator.symbolic.ir.expressions.AmplitudeExpr` /
:class:`~neuralqx.operator.symbolic.ir.predicates.PredicateExpr` /
:class:`~neuralqx.operator.symbolic.ir.update.UpdateProgram` expression trees
as JAX operations at trace time.

Architecture
-------------
For each IR term the lowerer generates a *term runner*: a Python function that,
given a single input configuration ``x`` (shape ``[hilbert_size]``), returns a
tuple ``(x_primes, mels, valids)`` of shape
``(max_conn_size, hilbert_size)``, ``(max_conn_size,)``, ``(max_conn_size,)`` respectively.

K-body terms use a static ``index_array`` (shape ``[M, K]``) and ``jax.vmap``
over its rows. Each row instantiates the iterator-label environment and
evaluates *all* emissions, producing ``E`` branches per row (total ``M * E``).

Branch-multiset note
---------------------
By default, duplicate connected states are coalesced by summing matrix
elements and dropping zero-amplitude entries (including invalidated branches).
Static target analysis determines the reduced output capacity and limits
runtime collision checks to unresolved target classes.
This behavior is controlled by
``SymbolicCompilerOptions.deduplicate_connected_components``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

import jax
from neuralqx.operator.symbolic.compiler.core.artifact import SymbolicCompiledArtifact
from neuralqx.operator.symbolic.compiler.core.connectivity import (
    ConnectivityDeduplicationPlan,
)
from neuralqx.operator.symbolic.compiler.core.context import SymbolicCompilationContext
from neuralqx.operator.symbolic.compiler.lowering.base import AbstractSymbolicLowerer
from neuralqx.operator.symbolic.compiler.lowering.coalescing import (
    coalesce_connected_components as _coalesce_connected_components,
)
from neuralqx.operator.symbolic.compiler.lowering.coalescing import (
    make_planned_coalescer,
)
from neuralqx.operator.symbolic.compiler.lowering.operator_registry import (
    SymbolicOperatorLoweringRegistry,
)
from neuralqx.operator.symbolic.compiler.lowering.operator_registry import (
    build_default_symbolic_operator_lowering_registry,
)
from neuralqx.operator.symbolic.compiler.passes.deduplication import (
    analyze_connectivity,
)
from neuralqx.operator.symbolic.core.compiled import create_compiled_operator
from neuralqx.operator.symbolic.debug import event as debug_event
from neuralqx.operator.symbolic.ir.expressions import AmplitudeExpr
from neuralqx.operator.symbolic.ir.expressions import parse_symbol_declaration_args
from neuralqx.operator.symbolic.ir.predicates import PredicateExpr
from neuralqx.operator.symbolic.ir.term import KBodyIteratorSpec
from neuralqx.operator.symbolic.ir.term import SymbolicIRTerm
from neuralqx.operator.symbolic.ir.update import UpdateOp
from neuralqx.operator.symbolic.ir.update import UpdateProgram
from neuralqx.utils.errors import NeuralqxKeyError
from neuralqx.utils.errors import NeuralqxValueError


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


def _infer_shift_mod_spec_from_hilbert(hilbert: Any):
    from neuralqx.operator.symbolic._hilbert import resolve_wrap_metadata
    from neuralqx.operator.symbolic._hilbert import shift_mod_metadata

    return resolve_wrap_metadata(shift_mod_metadata(hilbert))


def _resolve_shift_mod_spec(context):
    from neuralqx.operator.symbolic._hilbert import resolve_wrap_metadata

    if not _ir_uses_shift_mod(context.ir):
        return None
    metadata = context.ir.metadata_dict()
    if "shift_mod_spec" in metadata:
        return resolve_wrap_metadata(metadata)
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
        name, declaration = parse_symbol_declaration_args(expr.args)
        if name in env:
            if "dtype" in declaration:
                return jnp.asarray(env[name], dtype=np.dtype(declaration["dtype"]))
            return env[name]
        if "default" in declaration:
            if "dtype" in declaration:
                return jnp.asarray(
                    declaration["default"], dtype=np.dtype(declaration["dtype"])
                )
            return jnp.asarray(declaration["default"])
        raise NeuralqxKeyError(
            f"Symbol {name!r} not found in evaluation environment. "
            f"Available: {sorted(env.keys())!r}."
        )

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
            raise NeuralqxKeyError("static_index requires env['__x__'] to be set.")
        return x_arr[int(expr.args[0])]

    if op == "static_emitted_index":
        x_arr = env.get("__x_prime__")
        if x_arr is None:
            raise NeuralqxKeyError(
                "static_emitted_index requires env['__x_prime__'] to be set."
            )
        return x_arr[int(expr.args[0])]

    if op == "wrap_mod":
        from neuralqx.operator.symbolic._hilbert import SiteWrapDomains

        if isinstance(shift_mod_state_min, SiteWrapDomains):
            raise NeuralqxValueError(
                "wrap_mod expressions require a uniform unit-spaced integer basis. "
                "Use shift_mod(site, steps) for other local domains."
            )
        if shift_mod_state_min is None or shift_mod_mod_span is None:
            raise NeuralqxValueError(
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

    raise NeuralqxValueError(f"Unknown amplitude op: {op!r}")


def _eval_predicate_when(active: Any, expr: PredicateExpr, env: dict[str, Any]) -> Any:
    """
    Evaluates *expr* only when *active* is true, otherwise returns ``False``.

    This uses ``jax.lax.cond`` to avoid runtime execution of masked branches,
    which prevents invalid inactive-path numerics (for example divide-by-zero)
    from polluting downstream autodiff.
    """
    return jax.lax.cond(
        jnp.bool_(active),
        lambda _unused: jnp.bool_(_eval_predicate(expr, env)),
        lambda _unused: jnp.bool_(False),
        operand=None,
    )


def _eval_predicate(expr: PredicateExpr, env: dict[str, Any]) -> Any:
    op = expr.op
    if op == "const":
        return jnp.bool_(expr.args[0])
    if op == "not":
        return ~_eval_predicate(expr.args[0], env)
    if op == "and":
        result = jnp.bool_(True)
        for arg in expr.args:
            result = _eval_predicate_when(result, arg, env)
        return result
    if op == "or":
        result = jnp.bool_(False)
        for arg in expr.args:
            result = jax.lax.cond(
                jnp.bool_(result),
                lambda _unused: jnp.bool_(True),
                lambda _unused, pred=arg: jnp.bool_(_eval_predicate(pred, env)),
                operand=None,
            )
        return result
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
    raise NeuralqxValueError(f"Unknown predicate op: {op!r}")


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
            raise NeuralqxValueError(
                "Encountered shift_mod_site without a resolved shift_mod_spec."
            )

        idx = jnp.int32(_eval_amplitude(op.get("site"), env))
        delta = _eval_amplitude(op.get("delta"), env)

        cur = x_prime[idx]
        from neuralqx.operator.symbolic._hilbert import SiteWrapDomains

        if isinstance(shift_mod_state_min, SiteWrapDomains):
            return x_prime.at[idx].set(shift_mod_state_min.shift(idx, cur, delta))
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
        for flat_idx, val_expr in zip(flat_indices, value_exprs, strict=False):
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
                em_pred = _eval_predicate_when(pred, em.predicate, env)
                x_prime, uv = _apply_update_program(
                    x,
                    em.update_program,
                    env,
                    hilbert_size,
                    shift_mod_state_min=shift_mod_state_min,
                    shift_mod_mod_span=shift_mod_mod_span,
                )
                valid = jnp.bool_(em_pred & uv)

                def _eval_mel(_unused: Any, x_prime=x_prime, em=em) -> Any:
                    amp_env = _augment_env_with_emitted_state(env, x_prime, ())
                    mel = _eval_amplitude(
                        em.amplitude,
                        amp_env,
                        shift_mod_state_min=shift_mod_state_min,
                        shift_mod_mod_span=shift_mod_mod_span,
                    )
                    return jnp.asarray(mel, dtype=output_dtype)

                mel_cast = jax.lax.cond(
                    valid,
                    _eval_mel,
                    lambda _unused: jnp.zeros((), dtype=output_dtype),
                    operand=None,
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
                em_pred = _eval_predicate_when(pred, em.predicate, env)
                x_prime, uv = _apply_update_program(
                    x,
                    em.update_program,
                    env,
                    hilbert_size,
                    shift_mod_state_min=shift_mod_state_min,
                    shift_mod_mod_span=shift_mod_mod_span,
                )
                valid = jnp.bool_(em_pred & uv)

                def _eval_mel(_unused: Any, x_prime=x_prime, em=em) -> Any:
                    amp_env = _augment_env_with_emitted_state(env, x_prime, labels)
                    mel = _eval_amplitude(
                        em.amplitude,
                        amp_env,
                        shift_mod_state_min=shift_mod_state_min,
                        shift_mod_mod_span=shift_mod_mod_span,
                    )
                    return jnp.asarray(mel, dtype=output_dtype)

                mel_cast = jax.lax.cond(
                    valid,
                    _eval_mel,
                    lambda _unused: jnp.zeros((), dtype=output_dtype),
                    operand=None,
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
    *,
    operator_type: type[Any] | None = None,
    connection_method: str = "_get_conn_padded_kernel",
    operator_factory: Callable | None = None,
    is_diagonal: bool = False,
    support: tuple | None = None,
    deduplicate_connected_components: bool = True,
    deduplication_plan: ConnectivityDeduplicationPlan | None = None,
) -> Any:
    """Composes term runners into a single compiled operator instance."""
    _runners = tuple(term_runners)
    _padded = int(total_padded_size)
    _dedup = bool(deduplicate_connected_components)
    _coalescer = (
        make_planned_coalescer(deduplication_plan)
        if deduplication_plan is not None
        else _coalesce_connected_components
    )

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

        if _dedup:
            all_xp, all_mel = _coalescer(all_xp, all_mel, all_valid)
        else:
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
    def get_conn_padded_fn(x: Any) -> tuple:
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

    kwargs: dict[str, Any] = {
        "name": operator_name,
        "fn": get_conn_padded_fn,
        "is_hermitian": is_hermitian,
        "dtype": output_dtype,
        "max_conn_size": total_padded_size,
        "connection_method": connection_method,
        "operator_factory": operator_factory,
        "is_diagonal": is_diagonal,
        "support": support,
    }
    if operator_type is not None:
        kwargs["operator_type"] = operator_type
    return create_compiled_operator(hilbert, **kwargs)


#
#
#   JAX symbolic lowerer


class JAXSymbolicLowerer(AbstractSymbolicLowerer):
    """JAX-backend symbolic operator lowerer."""

    _LOWERER_NAME = "jax_symbolic_v1"
    _BACKEND = "jax"

    def __init__(
        self,
        *,
        operator_lowering_registry: SymbolicOperatorLoweringRegistry | None = None,
    ) -> None:
        if operator_lowering_registry is None:
            operator_lowering_registry = (
                build_default_symbolic_operator_lowering_registry()
            )
        self._operator_lowering_registry = operator_lowering_registry

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
        lowering_target = context.analysis("operator_lowering_target")
        if lowering_target is None:
            lowering_target = self._operator_lowering_registry.resolve(
                context.options.operator_lowering
            )
        debug_event(
            "starting jax lowering",
            scope="lowering",
            tag="LOWERING",
            operator_name=ir.operator_name,
            term_count=ir.term_count,
            dtype=ir.dtype_str,
            operator_lowering=lowering_target.name,
            target_operator_type=lowering_target.operator_type_qualname,
            target_connection_method=lowering_target.connection_method,
        )
        try:
            output_dtype = np.dtype(ir.dtype_str)
        except TypeError:
            output_dtype = np.complex64
            debug_event(
                "failed to parse dtype string; using fallback dtype",
                scope="lowering",
                tag="LOWERING",
                operator_name=ir.operator_name,
                dtype=ir.dtype_str,
                fallback=str(output_dtype),
            )

        total_max_conn_size: int = context.analysis("total_max_conn_size")
        if total_max_conn_size is None:
            total_max_conn_size = context.analysis("total_fanout")
        if total_max_conn_size is None:
            from neuralqx.operator.symbolic.compiler.passes.analysis import (
                _compute_term_max_conn_size,
            )

            total_max_conn_size = sum(
                _compute_term_max_conn_size(t, ir.hilbert_size) for t in ir.terms
            )
        deduplication_plan = context.analysis("connectivity_deduplication_plan")
        if context.options.deduplicate_connected_components:
            if deduplication_plan is None:
                deduplication_plan = analyze_connectivity(ir, context.operator.hilbert)
            total_max_conn_size = deduplication_plan.output_size
        total_max_conn_size = max(1, int(total_max_conn_size))
        debug_event(
            "resolved total max_conn_size",
            scope="lowering",
            tag="LOWERING",
            operator_name=ir.operator_name,
            total_max_conn_size=total_max_conn_size,
        )

        shift_mod_spec = _resolve_shift_mod_spec(context)
        if shift_mod_spec is None:
            shift_mod_state_min = None
            shift_mod_mod_span = None
            debug_event(
                "no shift_mod metadata required",
                scope="lowering",
                tag="LOWERING",
                operator_name=ir.operator_name,
            )
        else:
            shift_mod_state_min, shift_mod_mod_span = shift_mod_spec
            debug_event(
                "resolved shift_mod metadata",
                scope="lowering",
                tag="LOWERING",
                operator_name=ir.operator_name,
                state_min=shift_mod_state_min,
                mod_span=shift_mod_mod_span,
            )

        # Build per-term runners
        term_runners = []
        for idx, term in enumerate(ir.terms):
            runner = _make_kbody_runner(
                term,
                ir.hilbert_size,
                output_dtype,
                shift_mod_state_min=shift_mod_state_min,
                shift_mod_mod_span=shift_mod_mod_span,
            )
            term_runners.append(runner)
            debug_event(
                "built term runner",
                scope="lowering",
                tag="LOWERING",
                operator_name=ir.operator_name,
                term_index=idx,
                term_name=term.name,
            )

        from neuralqx.operator.symbolic.compiler.core.actions import _target

        supports = set()
        for term in ir.terms:
            rows = term.iterator.index_sets if term.iterator.labels else ((),)
            for row in rows:
                bindings = dict(zip(term.iterator.labels, row, strict=False))
                for emission in term.effective_emissions:
                    target = _target(
                        emission.update_program.ops, bindings, ir.hilbert_size
                    )
                    sites = (
                        tuple(range(ir.hilbert_size))
                        if target.opaque is not None
                        else tuple(site for site, _ in target.coordinates)
                    )
                    supports.add(sites)
        support = tuple(sorted(supports)) or ((),)
        is_diagonal = all(not sites for sites in support)

        kernel_plan = context.analysis("kernel_plan")
        kernel_summary = context.analysis(
            "kernel_planning",
            default={"enabled": False, "fallback_reason": "not_planned"},
        )
        if kernel_plan is not None and kernel_plan.fallback_reason is None:
            from neuralqx.operator.symbolic.compiler.lowering.kernel_lowerer import (
                kernel_schedule_summary,
            )
            from neuralqx.operator.symbolic.compiler.lowering.kernel_lowerer import (
                make_kernel_function,
            )

            def reference_factory() -> Any:
                return _build_compiled_operator(
                    hilbert=context.operator.hilbert,
                    operator_name=ir.operator_name,
                    is_hermitian=ir.is_hermitian,
                    output_dtype=output_dtype,
                    term_runners=term_runners,
                    total_padded_size=total_max_conn_size,
                    deduplicate_connected_components=context.options.deduplicate_connected_components,
                    deduplication_plan=deduplication_plan,
                ).get_conn_padded

            fn = make_kernel_function(
                kernel_plan,
                ir.hilbert_size,
                output_dtype,
                total_max_conn_size,
                reference_factory=reference_factory,
            )
            compiled_op = create_compiled_operator(
                context.operator.hilbert,
                name=ir.operator_name,
                fn=fn,
                is_hermitian=ir.is_hermitian,
                dtype=output_dtype,
                max_conn_size=total_max_conn_size,
                operator_type=lowering_target.operator_type,
                connection_method=lowering_target.connection_method,
                operator_factory=lowering_target.factory,
                is_diagonal=is_diagonal,
                support=support,
            )
            kernel_summary = {
                **kernel_summary,
                **kernel_schedule_summary(kernel_plan, ir.hilbert_size),
            }
        else:
            compiled_op = _build_compiled_operator(
                hilbert=context.operator.hilbert,
                operator_name=ir.operator_name,
                is_hermitian=ir.is_hermitian,
                output_dtype=output_dtype,
                term_runners=term_runners,
                total_padded_size=total_max_conn_size,
                operator_type=lowering_target.operator_type,
                connection_method=lowering_target.connection_method,
                operator_factory=lowering_target.factory,
                is_diagonal=is_diagonal,
                support=support,
                deduplicate_connected_components=context.options.deduplicate_connected_components,
                deduplication_plan=deduplication_plan,
            )

        context.set_selected_lowerer(self._LOWERER_NAME)
        debug_event(
            "completed jax lowering",
            scope="lowering",
            tag="LOWERING",
            operator_name=ir.operator_name,
            lowerer_name=self._LOWERER_NAME,
            total_padded_size=total_max_conn_size,
            operator_lowering=lowering_target.name,
        )

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
                "total_padded_size": total_max_conn_size,
                "is_hermitian": ir.is_hermitian,
                "deduplicate_connected_components": context.options.deduplicate_connected_components,
                "connectivity_deduplication": (
                    deduplication_plan.as_dict()
                    if deduplication_plan is not None
                    else None
                ),
                "connectivity_cost": context.analysis("connectivity_cost"),
                "kernel_planning": kernel_summary,
                "dsl_diagnostics": context.analysis("dsl_diagnostics", default=()),
                "operator_lowering": lowering_target.name,
                "target_operator_type": lowering_target.operator_type_qualname,
                "target_connection_method": lowering_target.connection_method,
            },
        )


def infer_shift_mod_spec_from_hilbert(hilbert: Any) -> tuple[int, int]:
    """
    Infers wrapped shift/mod metadata from one Hilbert space.

    Args:
        hilbert: Native Hilbert space exposing per-site ``local_spaces``.

    Returns:
        tuple[int, int]: ``(state_min, mod_span)`` for wrap-mod lowering.
    """
    return _infer_shift_mod_spec_from_hilbert(hilbert)


def eval_amplitude(
    expr: AmplitudeExpr,
    env: dict[str, Any],
    *,
    shift_mod_state_min: int | None = None,
    shift_mod_mod_span: int | None = None,
) -> Any:
    """
    Evaluates one amplitude expression against an environment mapping.

    Args:
        expr: Symbolic amplitude expression.
        env: Runtime evaluation environment.
        shift_mod_state_min: Optional lower bound for wrap-mod remapping.
        shift_mod_mod_span: Optional modulo span for wrap-mod remapping.

    Returns:
        Any: Evaluated scalar amplitude value.
    """
    return _eval_amplitude(
        expr,
        env,
        shift_mod_state_min=shift_mod_state_min,
        shift_mod_mod_span=shift_mod_mod_span,
    )


def eval_predicate(expr: PredicateExpr, env: dict[str, Any]) -> Any:
    """
    Evaluates one predicate expression against an environment mapping.

    Args:
        expr: Symbolic predicate expression.
        env: Runtime evaluation environment.

    Returns:
        Any: Truthy predicate evaluation result.
    """
    return _eval_predicate(expr, env)


def apply_single_update_op(
    op: UpdateOp,
    x: Any,
    env: dict[str, Any],
    hilbert_size: int,
    *,
    shift_mod_state_min: int | None = None,
    shift_mod_mod_span: int | None = None,
) -> Any:
    """
    Applies one update operation to a single configuration vector.

    Args:
        op: Update op to apply.
        x: Input configuration vector.
        env: Runtime environment map.
        hilbert_size: Number of Hilbert sites.
        shift_mod_state_min: Optional lower bound for wrap-mod remapping.
        shift_mod_mod_span: Optional modulo span for wrap-mod remapping.

    Returns:
        Any: Updated configuration vector.
    """
    return _apply_single_update_op(
        op,
        x,
        env,
        hilbert_size,
        shift_mod_state_min=shift_mod_state_min,
        shift_mod_mod_span=shift_mod_mod_span,
    )


def apply_update_program(
    x: Any,
    program: UpdateProgram,
    env: dict[str, Any],
    hilbert_size: int,
    *,
    shift_mod_state_min: int | None = None,
    shift_mod_mod_span: int | None = None,
) -> tuple[Any, Any]:
    """
    Applies one update program to a single configuration vector.

    Args:
        x: Input configuration vector.
        program: Update program to apply.
        env: Runtime environment map.
        hilbert_size: Number of Hilbert sites.
        shift_mod_state_min: Optional lower bound for wrap-mod remapping.
        shift_mod_mod_span: Optional modulo span for wrap-mod remapping.

    Returns:
        tuple[Any, Any]: ``(updated_configuration, is_valid)``.
    """
    return _apply_update_program(
        x,
        program,
        env,
        hilbert_size,
        shift_mod_state_min=shift_mod_state_min,
        shift_mod_mod_span=shift_mod_mod_span,
    )


def make_kbody_runner(
    term: SymbolicIRTerm,
    hilbert_size: int,
    output_dtype: Any,
    *,
    shift_mod_state_min: int | None = None,
    shift_mod_mod_span: int | None = None,
) -> Callable[[Any], tuple[Any, Any, Any]]:
    """
    Builds one executable runner for a single K-body IR term.

    Args:
        term: Symbolic IR term descriptor.
        hilbert_size: Number of Hilbert sites.
        output_dtype: Matrix-element output dtype.
        shift_mod_state_min: Optional lower bound for wrap-mod remapping.
        shift_mod_mod_span: Optional modulo span for wrap-mod remapping.

    Returns:
        Callable[[Any], tuple[Any, Any, Any]]: Runtime term runner.
    """
    return _make_kbody_runner(
        term,
        hilbert_size,
        output_dtype,
        shift_mod_state_min=shift_mod_state_min,
        shift_mod_mod_span=shift_mod_mod_span,
    )


def build_compiled_operator(
    hilbert: Any,
    *,
    operator_name: str,
    is_hermitian: bool,
    output_dtype: Any,
    term_runners: list[Callable[[Any], tuple[Any, Any, Any]]],
    total_padded_size: int,
    operator_type: type | None = None,
    connection_method: str = "_get_conn_padded_kernel",
    operator_factory: Callable | None = None,
    is_diagonal: bool = False,
    support: tuple | None = None,
    deduplicate_connected_components: bool = True,
) -> Any:
    """
    Builds one compiled operator instance from prepared term runners.

    Args:
        hilbert: Hilbert space for the resulting compiled operator.
        operator_name: Operator display name.
        is_hermitian: Hermiticity declaration.
        output_dtype: Matrix-element output dtype.
        term_runners: Per-term execution callables.
        total_padded_size: Output padding size for the connectivity kernel.
        operator_type: Optional custom operator class target.
        connection_method: Operator method name to attach connectivity callable to.
        deduplicate_connected_components: Whether to coalesce duplicate targets
            and drop zero-amplitude components.

    Returns:
        Any: Compiled operator object exposing the requested connection method.
    """
    return _build_compiled_operator(
        hilbert,
        operator_name=operator_name,
        is_hermitian=is_hermitian,
        output_dtype=output_dtype,
        term_runners=term_runners,
        total_padded_size=total_padded_size,
        operator_type=operator_type,
        connection_method=connection_method,
        operator_factory=operator_factory,
        is_diagonal=is_diagonal,
        support=support,
        deduplicate_connected_components=deduplicate_connected_components,
    )


__all__ = [
    "JAXSymbolicLowerer",
    "apply_single_update_op",
    "apply_update_program",
    "build_compiled_operator",
    "eval_amplitude",
    "eval_predicate",
    "infer_shift_mod_spec_from_hilbert",
    "make_kbody_runner",
]
