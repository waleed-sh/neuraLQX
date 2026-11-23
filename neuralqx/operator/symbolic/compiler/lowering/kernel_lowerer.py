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

"""JAX execution of structured scalar DAGs and deferred target index maps.

No operator names or physics families participate in lowering. The executable
schedule chooses bounded dense coordinate selection or sparse assembly and
retains exact collision handling for unproved relations.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable
from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np

import jax
from neuralqx.operator.symbolic.compiler.core.actions import _UNKNOWN
from neuralqx.operator.symbolic.compiler.core.actions import _literal
from neuralqx.operator.symbolic.compiler.core.actions import _Node
from neuralqx.operator.symbolic.compiler.core.kernel import KernelAction
from neuralqx.operator.symbolic.compiler.core.kernel import KernelPlan
from neuralqx.operator.symbolic.compiler.core.kernel import ScalarBatch
from neuralqx.operator.symbolic.compiler.core.kernel import scalar_template
from neuralqx.operator.symbolic.compiler.lowering.coalescing import (
    make_grouped_coalescer,
)
from neuralqx.utils.errors import NeuralqxKeyError
from neuralqx.utils.errors import NeuralqxValueError


@partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def _unit_shift(source: Any, endpoint: int, reset: int, delta: int) -> Any:
    """Computes a proved unit wrap with the modulo operation's floating tangent."""
    return jnp.where(
        source == endpoint, jnp.asarray(reset, source.dtype), source + delta
    )


@_unit_shift.defjvp
def _unit_shift_jvp(
    endpoint: int, reset: int, delta: int, primals: tuple[Any], tangents: tuple[Any]
) -> tuple[Any, Any]:
    # A direct select has a zero tangent at the endpoint, whereas JAX modulo
    # uses a unit source tangent. Preserve that convention without changing
    # any primal arithmetic or adding floating cancellation to the fast path.
    return _unit_shift(primals[0], endpoint, reset, delta), tangents[0]


def _constant_value(node: _Node) -> Any:
    """Reconstructs a literal without changing Python weak scalar promotion."""
    if node[0] == "array":
        return np.frombuffer(node[3], dtype=np.dtype(node[1])).reshape(node[2])
    if node[1] in {"bool", "int", "float", "complex"}:
        return node[-1]
    return np.asarray(node[-1], dtype=np.dtype(node[-2]))


def _unit_wrap_legal(
    dtype: Any, spec: tuple[int, int], delta: int, basis: tuple[Any, ...]
) -> bool:
    """Proves representability of every intermediate in an original unit wrap."""
    lo, q = spec
    hi = lo + q - 1
    if q <= 0 or len(basis) != q or any(v != lo + i for i, v in enumerate(basis)):
        return False
    lower = min(lo, lo + delta, delta, 0)
    upper = max(hi, hi + delta, q - 1 + delta, q)
    if jnp.issubdtype(dtype, jnp.integer):
        info = np.iinfo(dtype)
        return bool(info.min <= lower and upper <= info.max)
    if jnp.issubdtype(dtype, jnp.floating):
        limit = 2 ** (jnp.finfo(dtype).nmant + 1)  # type: ignore[no-untyped-call]
        return bool(-limit <= lower and upper <= limit)
    return False


def _evaluate_node(
    node: _Node,
    x: Any,
    inputs: tuple[Any, ...],
    targets: Any,
    branch_to_target: dict[int, int],
    plan: KernelPlan,
    cache: dict[_Node, Any],
) -> Any:
    """Lowers an ordered scalar DAG, memoizing only within its valid region."""
    if node in cache:
        return cache[node]
    op = node[0]

    def evaluate(child: _Node) -> Any:
        return _evaluate_node(child, x, inputs, targets, branch_to_target, plan, cache)

    def region(child: _Node) -> Any:
        return _evaluate_node(child, x, inputs, targets, branch_to_target, plan, {})

    if op in {"const", "array"}:
        result = jnp.asarray(_constant_value(node))
    elif op == "input":
        result = inputs[node[1]]
    elif op == "read":
        result = x[node[1]]
    elif op == "target_read":
        position = branch_to_target.get(node[1])
        result = x[node[2]] if position is None else targets[position, node[2]]
    elif op == "cast":
        result = jnp.asarray(evaluate(node[1]), dtype=x.dtype)
    elif op == "convert":
        result = jnp.asarray(evaluate(node[2]), dtype=np.dtype(node[1]))
    elif op == "symbol":
        raise NeuralqxKeyError(f"Unresolved kernel symbol {node[1]!r}.")
    elif op in {"select", "predicate:and", "predicate:or"}:
        if op == "select":
            pred = evaluate(node[1])
            result = jax.lax.cond(
                pred, lambda _: region(node[2]), lambda _: region(node[3]), None
            )
        else:
            is_and = op == "predicate:and"
            result = jnp.bool_(is_and)
            for child in node[1:]:
                result = jax.lax.cond(
                    result,
                    (
                        (lambda _, c=child: region(c))
                        if is_and
                        else (lambda _: jnp.bool_(True))
                    ),
                    (
                        (lambda _: jnp.bool_(False))
                        if is_and
                        else (lambda _, c=child: region(c))
                    ),
                    None,
                )
    elif op == "predicate:not":
        result = ~evaluate(node[1])
    elif op.startswith("predicate:"):
        a, b = evaluate(node[1]), evaluate(node[2])
        comparison = op[10:]
        result = {
            "eq": jnp.equal,
            "ne": jnp.not_equal,
            "lt": jnp.less,
            "le": jnp.less_equal,
            "gt": jnp.greater,
            "ge": jnp.greater_equal,
        }[comparison](a, b)
    elif op in {"wrap", "wrap_mod"}:
        spec = plan.wrap_spec
        if spec is None:
            raise NeuralqxValueError(
                "A modular expression requires a resolved shift_mod_spec."
            )
        argument = node[1]
        delta = _literal(argument[2]) if argument[0] == "add" else _UNKNOWN
        if (
            op == "wrap"
            and argument[0] == "add"
            and argument[1][0] in {"input", "read"}
            and type(delta) is int
            and delta in {-1, 1}
            and _unit_wrap_legal(x.dtype, spec, delta, plan.local_states)
        ):
            source = evaluate(argument[1])
            lo, q = spec
            endpoint, reset = (lo + q - 1, lo) if delta == 1 else (lo, lo + q - 1)
            result = _unit_shift(source, endpoint, reset, delta)
        else:
            value = evaluate(argument)
            dtype = x.dtype if op == "wrap" else value.dtype
            minimum = jnp.asarray(spec[0], dtype=dtype)
            period = jnp.asarray(spec[1], dtype=dtype)
            result = ((value - minimum) % period) + minimum
    else:
        unary: dict[str, Callable[..., Any]] = {
            "neg": jnp.negative,
            "sqrt": jnp.sqrt,
            "conj": jnp.conj,
            "abs_": jnp.abs,
        }
        binary: dict[str, Callable[..., Any]] = {
            "add": jnp.add,
            "sub": jnp.subtract,
            "mul": jnp.multiply,
            "div": jnp.divide,
            "pow": jnp.power,
        }
        if op in unary:
            result = unary[op](evaluate(node[1]))
        elif op in binary:
            left, right = evaluate(node[1]), evaluate(node[2])
            exponent = _literal(node[2]) if op == "pow" else _UNKNOWN
            promoted = jnp.result_type(left, right)
            if (
                op == "pow"
                and type(exponent) is int
                and 0 <= exponent <= 63
                and jnp.issubdtype(promoted, jnp.integer)
                and not bool(getattr(left, "weak_type", False))
            ):
                # For integer results, multiplication is exact modulo the
                # promoted dtype width. Preserve promotion before selecting a
                # static addition chain. The bounded exponent range matches
                # all bits consumed by JAX's dynamic integer-power algorithm.
                result = jax.lax.integer_pow(
                    jnp.asarray(left, dtype=promoted), exponent
                )
            else:
                result = binary[op](left, right)
        else:
            raise NeuralqxValueError(f"Unsupported kernel expression {op!r}.")
    cache[node] = result
    return result


def _source_inputs(
    x: Any, row: Any, indices: tuple[tuple[int, ...], ...]
) -> tuple[Any, ...]:
    """Keeps uniform coordinates static and maps only genuinely varying reads."""
    return tuple(
        x[indices[0][i]] if all(r[i] == indices[0][i] for r in indices) else x[row[i]]
        for i in range(len(indices[0]))
    )


def _static_take(values: Any, indices: np.ndarray) -> Any:
    """Turns contiguous static index maps into views instead of dynamic gathers."""
    flat = indices.reshape(-1)
    if len(flat) and np.array_equal(
        flat, np.arange(int(flat[0]), int(flat[0]) + len(flat))
    ):
        return values[int(flat[0]) : int(flat[0]) + len(flat)].reshape(indices.shape)
    return values[indices]


def _scalar_runner(
    batch: ScalarBatch, plan: KernelPlan, dtype: Any, branch_to_target: dict[int, int]
) -> Callable[[Any, Any], Any]:
    """Builds an action-axis map while preserving guarded evaluation regions."""
    indices = jnp.asarray(batch.indices, dtype=jnp.int32)
    valid = plan.actions[batch.actions[0]].valid

    def run(x: Any, targets: Any) -> Any:
        def lane(row: Any) -> Any:
            inputs = _source_inputs(x, row, batch.indices)
            term, emission, amplitude = batch.expressions

            def evaluate(node: _Node) -> Any:
                return _evaluate_node(
                    node, x, inputs, targets, branch_to_target, plan, {}
                )

            active = evaluate(term)
            active = jax.lax.cond(
                active, lambda _: evaluate(emission), lambda _: jnp.bool_(False), None
            )
            return jax.lax.cond(
                active & valid,
                lambda _: jnp.asarray(evaluate(amplitude), dtype=dtype),
                lambda _: jnp.zeros((), dtype=dtype),
                None,
            )

        return jax.vmap(lane)(indices) if batch.mapped else lane(indices[0])[None]

    return run


def _coordinate_schedule(
    plan: KernelPlan, size: int
) -> tuple[str, dict[Any, list[Any]]]:
    """Selects bounded dense local maps or sparse coordinate assembly."""
    groups: dict[Any, list[Any]] = {}
    local = True
    for position, representative in enumerate(plan.representatives):
        action = plan.actions[representative]
        if not action.mapped and any(
            _has_control(n) for _, n in action.target.coordinates
        ):
            continue
        for site, node in action.target.coordinates:
            template, indices = scalar_template((node,))
            groups.setdefault(template, []).append((position, site, node, indices))
            local &= indices in {(), (site,)} and not _has_control(template[0])
    dense = (
        local
        and len(groups) <= 8
        and len(plan.representatives) * size <= plan.dense_table_limit
    )
    return ("dense_select" if dense else "sparse_scatter"), groups


def _nonzero_groups(plan: KernelPlan, source_dtype: Any, dtype: Any) -> bool:
    """Proves nonvanishing by bounded exhaustive local-domain evaluation.

    Every assignment of every scalar input slot is covered, not sampled.
    Reduction proofs accept constant groups or sums of nonnegative, finite
    typed contributions with at least one normal positive contribution. The
    latter condition is independent of floating reduction reassociation.
    Unknowns, large domains, guards, and complex varying sums retain packing.
    """
    if (
        plan.connectivity is None
        or plan.connectivity.collision_components
        or not plan.local_states
    ):
        return False
    basis = np.asarray(plan.local_states)
    if not np.all(np.isfinite(basis)) or not np.array_equal(
        basis.astype(source_dtype), basis
    ):
        return False
    bounds: dict[int, tuple[Any, Any]] = {}
    remaining = 4096
    with jax.ensure_compile_time_eval():
        dummy = np.empty((0,), dtype=source_dtype)
        for batch in plan.scalar_batches:
            term, emission, amplitude = batch.expressions
            if _has_control(amplitude) or not plan.actions[batch.actions[0]].valid:
                return False
            # Source-dependent guards cannot establish unconditional activity.
            if scalar_template((term, emission))[1]:
                return False
            try:
                active = all(
                    bool(np.asarray(_evaluate_node(p, dummy, (), None, {}, plan, {})))
                    for p in (term, emission)
                )
            except (KeyError, ValueError, IndexError):
                return False
            if not active:
                return False
            slots = len(batch.indices[0])
            assignments = len(basis) ** slots
            if assignments > min(256, remaining):
                return False
            remaining -= assignments
            rows = np.asarray(
                list(itertools.product(basis, repeat=slots)), dtype=source_dtype
            ).reshape(assignments, slots)

            def evaluate(row: Any, amplitude=amplitude, slots=slots) -> Any:
                value = _evaluate_node(
                    amplitude,
                    dummy,
                    tuple(row[i] for i in range(slots)),
                    None,
                    {},
                    plan,
                    {},
                )
                return jnp.asarray(value, dtype=dtype)

            values = np.asarray(jax.vmap(evaluate)(jnp.asarray(rows)))
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                return False
            if np.iscomplexobj(values):
                if not np.all(values == values[0]):
                    return False
                lo = hi = values[0]
            else:
                lo, hi = values.min(), values.max()
            for action in batch.actions:
                bounds[action] = (lo, hi)
        for group in plan.connectivity.groups:
            intervals = [bounds[i] for i in group]
            kind = np.dtype(dtype).kind
            if len(intervals) == 1 and intervals[0][0] == intervals[0][1]:
                value = intervals[0][0]
                if value == 0:
                    return False
                if kind in "fc":
                    real_dtype = np.empty((), dtype=dtype).real.dtype
                    tiny = np.finfo(real_dtype).tiny
                    if max(abs(np.real(value)), abs(np.imag(value))) < tiny:
                        return False
            elif kind == "f":
                tiny = np.finfo(dtype).tiny
                positive = all(lo >= 0 for lo, _ in intervals) and any(
                    lo >= tiny for lo, _ in intervals
                )
                negative = all(hi <= 0 for _, hi in intervals) and any(
                    hi <= -tiny for _, hi in intervals
                )
                if not (positive or negative):
                    return False
            else:
                return False

    return True


def _has_control(node: _Node) -> bool:
    """Identifies region-bearing nodes that cannot use an elementwise map."""
    if node[0] in {"select", "symbol", "target_read"} or node[0].startswith(
        "predicate:"
    ):
        return True
    if node[0] in {"const", "array"}:
        return False
    return any(_has_control(a) for a in node[1:] if isinstance(a, tuple))


def _requires_restricted_domain(node: _Node) -> bool:
    """Identifies arithmetic whose inactive derivatives may be nonfinite.

    Total elementwise transforms can share the full source array. Partial
    transforms must evaluate only coordinates used by their access region.
    Keep powers conservative even when a particular exponent is total.
    """
    if node[0] in {"sqrt", "div", "pow"}:
        return True
    if node[0] in {"const", "array"}:
        return False
    return any(_requires_restricted_domain(a) for a in node[1:] if isinstance(a, tuple))


def kernel_schedule_summary(plan: KernelPlan, size: int) -> dict[str, Any]:
    """Describes structural scheduling decisions without compiling a device kernel.

    Args:
        plan: Immutable structured kernel plan.
        size: Number of configuration coordinates.

    Returns:
        Selected coordinate strategy, DAG-family count, and explicit axis names.
    """
    strategy, groups = _coordinate_schedule(plan, size)
    return {
        "target_strategy": strategy,
        "coordinate_families": len(groups),
        "axes": ("batch", "action", "coordinate"),
        "unit_wrap": "dtype_guarded_select",
        "precision_fallback": "typed_subnormal_constant_subgraphs",
    }


def make_kernel_function(
    plan: KernelPlan,
    size: int,
    dtype: Any,
    capacity: int,
    reference_factory: Callable[[], Callable[[Any], tuple[Any, Any]]] | None = None,
) -> Callable[[Any], tuple[Any, Any]]:
    """Generates a batched connection function from a structured kernel plan.

    Args:
        plan: Scalar DAGs, final coordinate maps, and exact assembly obligations.
        size: Number of configuration coordinates.
        dtype: Matrix-element output dtype.
        capacity: Proven padded output size.
        reference_factory: Lazy reference kernel for precision-sensitive cases.

    Returns:
        JIT-compatible function supporting arbitrary batch leading dimensions.
    """
    # Import lazily: the generic interpreter is a fallback, not a dependency of
    # the planning IR or its module initialization.
    from neuralqx.operator.symbolic.compiler.lowering.jax_lowerer import (
        _apply_update_program,
    )

    count = len(plan.representatives)
    branch_to_target = {
        branch: position
        for position, group in enumerate(
            plan.connectivity.groups
            if plan.connectivity
            else tuple((i,) for i in range(len(plan.actions)))
        )
        for branch in group
    }
    strategy, coordinate_groups = _coordinate_schedule(plan, size)
    runners = tuple(
        _scalar_runner(batch, plan, dtype, branch_to_target)
        for batch in plan.scalar_batches
    )
    order = tuple(i for b in plan.scalar_batches for i in b.actions)
    restore = np.argsort(order).astype(np.int32)
    connectivity = plan.connectivity
    finish = make_grouped_coalescer(connectivity) if connectivity is not None else None
    specialized_finish: dict[
        tuple[str, bool, bool], Callable[[Any, Any], tuple[Any, Any]]
    ] = {}
    precision_fallbacks: dict[tuple[str, bool], Callable[[Any], tuple[Any, Any]]] = {}
    precision_decisions: dict[tuple[str, bool], bool] = {}

    def precision_sensitive(source_dtype: Any) -> bool:
        key = (np.dtype(source_dtype).str, bool(jax.config.jax_enable_x64))
        if key in precision_decisions:
            return precision_decisions[key]
        types = (np.dtype(source_dtype), np.dtype(jax.dtypes.canonicalize_dtype(dtype)))
        tiny = max(
            (
                np.finfo(np.empty((), dtype=d).real.dtype).tiny
                for d in types
                if d.kind in "fc"
            ),
            default=0,
        )

        decisions: dict[_Node, tuple[bool, bool]] = {}
        constants: dict[_Node, Any] = {}
        dummy = np.empty((0,), dtype=source_dtype)

        def tiny_components(value: np.ndarray[Any, Any]) -> bool:
            if value.dtype.kind not in "fc":
                return False
            components = np.concatenate(
                (value.real.reshape(-1), value.imag.reshape(-1))
            )
            return bool(np.any((abs(components) > 0) & (abs(components) < tiny)))

        def inspect(node: _Node) -> tuple[bool, bool]:
            if node in decisions:
                return decisions[node]
            if node[0] in {"read", "input", "target_read", "symbol"}:
                return False, False
            children = (
                []
                if node[0] in {"const", "array"}
                else [inspect(arg) for arg in node[1:] if isinstance(arg, tuple)]
            )
            static = all(child[0] for child in children)
            sensitive = any(child[1] for child in children)
            if node[0] in {"const", "array"}:
                sensitive |= tiny_components(np.asarray(_constant_value(node)))
            if static and not sensitive:
                value = np.asarray(
                    _evaluate_node(node, dummy, (), None, {}, plan, constants)
                )
                if value.dtype.kind in "fc":
                    sensitive = tiny_components(value)
                    # Eager execution may already flush a computed subnormal,
                    # while XLA's constant folder retains it. Detect lost
                    # nonzero results as well as representable tiny values.
                    if node[0] in {"mul", "div", "pow", "add", "sub"}:
                        left, right = (
                            np.asarray(constants[child]) for child in node[1:]
                        )
                        nonzero = (
                            (left != 0) & (right != 0)
                            if node[0] in {"mul", "div", "pow"}
                            else left != (-right if node[0] == "add" else right)
                        )
                        sensitive |= bool(np.any((value == 0) & nonzero))
            decisions[node] = static, sensitive
            return static, sensitive

        with jax.ensure_compile_time_eval():
            result = any(
                inspect(n)[1]
                for action in plan.actions
                for n in (
                    *action.expressions,
                    *(v for _, v in action.target.coordinates),
                )
            )
        precision_decisions[key] = result
        return result

    widths: dict[int, list[int]] = {}
    if connectivity is not None:
        for i, group in enumerate(connectivity.groups):
            widths.setdefault(len(group), []).append(i)
    buckets = (
        tuple(
            restore[np.asarray([connectivity.groups[i] for i in ids], np.int32)]
            for ids in widths.values()
        )
        if connectivity
        else ()
    )
    group_restore = np.argsort([i for ids in widths.values() for i in ids]).astype(
        np.int32
    )

    def materialize(flat: Any, scalar_source: bool) -> Any:
        xp = jnp.broadcast_to(flat[:, None, :], (flat.shape[0], count, size))
        values: list[Any] = []
        positions: list[int] = []
        sites: list[int] = []
        shared: dict[tuple[int, ...], dict[_Node, Any]] = {}
        for template, members in coordinate_groups.items():
            if strategy == "dense_select":
                mask = np.zeros((count, size), dtype=bool)
                for position, site, _, _ in members:
                    mask[position, site] = True
                if members[0][3]:
                    used = (
                        tuple(map(int, np.flatnonzero(np.any(mask, axis=0))))
                        if _requires_restricted_domain(template[0])
                        else tuple(range(size))
                    )
                    source = flat if len(used) == size else flat[:, np.asarray(used)]
                    cache = shared.setdefault(used, {})
                    transformed = _evaluate_node(
                        template[0], flat, (source,), None, {}, plan, cache
                    )
                    if len(used) != size:
                        expansion = np.zeros(size, dtype=np.int32)
                        expansion[np.asarray(used)] = np.arange(len(used))
                        transformed = transformed[:, expansion]
                else:
                    transformed = _evaluate_node(
                        template[0], flat, (), None, {}, plan, shared.setdefault((), {})
                    )
                    transformed = jnp.broadcast_to(transformed, flat.shape)
                xp = jnp.where(
                    jnp.asarray(mask)[None, :, :], transformed[:, None, :], xp
                )
            else:
                table = tuple(m[3] for m in members)
                index_array = jnp.asarray(table, dtype=jnp.int32)

                def source(
                    x: Any, table=table, template=template, index_array=index_array
                ) -> Any:
                    def coordinate(row: Any) -> Any:
                        inputs = _source_inputs(x, row, table)
                        return _evaluate_node(
                            template[0], x, inputs, None, {}, plan, {}
                        )

                    return jax.vmap(coordinate)(index_array)

                values.append(jax.vmap(source)(flat))
                positions.extend(m[0] for m in members)
                sites.extend(m[1] for m in members)
        if values:
            xp = xp.at[:, np.asarray(positions), np.asarray(sites)].set(
                jnp.concatenate(values, axis=1), unique_indices=True
            )
        for position, representative in enumerate(plan.representatives):
            action = plan.actions[representative]
            preserve_region = not action.mapped and any(
                _has_control(n) for _, n in action.target.coordinates
            )
            if action.target.opaque is not None or preserve_region:

                def fallback(x: Any, a: KernelAction = action) -> Any:
                    env: dict[str, Any] = {"__x__": x}
                    for label, site in a.bindings:
                        env[f"site:{label}:index"] = jnp.int32(site)
                        env[f"site:{label}:value"] = x[site]
                    return _apply_update_program(
                        x,
                        a.program,
                        env,
                        size,
                        shift_mod_state_min=(
                            plan.wrap_spec[0] if plan.wrap_spec else None
                        ),
                        shift_mod_mod_span=(
                            plan.wrap_spec[1] if plan.wrap_spec else None
                        ),
                    )[0]

                target = (
                    fallback(flat[0])[None]
                    if scalar_source and not action.mapped
                    else jax.vmap(fallback)(flat)
                )
                xp = xp.at[:, position, :].set(target)
        return xp

    def amplitudes(x: Any, xp: Any) -> Any:
        raw = jnp.concatenate([run(x, xp) for run in runners])
        if connectivity is None:
            return _static_take(raw, restore)
        parts = []
        for members in buckets:
            values = _static_take(raw, members)
            parts.append(
                values[:, 0] if members.shape[1] == 1 else jnp.sum(values, axis=1)
            )
        return _static_take(jnp.concatenate(parts), group_restore)

    @jax.jit
    def execute(x: Any) -> tuple[Any, Any]:
        if x.ndim < 1 or x.shape[-1] != size:
            raise NeuralqxValueError(
                f"Expected configurations with final axis of length {size}."
            )
        # Constant folding and runtime comparisons can treat subnormals
        # differently on a flushing backend. Retain the original graph rather
        # than changing sparsity through a materialization optimization.
        if precision_sensitive(x.dtype):
            if reference_factory is None:
                raise NeuralqxValueError(
                    "Precision-sensitive constants require a reference kernel factory."
                )
            precision_key = (np.dtype(x.dtype).str, bool(jax.config.jax_enable_x64))
            if precision_key not in precision_fallbacks:
                precision_fallbacks[precision_key] = reference_factory()
            return precision_fallbacks[precision_key](x)
        leading = x.shape[:-1]
        flat = x.reshape((-1, size))
        if not count:
            return (
                jnp.zeros((*leading, capacity, size), x.dtype),
                jnp.zeros((*leading, capacity), dtype),
            )
        if finish is not None:
            assert connectivity is not None
            weak = bool(getattr(x, "weak_type", False))
            key = (np.dtype(x.dtype).str, bool(jax.config.jax_enable_x64), weak)
            if key not in specialized_finish:
                if (
                    not weak
                    and connectivity.compact
                    and _nonzero_groups(plan, x.dtype, dtype)
                ):
                    specialized_finish[key] = make_grouped_coalescer(
                        dataclasses.replace(connectivity, compact=False)
                    )
                else:
                    specialized_finish[key] = finish
        xp = materialize(flat, x.ndim == 1)
        # Preserve unmapped global conditional semantics for a scalar source.
        mels = (
            amplitudes(x, xp[0])[None]
            if x.ndim == 1
            else jax.vmap(amplitudes)(flat, xp)
        )
        if finish is not None:
            xp, mels = jax.vmap(specialized_finish[key])(xp, mels)
        elif count < capacity:
            xp = jnp.pad(xp, ((0, 0), (0, capacity - count), (0, 0)))
            mels = jnp.pad(mels, ((0, 0), (0, capacity - count)))
        elif count > capacity:
            xp, mels = xp[:, :capacity], mels[:, :capacity]
        return xp.reshape((*leading, capacity, size)), mels.reshape(
            (*leading, capacity)
        )

    return execute
