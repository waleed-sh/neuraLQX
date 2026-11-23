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


"""Static target-equivalence and collision analysis for connectivity lowering.

Update programs are partially evaluated with static iterator bindings. Their
sparse target expressions form value-numbered equivalence classes. A bounded
finite-domain analysis then proves which classes cannot collide. Unproved
relations remain runtime obligations; sampled states never justify an output
bound or an optimization.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from neuralqx.operator.symbolic._hilbert import uniform_local_values
from neuralqx.operator.symbolic.compiler.core.actions import _UNKNOWN
from neuralqx.operator.symbolic.compiler.core.actions import _dependencies
from neuralqx.operator.symbolic.compiler.core.actions import _evaluate
from neuralqx.operator.symbolic.compiler.core.actions import _expression
from neuralqx.operator.symbolic.compiler.core.actions import _literal
from neuralqx.operator.symbolic.compiler.core.actions import _Node
from neuralqx.operator.symbolic.compiler.core.actions import _Target
from neuralqx.operator.symbolic.compiler.core.actions import _target
from neuralqx.operator.symbolic.compiler.core.connectivity import (
    ConnectivityDeduplicationPlan,
)
from neuralqx.operator.symbolic.compiler.core.context import SymbolicCompilationContext
from neuralqx.operator.symbolic.compiler.passes.base import AbstractSymbolicPass
from neuralqx.operator.symbolic.ir.program import SymbolicOperatorIR

_MAX_LOCAL_ASSIGNMENTS = 256
_MAX_PROOF_ASSIGNMENTS = 100_000
_MAX_COLLISION_PAIRS = 65_536


def _copy_image(node: _Node, local_states: Any) -> frozenset[Any] | None:
    """Bounds the image of copying, constant, and conditional target expressions.

    Configuration-dtype casts cannot increase this image's cardinality. General
    arithmetic is excluded because intermediate rounding can split symbolic
    equalities, even when all source basis values are finite.
    """
    if node[0] == "read":
        return None if local_states is None else frozenset(local_states)
    if node[0] == "const":
        value = _literal(node)
        return (
            frozenset((value,))
            if isinstance(value, (int, float, complex, bool))
            else None
        )
    if node[0] == "cast":
        return _copy_image(node[1], local_states)
    if node[0] == "select":
        left, right = _copy_image(node[2], local_states), _copy_image(
            node[3], local_states
        )
        return None if left is None or right is None else left | right
    return None


def _component_capacity(
    component: tuple[int, ...],
    columns: tuple[int, ...],
    targets: tuple[_Target, ...],
    local_states: Any,
) -> int:
    bound = 1
    if any(targets[i].opaque is not None for i in component):
        return len(component)
    for site in columns:
        image: frozenset[Any] = frozenset()
        for i in component:
            values = _copy_image(targets[i].value(site), local_states)
            if values is None:
                return len(component)
            image |= values
        bound *= len(image)
        if bound >= len(component):
            return len(component)
    return min(len(component), bound)


def _key_ranges(
    component: tuple[int, ...],
    columns: tuple[int, ...],
    targets: tuple[_Target, ...],
    proofs: _CollisionProofs,
) -> tuple[tuple[int, int], ...] | None:
    """Derives lossless bounded-integer keys without hashing target configurations."""
    if any(targets[i].opaque is not None for i in component):
        return None
    ranges = []
    for site in columns:
        image: list[int] = []
        for i in component:
            node = targets[i].value(site)
            affine = proofs.affine(node)
            if affine is not None:
                lo, hi = proofs.affine_interval(*affine)
                if not -(2**63) <= lo <= hi < 2**63:
                    return None
                image.extend((lo, hi))
                continue
            dependencies = sorted(_dependencies(node))
            if dependencies and not proofs.domain:
                return None
            count = len(proofs.domain) ** len(dependencies)
            if count > min(_MAX_LOCAL_ASSIGNMENTS, proofs.remaining):
                return None
            proofs.remaining -= count
            for row in itertools.product(proofs.domain, repeat=len(dependencies)):
                value = _evaluate(
                    node, dict(zip(dependencies, row, strict=False)), proofs.bounds
                )
                if value is _UNKNOWN or not -(2**63) <= value < 2**63:
                    return None
                image.append(int(value))
        lo, hi = min(image), max(image)
        if hi - lo >= 2**32:
            return None
        ranges.append((lo, hi))
    return tuple(ranges)


class _CollisionProofs:
    """Affine intervals and bounded finite domains with integer range guards."""

    def __init__(self, local_states: Any) -> None:
        self.domain: tuple[int, ...] = ()
        self.interval: tuple[int, int] | None = None
        if local_states is not None:
            values = np.asarray(local_states)
            if values.ndim == 1 and len(values) > 0:
                if np.all(np.isfinite(values)) and np.all(values == np.floor(values)):
                    self.interval = (int(np.min(values)), int(np.max(values)))
                    if len(values) <= 32:
                        self.domain = tuple(map(int, values))
        self.bounds = [0, 0, 2**63]
        self.remaining = _MAX_PROOF_ASSIGNMENTS
        self.affine_cache: dict[_Node, tuple[dict[int, int], int] | None] = {}

    def affine_interval(
        self, coefficients: Mapping[int, int], bias: int
    ) -> tuple[int, int]:
        """Bounds a normalized affine expression over the local basis interval."""
        lower = upper = bias
        assert self.interval is not None or not coefficients
        for coefficient in coefficients.values():
            assert self.interval is not None
            lo, hi = self.interval
            lower += coefficient * (lo if coefficient >= 0 else hi)
            upper += coefficient * (hi if coefficient >= 0 else lo)
        return lower, upper

    def affine(self, node: _Node) -> tuple[dict[int, int], int] | None:
        """Normalizes integer affine expressions without changing target keys.

        Every original intermediate contributes to the dtype guard. Affine
        normalization is used only to prove disjointness; it never merges
        structurally distinct programs or reassociates their emitted arithmetic.
        """
        if node in self.affine_cache:
            return self.affine_cache[node]
        result = None
        kind = node[0]
        if kind == "read" and self.interval is not None:
            result = ({int(node[1]): 1}, 0)
        elif kind == "const":
            value = _evaluate(node, {}, self.bounds)
            if value is not _UNKNOWN:
                result = ({}, int(value))
        elif kind == "cast":
            result = self.affine(node[1])
        elif kind == "neg":
            operand = self.affine(node[1])
            if operand is not None:
                coefficients, bias = operand
                result = ({site: -value for site, value in coefficients.items()}, -bias)
        elif kind in {"add", "sub", "mul"}:
            left, right = self.affine(node[1]), self.affine(node[2])
            if left is not None and right is not None:
                a, b = left
                c, d = right
                if kind in {"add", "sub"}:
                    sign = 1 if kind == "add" else -1
                    coefficients = {
                        site: a.get(site, 0) + sign * c.get(site, 0)
                        for site in a.keys() | c.keys()
                    }
                    result = (
                        {site: value for site, value in coefficients.items() if value},
                        b + sign * d,
                    )
                elif not a:
                    result = (
                        {site: b * value for site, value in c.items() if b * value},
                        b * d,
                    )
                elif not c:
                    result = (
                        {site: d * value for site, value in a.items() if d * value},
                        b * d,
                    )
        if result is not None:
            lower, upper = self.affine_interval(*result)
            self.bounds[0], self.bounds[1] = min(self.bounds[0], lower), max(
                self.bounds[1], upper
            )
        self.affine_cache[node] = result
        return result

    def different(self, a: _Node, b: _Node, predicate: _Node) -> bool:
        if a == b:
            return False
        if self.remaining > 0:
            left_affine, right_affine = self.affine(a), self.affine(b)
            if left_affine is not None and right_affine is not None:
                left_coefficients, left_bias = left_affine
                right_coefficients, right_bias = right_affine
                coefficients = {
                    site: left_coefficients.get(site, 0)
                    - right_coefficients.get(site, 0)
                    for site in left_coefficients.keys() | right_coefficients.keys()
                }
                coefficients = {
                    site: value for site, value in coefficients.items() if value
                }
                lower, upper = self.affine_interval(
                    coefficients, left_bias - right_bias
                )
                if lower > 0 or upper < 0:
                    return True
        dependencies = sorted(_dependencies(a) | _dependencies(b))
        if dependencies and not self.domain:
            return False
        count = len(self.domain) ** len(dependencies)
        if count > min(_MAX_LOCAL_ASSIGNMENTS, self.remaining):
            return False
        self.remaining -= count
        for row in itertools.product(self.domain, repeat=len(dependencies)):
            values = dict(zip(dependencies, row, strict=False))
            active = _evaluate(predicate, values, self.bounds)
            if active is not _UNKNOWN and not active:
                continue
            left = _evaluate(a, values, self.bounds)
            right = _evaluate(b, values, self.bounds)
            if left is _UNKNOWN or right is _UNKNOWN or left == right:
                return False
        return True


def _always(predicate: _Node, value: bool) -> bool:
    bounds = [0, 0, 2**63]
    result = _evaluate(predicate, {}, bounds)
    exact = -bounds[2] <= bounds[0] <= bounds[1] <= bounds[2]
    return exact and result is not _UNKNOWN and bool(result) == value


def analyze_connectivity(
    ir: SymbolicOperatorIR, hilbert: Any
) -> ConnectivityDeduplicationPlan:
    """Builds a sound assembly plan without executing a JAX connectivity kernel.

    Args:
        ir: Validated symbolic operator IR.
        hilbert: Hilbert space supplying the finite local basis when available.

    Returns:
        Target-equivalence classes, residual collision components, and a reduced
        static output bound.
    """
    classes: dict[_Target, list[int]] = {}
    predicates: dict[_Target, list[_Node]] = {}
    nonzero: dict[_Target, list[bool]] = {}
    constants: dict[_Target, list[Any]] = {}
    raw_size = 0
    wrap_spec = None
    local_states = uniform_local_values(hilbert)
    if local_states is not None:
        states = np.asarray(local_states)
        if (
            states.ndim == 1
            and len(states) > 0
            and np.all(np.isfinite(states))
            and np.array_equal(states, np.arange(states[0], states[0] + len(states)))
            and states[0] == int(states[0])
        ):
            wrap_spec = (int(states[0]), len(states))
    metadata_spec = ir.metadata_dict().get("shift_mod_spec")
    if (
        metadata_spec is not None
        and metadata_spec["version"] == "uniform_integer_wrap_v1"
    ):
        wrap_spec = (int(metadata_spec["state_min"]), int(metadata_spec["mod_span"]))
    for term in ir.terms:
        for row in term.iterator.index_sets if term.iterator.labels else ((),):
            bindings = dict(zip(term.iterator.labels, row, strict=False))
            for emission in term.effective_emissions:
                branch = raw_size
                raw_size += 1
                predicate = (
                    "predicate:and",
                    _expression(term.predicate, bindings),
                    _expression(emission.predicate, bindings),
                )
                literal = _literal(_expression(emission.amplitude, bindings))
                if (
                    emission.update_program.has_invalidate()
                    or _always(predicate, False)
                    or (literal is not _UNKNOWN and literal == 0)
                ):
                    continue
                target = _target(
                    emission.update_program.ops, bindings, ir.hilbert_size, wrap_spec
                )
                classes.setdefault(target, []).append(branch)
                predicates.setdefault(target, []).append(predicate)
                # Only exact, small integral constants establish nonvanishing
                # amplitudes without a runtime check.
                known_nonzero = (
                    literal is not _UNKNOWN
                    and isinstance(literal, (int, float))
                    and math.isfinite(literal)
                    and literal == int(literal)
                    and 0 < abs(literal) <= 1024
                    and _always(predicate, True)
                )
                nonzero.setdefault(target, []).append(known_nonzero)
                constants.setdefault(target, []).append(
                    literal if _always(predicate, True) else _UNKNOWN
                )

    # Two exactly opposite constants cancel in every configuration dtype and
    # under every reduction ordering. Larger floating reductions are retained;
    # reassociation must not justify silently removing a potentially nonzero sum.
    for target in tuple(classes):
        values = constants[target]
        if len(values) == 2 and all(value is not _UNKNOWN for value in values):
            a, b = values
            if np.isfinite(a) and np.isfinite(b) and a == -b:
                del classes[target]

    targets = tuple(classes)
    groups = tuple(tuple(classes[t]) for t in targets)
    guards = tuple(("predicate:or", *predicates[t]) for t in targets)
    proofs = _CollisionProofs(uniform_local_values(hilbert))
    parent = list(range(len(targets)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a: int, b: int) -> None:
        parent[root(b)] = root(a)

    supports = []
    must_change = []
    for target, guard in zip(targets, guards, strict=False):
        sites = frozenset(s for s, _ in target.coordinates)
        supports.append(
            sites if target.opaque is None else frozenset(range(ir.hilbert_size))
        )
        must_change.append(
            frozenset(
                s
                for s, v in target.coordinates
                if proofs.different(v, ("read", s), guard)
            )
        )

    # Exact changed-site sets partition the collision graph before pairwise
    # reasoning. This keeps large local-operator families linear in site count.
    buckets: dict[frozenset[int], list[int]] = {}
    uncertain = []
    for i, sites in enumerate(supports):
        if sites == must_change[i] and targets[i].opaque is None:
            buckets.setdefault(sites, []).append(i)
        else:
            uncertain.append(i)
    uncertain_ids = frozenset(uncertain)
    candidates = itertools.chain(
        *(itertools.combinations(bucket, 2) for bucket in buckets.values()),
        ((a, b) for a in uncertain for b in range(a)),
        (
            (a, b)
            for a in uncertain
            for b in range(a + 1, len(targets))
            if b not in uncertain_ids
        ),
    )
    examined = 0
    for a, b in candidates:
        examined += 1
        if examined > _MAX_COLLISION_PAIRS:
            # The fallback remains exact when analysis reaches its work budget.
            for i in range(1, len(targets)):
                join(0, i)
            break
        if must_change[a] - supports[b] or must_change[b] - supports[a]:
            continue
        left, right = targets[a], targets[b]
        guard = ("predicate:and", guards[a], guards[b])
        if left.opaque is None and right.opaque is None:
            if any(
                proofs.different(left.value(s), right.value(s), guard)
                for s in sorted(supports[a] | supports[b])
            ):
                continue
        join(a, b)

    components: dict[int, list[int]] = {}
    for i in range(len(targets)):
        components.setdefault(root(i), []).append(i)
    collisions = tuple(tuple(c) for c in components.values() if len(c) > 1)
    columns = tuple(
        tuple(sorted(frozenset().union(*(supports[i] for i in c)))) for c in collisions
    )
    capacities = tuple(
        _component_capacity(c, cols, targets, uniform_local_values(hilbert))
        for c, cols in zip(collisions, columns, strict=False)
    )
    key_ranges = tuple(
        _key_ranges(c, cols, targets, proofs)
        for c, cols in zip(collisions, columns, strict=False)
    )
    compact = bool(collisions) or any(
        len(nonzero[t]) != 1 or not all(nonzero[t]) for t in targets
    )
    return ConnectivityDeduplicationPlan(
        raw_size=raw_size,
        groups=groups,
        collision_components=collisions,
        key_columns=columns,
        component_capacities=capacities,
        key_ranges=key_ranges,
        compact=compact,
        proof_min=proofs.bounds[0],
        proof_max=proofs.bounds[1],
        proof_integer_limit=proofs.bounds[2],
        output_size=max(1, len(groups) - sum(map(len, collisions)) + sum(capacities)),
    )


class SymbolicDeduplicationPass(AbstractSymbolicPass):
    """Plans target aggregation and derives the deduplicated output bound.

    Analysis keys written:
        ``"connectivity_deduplication_plan"`` - immutable assembly plan.
        ``"raw_max_conn_size"`` - output bound without deduplication.
        ``"total_max_conn_size"`` - reduced bound used by the lowerer.
    """

    @property
    def name(self) -> str:
        return "symbolic_deduplication"

    def run(self, context: SymbolicCompilationContext) -> Mapping[str, Any] | None:
        if (
            not context.options.deduplicate_connected_components
            or context.ir.mode != "symbolic"
        ):
            return {"enabled": False}
        plan = analyze_connectivity(context.ir, context.operator.hilbert)
        context.set_analysis("connectivity_deduplication_plan", plan)
        context.set_analysis(
            "raw_max_conn_size", context.analysis("total_max_conn_size")
        )
        context.set_analysis("total_max_conn_size", plan.output_size)
        context.set_analysis("total_fanout", plan.output_size)
        return {"enabled": True, **plan.as_dict()}


__all__ = [
    "ConnectivityDeduplicationPlan",
    "SymbolicDeduplicationPass",
    "analyze_connectivity",
]
