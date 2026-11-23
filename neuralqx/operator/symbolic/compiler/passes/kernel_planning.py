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

"""Structured kernel planning from ordered actions, not Hamiltonian names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from neuralqx.operator.symbolic._hilbert import uniform_local_values
from neuralqx.operator.symbolic.compiler.core.actions import _constant
from neuralqx.operator.symbolic.compiler.core.actions import _expression
from neuralqx.operator.symbolic.compiler.core.actions import _Node
from neuralqx.operator.symbolic.compiler.core.actions import _Target
from neuralqx.operator.symbolic.compiler.core.actions import _target
from neuralqx.operator.symbolic.compiler.core.context import SymbolicCompilationContext
from neuralqx.operator.symbolic.compiler.core.kernel import KernelAction
from neuralqx.operator.symbolic.compiler.core.kernel import KernelPlan
from neuralqx.operator.symbolic.compiler.core.kernel import ScalarBatch
from neuralqx.operator.symbolic.compiler.core.kernel import scalar_template
from neuralqx.operator.symbolic.compiler.passes.base import AbstractSymbolicPass
from neuralqx.operator.symbolic.ir.expressions import AmplitudeExpr
from neuralqx.operator.symbolic.ir.expressions import parse_symbol_declaration_args

_MAX_ACTIONS = 65_536
_MAX_SCALAR_BATCHES = 512


def _amplitude_node(
    expr: Any, bindings: Mapping[str, int], target: _Target, branch: int, size: int
) -> _Node:
    """Resolves emitted reads against final coordinates, retaining all casts."""

    def emitted_value(site: int) -> _Node:
        if target.opaque is not None:
            return ("target_read", branch, site)
        # Match JAX getitem: normalize negatives, then clip invalid reads.
        # Diagnostics still report out-of-bounds static selectors.
        normalized = (
            min(max(site + size if site < 0 else site, 0), size - 1) if size else site
        )
        return target.value(normalized)

    if isinstance(expr, AmplitudeExpr):
        site = None
        if expr.op == "static_emitted_index":
            site = int(expr.args[0])
        elif expr.op == "symbol":
            name, declaration = parse_symbol_declaration_args(expr.args)
            parts = name.split(":")
            if len(parts) == 3 and parts[0] == "emit" and parts[1] in bindings:
                site = bindings[parts[1]]
                node = (
                    _constant(np.int32(site))
                    if parts[2] == "index"
                    else (emitted_value(site))
                )
                return (
                    ("convert", declaration["dtype"], node)
                    if "dtype" in declaration
                    else node
                )
        if site is not None:
            return emitted_value(site)
        if expr.op in {"const", "symbol", "static_index"}:
            return _expression(expr, bindings)
        return (
            expr.op,
            *(
                _amplitude_node(arg, bindings, target, branch, size)
                for arg in expr.args
            ),
        )
    return _constant(expr)


class SymbolicKernelPlanningPass(AbstractSymbolicPass):
    """Plans scalar DAG batches and deferred target construction.

    The pass shares ordered symbolic execution with connectivity analysis.
    Static iterator bindings parameterize source reads, while constants, casts,
    guards, and amplitude reductions retain their original semantics. Dynamic
    update indices remain explicit fallback actions. Schedule decisions that
    require input shape or dtype are deferred to JAX specialization.

    Analysis keys written:
        ``"kernel_plan"``: Immutable :class:`~neuralqx.operator.symbolic.compiler.core.kernel.KernelPlan`.
        ``"kernel_planning"``: Bounded, JSON-compatible planning summary.
    """

    @property
    def name(self) -> str:
        """Returns the pass name used by compilation reports."""
        return "symbolic_kernel_planning"

    def run(self, context: SymbolicCompilationContext) -> Mapping[str, Any] | None:
        """Builds an executable plan without evaluating sampled configurations.

        Args:
            context: Validated compilation context and connectivity analyses.

        Returns:
            Planning summary, including any conservative resource fallback.
        """
        if not context.options.optimize_kernels or context.ir.mode != "symbolic":
            summary = {"enabled": False, "fallback_reason": "disabled"}
            context.set_analysis("kernel_planning", summary)
            return summary
        ir = context.ir
        local = uniform_local_values(context.operator.hilbert)
        basis = tuple(np.asarray(local).tolist()) if local is not None else ()
        spec = ir.metadata_dict().get("shift_mod_spec")
        if spec and spec["version"] == "site_domains_v2":
            summary = {"enabled": False, "fallback_reason": "per_site_modular_domains"}
            context.set_analysis("kernel_planning", summary)
            return summary
        wrap = (int(spec["state_min"]), int(spec["mod_span"])) if spec else None
        if (
            wrap is None
            and basis
            and np.array_equal(basis, np.arange(basis[0], basis[0] + len(basis)))
        ):
            if all(np.isfinite(v) and v == int(v) for v in basis):
                wrap = (int(basis[0]), len(basis))
        connectivity = context.analysis("connectivity_deduplication_plan")
        if context.options.deduplicate_connected_components and connectivity is None:
            summary = {
                "enabled": False,
                "fallback_reason": "connectivity_analysis_missing",
            }
            context.set_analysis("kernel_planning", summary)
            return summary
        actions: list[KernelAction] = []
        reason = None
        raw_count = sum(
            (len(t.iterator.index_sets) if t.iterator.labels else 1)
            * len(t.effective_emissions)
            for t in ir.terms
        )
        if raw_count > _MAX_ACTIONS:
            reason = "action_budget_exceeded"
        else:
            for ti, term in enumerate(ir.terms):
                rows = term.iterator.index_sets if term.iterator.labels else ((),)
                for ri, row in enumerate(rows):
                    bindings = dict(zip(term.iterator.labels, row, strict=False))
                    for ei, em in enumerate(term.effective_emissions):
                        target = _target(
                            em.update_program.ops, bindings, ir.hilbert_size, wrap
                        )
                        branch = len(actions)
                        actions.append(
                            KernelAction(
                                origin=(ti, ri, ei),
                                bindings=tuple(bindings.items()),
                                target=target,
                                expressions=(
                                    _expression(term.predicate, bindings),
                                    _expression(em.predicate, bindings),
                                    _amplitude_node(
                                        em.amplitude,
                                        bindings,
                                        target,
                                        branch,
                                        ir.hilbert_size,
                                    ),
                                ),
                                program=em.update_program,
                                mapped=bool(term.iterator.labels),
                                valid=not em.update_program.has_invalidate(),
                            )
                        )
        grouped: dict[Any, list[tuple[int, tuple[int, ...]]]] = {}
        templates: dict[Any, tuple[_Node, ...]] = {}
        for i, action in enumerate(actions):
            template, indices = scalar_template(action.expressions)
            # An unmapped global conditional must retain scalar cond semantics.
            key = (
                template,
                action.valid,
                action.mapped,
                None if action.mapped and context.options.enable_fusion else i,
            )
            templates[key] = template
            grouped.setdefault(key, []).append((i, indices))
        batches = tuple(
            ScalarBatch(
                tuple(i for i, _ in members),
                templates[key],
                tuple(indices for _, indices in members),
                bool(key[2]),
            )
            for key, members in grouped.items()
        )
        if sum(a.target.opaque is not None for a in actions) > 128:
            reason = "opaque_target_budget_exceeded"
        if len(batches) > _MAX_SCALAR_BATCHES:
            reason = "scalar_batch_budget_exceeded"
        representatives = (
            tuple(g[0] for g in connectivity.groups)
            if connectivity is not None
            else tuple(range(len(actions)))
        )
        if not actions:
            representatives = ()
        plan = KernelPlan(
            tuple(actions),
            batches,
            representatives,
            connectivity,
            basis,
            wrap,
            fallback_reason=reason,
        )
        context.set_analysis("kernel_plan", plan)
        summary = plan.as_dict()
        context.set_analysis("kernel_planning", summary)
        return summary


__all__ = ["SymbolicKernelPlanningPass"]
