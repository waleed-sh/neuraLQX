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


"""Static assembly-cost diagnostics for deduplicated connectivity kernels."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any

from neuralqx.operator.symbolic.compiler.core.connectivity import (
    ConnectivityDeduplicationPlan,
)
from neuralqx.operator.symbolic.compiler.core.connectivity import collision_strategy
from neuralqx.operator.symbolic.compiler.core.connectivity import packed_key_width
from neuralqx.operator.symbolic.compiler.core.context import SymbolicCompilationContext
from neuralqx.operator.symbolic.compiler.diagnostics import (
    filter_diagnostics_by_minimum_severity,
)
from neuralqx.operator.symbolic.compiler.diagnostics import format_diagnostics_block
from neuralqx.operator.symbolic.compiler.diagnostics import (
    normalize_diagnostic_severity,
)
from neuralqx.operator.symbolic.compiler.diagnostics.models import DSLDiagnostic
from neuralqx.operator.symbolic.compiler.passes.base import AbstractSymbolicPass
from neuralqx.operator.symbolic.compiler.passes.diagnostics import (
    _diagnostics_summary_payload,
)
from neuralqx.operator.symbolic.errors import SymbolicDiagnosticsError


def estimate_connectivity_cost(
    plan: ConnectivityDeduplicationPlan, hilbert_size: int
) -> dict[str, Any]:
    """Estimates assembly work in scalar-operation units, excluding amplitudes.

    Args:
        plan: Static connectivity assembly plan.
        hilbert_size: Number of state coordinates.

    Returns:
        Raw and deduplicated work estimates and the selected runtime strategies.
        These estimates describe connectivity assembly, not wall-clock latency
        or the cost of evaluating a variational model on the output states.
    """
    classes = len(plan.groups)
    raw = max(1, plan.raw_size * hilbert_size)
    cost = classes * hilbert_size + max(0, sum(map(len, plan.groups)) - classes)
    strategies = []
    components = tuple(
        zip(plan.collision_components, plan.key_columns, plan.key_ranges, strict=False)
    )
    usable = (
        -plan.proof_integer_limit
        <= plan.proof_min
        <= plan.proof_max
        <= plan.proof_integer_limit
    )
    if not usable and classes > 1:
        components = ((tuple(range(classes)), tuple(range(hilbert_size)), None),)
    for component, columns, ranges in components:
        packed_width = packed_key_width(ranges) if ranges is not None else len(columns)
        size, width = len(component), max(1, packed_width)
        if ranges is not None:
            cost += 2 * size * len(columns)
        key_bits = (
            sum((hi - lo).bit_length() for lo, hi in ranges)
            if ranges is not None
            else None
        )
        strategy = collision_strategy(size, width, key_bits=key_bits)
        strategies.append(strategy)
        if strategy == "workspace":
            assert key_bits is not None
            cost += 2 * size + 3 * 2**key_bits
        elif strategy == "pairwise":
            cost += size * size * width + 2 * size
        else:
            cost += size * math.ceil(math.log2(size)) * width + 5 * size
    if plan.compact or not usable:
        cost += classes * (hilbert_size + 4)
    return {
        "raw_assembly_work": raw,
        "deduplicated_assembly_work": cost,
        "estimated_assembly_ratio": cost / raw,
        "runtime_strategies": tuple(strategies),
    }


class SymbolicConnectivityCostPass(AbstractSymbolicPass):
    """Warns when the assembly cost model predicts a deduplication penalty.

    Analysis keys written:
        ``"connectivity_cost"`` - deterministic cost estimates and strategies.
        ``"dsl_diagnostics"`` - lint findings including ``NQXDSL-W401`` when shown.

    The pass preserves the requested deduplication semantics. Diagnostics obey
    the same visibility and strictness options as the DSL diagnostics pass.
    """

    @property
    def name(self) -> str:
        return "symbolic_connectivity_cost"

    def run(self, context: SymbolicCompilationContext) -> Mapping[str, Any] | None:
        plan = context.analysis("connectivity_deduplication_plan")
        if not context.options.deduplicate_connected_components or plan is None:
            return {"enabled": False}
        cost = estimate_connectivity_cost(plan, context.ir.hilbert_size)
        context.set_analysis("connectivity_cost", cost)
        if (
            not context.options.diagnostics_enabled
            or cost["deduplicated_assembly_work"] <= cost["raw_assembly_work"]
        ):
            return {"enabled": True, **cost}
        diagnostic = DSLDiagnostic.create(
            code="NQXDSL-W401",
            severity="warning",
            operator_name=context.ir.operator_name,
            message=(
                "Deduplication is estimated to increase connectivity assembly work "
                f"({plan.raw_size} raw branches, {plan.output_size} output slots, "
                f"{len(cost['runtime_strategies'])} runtime collision components). "
                "Actual latency depends on the backend, batch size, and input states."
            ),
            suggestion=(
                "Benchmark get_conn_padded with deduplicate_connected_components=False "
                "if retaining repeated targets is acceptable. Fewer output states can "
                "still reduce the cost of downstream model evaluation."
            ),
            context={**plan.as_dict(), **cost},
        )
        visible = filter_diagnostics_by_minimum_severity(
            (diagnostic,),
            minimum=normalize_diagnostic_severity(
                context.options.diagnostics_min_severity
            ),
        )
        # Keep one coherent diagnostic payload for custom passes and artifacts.
        existing = context.analysis("dsl_diagnostics", default=())
        previous = tuple(DSLDiagnostic(**entry) for entry in existing)
        all_diagnostics = ((*previous, diagnostic))[: context.options.max_diagnostics]
        all_visible = filter_diagnostics_by_minimum_severity(
            all_diagnostics,
            minimum=normalize_diagnostic_severity(
                context.options.diagnostics_min_severity
            ),
        )
        summary = _diagnostics_summary_payload(
            all_diagnostics=all_diagnostics,
            visible_diagnostics=all_visible,
        )
        summary.update(
            enabled=True, minimum_severity=context.options.diagnostics_min_severity
        )
        context.set_analysis("dsl_diagnostics", summary["diagnostics"])
        context.set_analysis("dsl_diagnostics_summary", summary)
        if visible and diagnostic in all_diagnostics:
            message = (
                f"Symbolic diagnostics for operator {context.ir.operator_name!r}:\n"
                f"{format_diagnostics_block(visible)}"
            )
            if context.options.fail_on_warnings:
                raise SymbolicDiagnosticsError(message)
            warnings.warn(message, category=UserWarning, stacklevel=3)
        return {"enabled": True, **cost}


__all__ = ["SymbolicConnectivityCostPass", "estimate_connectivity_cost"]
