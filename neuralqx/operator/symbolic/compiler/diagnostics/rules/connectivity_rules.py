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


"""Connectivity diagnostics rules based on lowered term-runner execution."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from neuralqx.operator.symbolic.compiler.diagnostics.models import DSLDiagnostic
from neuralqx.operator.symbolic.compiler.diagnostics.rules.base import (
    AbstractDiagnosticRule,
)
from neuralqx.operator.symbolic.compiler.diagnostics.rules.base import (
    DiagnosticRuleContext,
)
from neuralqx.operator.symbolic.compiler.diagnostics.state_sampling import (
    build_hilbert_support_lookup,
)
from neuralqx.operator.symbolic.compiler.diagnostics.state_sampling import (
    evaluate_constraint_accepts_state,
)
from neuralqx.operator.symbolic.compiler.diagnostics.state_sampling import (
    illegal_local_state_positions,
)
from neuralqx.operator.symbolic.compiler.diagnostics.state_sampling import (
    sample_source_states,
)
from neuralqx.operator.symbolic.compiler.diagnostics.state_sampling import (
    state_to_tuple,
)
from neuralqx.operator.symbolic.compiler.lowering.jax_lowerer import (
    infer_shift_mod_spec_from_hilbert,
)
from neuralqx.operator.symbolic.compiler.lowering.jax_lowerer import make_kbody_runner


def _example_payload(
    source_state: np.ndarray,
    target_state: np.ndarray,
    *,
    emission_index: int,
    illegal_positions: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Builds one structured example payload.

    Args:
        source_state: Source state vector.
        target_state: Connected/target state vector.
        emission_index: Emission index inside the term.
        illegal_positions: Optional illegal local-state site indices.

    Returns:
        Structured context payload.
    """
    payload: dict[str, Any] = {
        "source_state": tuple(np.asarray(source_state).tolist()),
        "target_state": tuple(np.asarray(target_state).tolist()),
        "emission_index": int(emission_index),
    }
    if illegal_positions is not None:
        payload["illegal_positions"] = illegal_positions
    return payload


def _record_example(
    bucket: dict[str, list[dict[str, Any]]],
    term_name: str,
    payload: dict[str, Any],
    *,
    max_examples_per_term: int = 3,
) -> None:
    """Records one example payload in a per-term diagnostics bucket.

    Args:
        bucket: Per-term mutable example bucket.
        term_name: Target term name.
        payload: Example payload to append.
        max_examples_per_term: Maximum examples retained per term.
    """
    examples = bucket.setdefault(term_name, [])
    if len(examples) < max_examples_per_term:
        examples.append(payload)


class GeneratedConnectivityValidityRule(AbstractDiagnosticRule):
    """Reports invalid generated connected states from sampled runner execution.

    This rule intentionally reuses the JAX lowerer term-runner builder so
    diagnostics and lowering semantics remain in lockstep as DSL behavior
    evolves.
    """

    @property
    def name(self) -> str:
        return "generated_connectivity_validity"

    def run(self, context: DiagnosticRuleContext) -> tuple[DSLDiagnostic, ...]:
        operator = context.operator
        ir = context.ir
        options = context.options
        hilbert = operator.hilbert

        samples = sample_source_states(
            hilbert,
            max_samples=options.lint_state_sample_size,
        )
        if samples is None or samples.size == 0:
            return (
                DSLDiagnostic.create(
                    code="NQXDSL-I303",
                    severity="info",
                    message=(
                        "Skipped sampled connectivity checks because no source states "
                        "could be obtained through direct sampling or bounded indexing."
                    ),
                    operator_name=ir.operator_name,
                    suggestion=(
                        "Provide a constraint-aware hilbert.random_state implementation "
                        "to enable connectivity diagnostics without Hilbert enumeration."
                    ),
                    context={"lint_state_sample_size": options.lint_state_sample_size},
                ),
            )

        support_lookup = build_hilbert_support_lookup(
            hilbert,
            max_states=options.lint_max_exact_hilbert_states,
        )
        local_states = getattr(hilbert, "local_states", None)
        try:
            shift_mod_spec = infer_shift_mod_spec_from_hilbert(hilbert)
        except Exception:
            shift_mod_spec = None
        if shift_mod_spec is None:
            shift_mod_state_min = None
            shift_mod_mod_span = None
        else:
            shift_mod_state_min, shift_mod_mod_span = shift_mod_spec

        outside_support_counts: dict[str, int] = {}
        outside_support_examples: dict[str, list[dict[str, Any]]] = {}
        constraint_counts: dict[str, int] = {}
        constraint_examples: dict[str, list[dict[str, Any]]] = {}
        illegal_counts: dict[str, int] = {}
        illegal_examples: dict[str, list[dict[str, Any]]] = {}
        evaluation_failures: dict[str, int] = {}

        sampled_branches = 0
        max_branch_samples = max(1, int(options.lint_branch_sample_cap))

        for term in ir.terms:
            emission_count = max(1, len(term.effective_emissions))
            try:
                runner = make_kbody_runner(
                    term,
                    ir.hilbert_size,
                    np.dtype(ir.dtype_str),
                    shift_mod_state_min=shift_mod_state_min,
                    shift_mod_mod_span=shift_mod_mod_span,
                )
            except Exception:
                name = str(term.name)
                evaluation_failures[name] = evaluation_failures.get(name, 0) + 1
                continue

            for source_state in samples:
                source_state = np.asarray(source_state)
                try:
                    x_primes, _mels, valids = runner(jnp.asarray(source_state))
                except Exception:
                    name = str(term.name)
                    evaluation_failures[name] = evaluation_failures.get(name, 0) + 1
                    continue

                x_primes_np = np.asarray(x_primes)
                valids_np = np.asarray(valids).reshape(-1).astype(bool)
                if x_primes_np.ndim == 1:
                    x_primes_np = x_primes_np[None, :]

                for branch_index, is_valid in enumerate(valids_np):
                    if sampled_branches >= max_branch_samples:
                        break
                    if not bool(is_valid):
                        continue
                    sampled_branches += 1
                    target_state = np.asarray(x_primes_np[branch_index])
                    emission_index = int(branch_index % emission_count)

                    illegal_positions = illegal_local_state_positions(
                        target_state, local_states, hilbert=hilbert
                    )
                    if illegal_positions:
                        name = str(term.name)
                        illegal_counts[name] = illegal_counts.get(name, 0) + 1
                        _record_example(
                            illegal_examples,
                            name,
                            _example_payload(
                                source_state,
                                target_state,
                                emission_index=emission_index,
                                illegal_positions=illegal_positions,
                            ),
                        )

                    constraint_ok = evaluate_constraint_accepts_state(
                        hilbert, target_state
                    )
                    if constraint_ok is False:
                        name = str(term.name)
                        constraint_counts[name] = constraint_counts.get(name, 0) + 1
                        _record_example(
                            constraint_examples,
                            name,
                            _example_payload(
                                source_state,
                                target_state,
                                emission_index=emission_index,
                            ),
                        )

                    if (
                        support_lookup is not None
                        and state_to_tuple(target_state) not in support_lookup
                    ):
                        name = str(term.name)
                        outside_support_counts[name] = (
                            outside_support_counts.get(name, 0) + 1
                        )
                        _record_example(
                            outside_support_examples,
                            name,
                            _example_payload(
                                source_state,
                                target_state,
                                emission_index=emission_index,
                            ),
                        )

                if sampled_branches >= max_branch_samples:
                    break
            if sampled_branches >= max_branch_samples:
                break

        diagnostics: list[DSLDiagnostic] = []

        if support_lookup is None:
            diagnostics.append(
                DSLDiagnostic.create(
                    code="NQXDSL-I301",
                    severity="info",
                    message=(
                        "Skipped exact Hilbert-support membership checks because "
                        "bounded support enumeration is unavailable or exceeds the "
                        "configured Hilbert-state limit."
                    ),
                    operator_name=ir.operator_name,
                    context={
                        "lint_max_exact_hilbert_states": options.lint_max_exact_hilbert_states
                    },
                )
            )

        for term_name in sorted(evaluation_failures):
            diagnostics.append(
                DSLDiagnostic.create(
                    code="NQXDSL-I302",
                    severity="info",
                    message=(
                        f"Skipped {evaluation_failures[term_name]} sampled branch evaluation(s) "
                        "because term execution failed in diagnostics mode."
                    ),
                    operator_name=ir.operator_name,
                    term_name=term_name,
                    suggestion=(
                        "Resolve unresolved symbols or unsupported runtime values to enable "
                        "full connectivity diagnostics for this term."
                    ),
                    context={
                        "evaluation_failure_count": evaluation_failures[term_name]
                    },
                )
            )

        for term_name in sorted(illegal_counts):
            diagnostics.append(
                DSLDiagnostic.create(
                    code="NQXDSL-W303",
                    severity="warning",
                    message=(
                        f"Generated connected states contain illegal local basis values "
                        f"for {illegal_counts[term_name]} sampled branch(es)."
                    ),
                    operator_name=ir.operator_name,
                    term_name=term_name,
                    suggestion="Adjust updates so each site remains in the local domain at that site.",
                    context={
                        "sampled_branch_count": illegal_counts[term_name],
                        "examples": tuple(illegal_examples.get(term_name, ())),
                    },
                )
            )

        for term_name in sorted(constraint_counts):
            diagnostics.append(
                DSLDiagnostic.create(
                    code="NQXDSL-W302",
                    severity="warning",
                    message=(
                        f"Generated connected states violate Hilbert constraints "
                        f"for {constraint_counts[term_name]} sampled branch(es)."
                    ),
                    operator_name=ir.operator_name,
                    term_name=term_name,
                    suggestion=(
                        "Restrict predicates/updates so generated states satisfy "
                        "hilbert.constraint(state)."
                    ),
                    context={
                        "sampled_branch_count": constraint_counts[term_name],
                        "examples": tuple(constraint_examples.get(term_name, ())),
                    },
                )
            )

        for term_name in sorted(outside_support_counts):
            diagnostics.append(
                DSLDiagnostic.create(
                    code="NQXDSL-W301",
                    severity="warning",
                    message=(
                        f"Generated connected states fall outside the Hilbert state space "
                        f"for {outside_support_counts[term_name]} sampled branch(es)."
                    ),
                    operator_name=ir.operator_name,
                    term_name=term_name,
                    suggestion="Ensure emitted states remain valid Hilbert configurations.",
                    context={
                        "sampled_branch_count": outside_support_counts[term_name],
                        "examples": tuple(outside_support_examples.get(term_name, ())),
                    },
                )
            )

        return tuple(diagnostics)


__all__ = ["GeneratedConnectivityValidityRule"]
