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


"""Default pipeline and registry factories for the symbolic compiler."""

from __future__ import annotations

from neuralqx.operator.symbolic.compiler.cache.store import (
    InMemorySymbolicArtifactStore,
)
from neuralqx.operator.symbolic.compiler.core.pipeline import SymbolicPassPipeline
from neuralqx.operator.symbolic.compiler.lowering.jax_lowerer import JAXSymbolicLowerer
from neuralqx.operator.symbolic.compiler.lowering.operator_registry import (
    SymbolicOperatorLoweringRegistry,
)
from neuralqx.operator.symbolic.compiler.lowering.operator_registry import (
    build_default_symbolic_operator_lowering_registry,
)
from neuralqx.operator.symbolic.compiler.lowering.registry import (
    SymbolicLowererRegistry,
)
from neuralqx.operator.symbolic.compiler.passes.analysis import (
    SymbolicMaxConnSizeAnalysisPass,
)
from neuralqx.operator.symbolic.compiler.passes.connectivity_cost import (
    SymbolicConnectivityCostPass,
)
from neuralqx.operator.symbolic.compiler.passes.deduplication import (
    SymbolicDeduplicationPass,
)
from neuralqx.operator.symbolic.compiler.passes.diagnostics import (
    SymbolicDiagnosticsPass,
)
from neuralqx.operator.symbolic.compiler.passes.fusion import SymbolicFusionPass
from neuralqx.operator.symbolic.compiler.passes.kernel_planning import (
    SymbolicKernelPlanningPass,
)
from neuralqx.operator.symbolic.compiler.passes.normalization import (
    SymbolicNormalizationPass,
)
from neuralqx.operator.symbolic.compiler.passes.validation import SymbolicValidationPass

# Module-level shared in-memory store (one per process)
_DEFAULT_STORE: InMemorySymbolicArtifactStore | None = None
_DEFAULT_OPERATOR_LOWERING_REGISTRY = build_default_symbolic_operator_lowering_registry(
    default_from_config=True
)

LOWERING_REGISTRY = _DEFAULT_OPERATOR_LOWERING_REGISTRY
"""Shared lowering registry. New registrations become the configured default."""


def default_symbolic_pass_pipeline() -> SymbolicPassPipeline:
    """
    Builds the default two-stage symbolic compiler pass pipeline.

    **Pre-cache passes** (run on every :meth:`compile` call):
        1. :class:`~neuralqx.operator.symbolic.compiler.passes.validation.SymbolicValidationPass`
           - validates IR symbol scopes and update-op parameters.
        2. :class:`~neuralqx.operator.symbolic.compiler.passes.diagnostics.SymbolicDiagnosticsPass`
           - runs DSL lint rules and reports diagnostics.
        3. :class:`~neuralqx.operator.symbolic.compiler.passes.normalization.SymbolicNormalizationPass`
           - computes the IR fingerprint and resolves the target backend.

    **Post-cache passes** (run only on cache misses):
        1. :class:`~neuralqx.operator.symbolic.compiler.passes.analysis.SymbolicMaxConnSizeAnalysisPass`
           - derives raw per-term and total max-connection-size bounds.
        2. :class:`~neuralqx.operator.symbolic.compiler.passes.deduplication.SymbolicDeduplicationPass`
           - plans target aggregation, residual collisions, and reduced output capacity.
        3. :class:`~neuralqx.operator.symbolic.compiler.passes.connectivity_cost.SymbolicConnectivityCostPass`
           - reports estimated deduplication assembly overhead.
        4. :class:`~neuralqx.operator.symbolic.compiler.passes.fusion.SymbolicFusionPass`
           - groups terms into fusion-compatible clusters for the lowerer.
        5. :class:`~neuralqx.operator.symbolic.compiler.passes.kernel_planning.SymbolicKernelPlanningPass`
           - plans scalar DAG batches and deferred target coordinate generation.

    Returns:
        Configured :class:`~neuralqx.operator.symbolic.compiler.core.pipeline.SymbolicPassPipeline`.
    """
    return SymbolicPassPipeline(
        pre_cache_passes=[
            SymbolicValidationPass(),
            SymbolicDiagnosticsPass(),
            SymbolicNormalizationPass(),
        ],
        post_cache_passes=[
            SymbolicMaxConnSizeAnalysisPass(),
            SymbolicDeduplicationPass(),
            SymbolicConnectivityCostPass(),
            SymbolicFusionPass(),
            SymbolicKernelPlanningPass(),
        ],
    )


def default_symbolic_lowerer_registry(
    *,
    operator_lowering_registry: SymbolicOperatorLoweringRegistry | None = None,
) -> SymbolicLowererRegistry:
    """
    Builds the default symbolic lowerer registry.

    Currently registers only the JAX backend lowerer
    (:class:`~neuralqx.operator.symbolic.compiler.lowering.jax_lowerer.JAXSymbolicLowerer`).
    The lowerer is configured with *operator_lowering_registry*.

    Returns:
        Configured :class:`~neuralqx.operator.symbolic.compiler.lowering.registry.SymbolicLowererRegistry`.
    """
    resolved_targets = (
        operator_lowering_registry
        if operator_lowering_registry is not None
        else default_symbolic_operator_lowering_registry()
    )
    registry = SymbolicLowererRegistry()
    registry.register(
        JAXSymbolicLowerer(
            operator_lowering_registry=resolved_targets,
        )
    )
    return registry


def default_symbolic_operator_lowering_registry() -> SymbolicOperatorLoweringRegistry:
    """
    Returns the module-level shared operator-lowering target registry.

    This is the same object as :data:`neuralqx.operator.symbolic.LOWERING_REGISTRY`. It provides
    the native neuraLQX target and follows ``neuralqx.operator.symbolic.config.LOWERING``.
    Registering a custom target selects it unless ``set_default=False``.
    """
    return _DEFAULT_OPERATOR_LOWERING_REGISTRY


def default_symbolic_artifact_store() -> InMemorySymbolicArtifactStore:
    """
    Returns the module-level shared in-memory artifact store.

    The store is lazily created and reused across compiler instances in the
    same process. Call :meth:`InMemorySymbolicArtifactStore.clear` to
    evict all compiled artifacts if needed.

    Returns:
        Shared :class:`~neuralqx.operator.symbolic.compiler.cache.store.InMemorySymbolicArtifactStore`.
    """
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = InMemorySymbolicArtifactStore()
    return _DEFAULT_STORE


def reset_default_symbolic_singletons() -> None:
    """
    Resets the shared artifact store, registered targets, and lowering override.

    This helper primarily exists for tests that need deterministic singleton
    lifecycle behavior across cases. The public registry keeps its identity.
    """
    global _DEFAULT_STORE
    _DEFAULT_STORE = None
    replacement = build_default_symbolic_operator_lowering_registry(
        default_from_config=True
    )
    registry = _DEFAULT_OPERATOR_LOWERING_REGISTRY
    with registry._lock:
        registry._targets = replacement._targets
        registry._lazy_targets = replacement._lazy_targets
        registry._default_target = replacement._default_target
    from neuralqx.operator.symbolic.configs import cfg

    cfg.clear_override("LOWERING")


__all__ = [
    "LOWERING_REGISTRY",
    "default_symbolic_artifact_store",
    "default_symbolic_lowerer_registry",
    "default_symbolic_operator_lowering_registry",
    "default_symbolic_pass_pipeline",
    "reset_default_symbolic_singletons",
]
