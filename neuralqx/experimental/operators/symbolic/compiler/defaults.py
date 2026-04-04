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


"""Default pipeline and registry factories for the symbolic compiler."""

from __future__ import annotations

from neuralqx.experimental.operators.symbolic.compiler.cache.store import (
    InMemorySymbolicArtifactStore,
)
from neuralqx.experimental.operators.symbolic.compiler.core.pipeline import (
    SymbolicPassPipeline,
)
from neuralqx.experimental.operators.symbolic.compiler.lowering.jax_lowerer import (
    JAXSymbolicLowerer,
)
from neuralqx.experimental.operators.symbolic.compiler.lowering.registry import (
    SymbolicLowererRegistry,
)
from neuralqx.experimental.operators.symbolic.compiler.passes.analysis import (
    SymbolicFanoutAnalysisPass,
)
from neuralqx.experimental.operators.symbolic.compiler.passes.fusion import (
    SymbolicFusionPass,
)
from neuralqx.experimental.operators.symbolic.compiler.passes.normalization import (
    SymbolicNormalizationPass,
)
from neuralqx.experimental.operators.symbolic.compiler.passes.validation import (
    SymbolicValidationPass,
)

# Module-level shared in-memory store (one per process)
_DEFAULT_STORE: InMemorySymbolicArtifactStore | None = None


def default_symbolic_pass_pipeline() -> SymbolicPassPipeline:
    """
    Builds the default two-stage symbolic compiler pass pipeline.

    **Pre-cache passes** (run on every :meth:`compile` call):
        1. :class:`~neuralqx.experimental.operators.symbolic.compiler.passes.validation.SymbolicValidationPass`
           - validates IR symbol scopes and update-op parameters.
        2. :class:`~neuralqx.experimental.operators.symbolic.compiler.passes.normalization.SymbolicNormalizationPass`
           - computes the IR fingerprint and resolves the target backend.

    **Post-cache passes** (run only on cache misses):
        1. :class:`~neuralqx.experimental.operators.symbolic.compiler.passes.analysis.SymbolicFanoutAnalysisPass`
           - derives per-term fanout bounds and the total padded output size.
        2. :class:`~neuralqx.experimental.operators.symbolic.compiler.passes.fusion.SymbolicFusionPass`
           - groups terms into fusion-compatible clusters for the lowerer.

    Returns:
        Configured :class:`~neuralqx.experimental.operators.symbolic.compiler.core.pipeline.SymbolicPassPipeline`.
    """
    return SymbolicPassPipeline(
        pre_cache_passes=[
            SymbolicValidationPass(),
            SymbolicNormalizationPass(),
        ],
        post_cache_passes=[
            SymbolicFanoutAnalysisPass(),
            SymbolicFusionPass(),
        ],
    )


def default_symbolic_lowerer_registry() -> SymbolicLowererRegistry:
    """
    Builds the default symbolic lowerer registry.

    Currently registers only the JAX backend lowerer
    (:class:`~neuralqx.experimental.operators.symbolic.compiler.lowering.jax_lowerer.JAXSymbolicLowerer`).

    Returns:
        Configured :class:`~neuralqx.experimental.operators.symbolic.compiler.lowering.registry.SymbolicLowererRegistry`.
    """
    registry = SymbolicLowererRegistry()
    registry.register(JAXSymbolicLowerer())
    return registry


def default_symbolic_artifact_store() -> InMemorySymbolicArtifactStore:
    """
    Returns the module-level shared in-memory artifact store.

    The store is lazily created and reused across compiler instances in the
    same process. Call :meth:`InMemorySymbolicArtifactStore.clear` to
    evict all compiled artifacts if needed.

    Returns:
        Shared :class:`~neuralqx.experimental.operators.symbolic.compiler.cache.store.InMemorySymbolicArtifactStore`.
    """
    global _DEFAULT_STORE  # noqa: PLW0603
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = InMemorySymbolicArtifactStore()
    return _DEFAULT_STORE


__all__ = [
    "default_symbolic_pass_pipeline",
    "default_symbolic_lowerer_registry",
    "default_symbolic_artifact_store",
]
