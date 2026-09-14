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


"""Exchange Metropolis transition rule."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

import jax
from neuralqx.graph import AbstractGraph
from neuralqx.utils.struct import field

from ..base import AbstractTransitionRule
from .graph import clusters_from_graph
from .incidence import incident_cluster_table
from .kernel import exchange_batch
from neuralqx.utils.typing import ExchangeEntity


class ExchangeRule(AbstractTransitionRule):
    """Swap the values of one allowed pair of flat sites."""

    clusters: Any = field(pytree=False, default=None)
    graph: AbstractGraph | None = field(pytree=False, default=None)
    d_max: int = field(static=True, default=1)
    entity: ExchangeEntity = field(static=True, default="vertex")
    probabilities: Any | None = field(pytree=False, default=None)
    site_i: Any = field(pytree=False, init=False, default=None, repr=False)
    site_j: Any = field(pytree=False, init=False, default=None, repr=False)
    incident_clusters: Any = field(pytree=False, init=False, default=None, repr=False)
    incident_valid: Any = field(pytree=False, init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        clusters = _resolve_clusters(
            clusters=self.clusters,
            graph=self.graph,
            d_max=self.d_max,
            entity=self.entity,
        )
        cluster_array = _validate_clusters(clusters)
        probabilities = _validate_probabilities(
            self.probabilities, cluster_array.shape[0]
        )
        incident_clusters, incident_valid = incident_cluster_table(clusters)

        object.__setattr__(self, "clusters", cluster_array)
        object.__setattr__(self, "site_i", cluster_array[:, 0])
        object.__setattr__(self, "site_j", cluster_array[:, 1])
        object.__setattr__(self, "incident_clusters", incident_clusters)
        object.__setattr__(self, "incident_valid", incident_valid)
        object.__setattr__(self, "probabilities", probabilities)

    def transition(
        self,
        sampler,
        apply_fn: Callable[[Any, jax.Array], jax.Array],
        parameters: Any,
        sampler_state,
        key: jax.Array,
        sigma: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        del sampler, apply_fn, parameters, sampler_state
        return exchange_batch(
            site_i=self.site_i,
            site_j=self.site_j,
            incident_clusters=self.incident_clusters,
            incident_valid=self.incident_valid,
            probabilities=self.probabilities,
            key=key,
            sigma=sigma,
        )

    def __hash__(self) -> int:
        return hash((type(self), self.d_max, self.entity, self.graph))


def _resolve_clusters(
    *,
    clusters: Any,
    graph: AbstractGraph | None,
    d_max: int,
    entity: ExchangeEntity,
) -> tuple[tuple[int, int], ...]:
    if clusters is None and graph is None:
        raise ValueError("ExchangeRule needs explicit clusters or a graph.")
    if clusters is not None and graph is not None:
        raise ValueError("Specify either clusters or graph, not both.")
    if d_max <= 0:
        raise ValueError("d_max must be positive.")
    if clusters is None:
        return clusters_from_graph(graph, d_max, entity)
    return tuple((int(i), int(j)) for i, j in clusters)


def _validate_clusters(clusters: tuple[tuple[int, int], ...]) -> np.ndarray:
    if not clusters:
        raise ValueError("ExchangeRule requires at least one exchange cluster.")
    cluster_array = np.asarray(clusters, dtype=np.int32)
    if cluster_array.ndim != 2 or cluster_array.shape[1] != 2:
        raise ValueError("clusters must have shape (n_clusters, 2).")
    if bool(np.any(cluster_array[:, 0] == cluster_array[:, 1])):
        raise ValueError("Exchange clusters must contain two distinct sites.")
    return cluster_array


def _validate_probabilities(
    probabilities: Any | None, n_clusters: int
) -> np.ndarray | None:
    if probabilities is None:
        return None
    out = np.asarray(probabilities)
    if out.shape != (int(n_clusters),):
        raise ValueError("probabilities must have one entry per exchange cluster.")
    if bool(np.any(out <= 0)):
        raise ValueError("probabilities must be positive.")
    return out


__all__ = ["ExchangeRule"]
