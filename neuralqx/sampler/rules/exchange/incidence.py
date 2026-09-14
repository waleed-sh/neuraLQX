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


"""Incident-cluster metadata for exchange transition rules."""

from __future__ import annotations

import numpy as np


def incident_cluster_table(
    clusters: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded incident-cluster indices and validity mask by site."""
    n_sites = max(max(i, j) for i, j in clusters) + 1
    per_site: list[list[int]] = [[] for _ in range(n_sites)]
    for cluster, (i, j) in enumerate(clusters):
        per_site[i].append(cluster)
        per_site[j].append(cluster)

    max_incident = max(len(row) for row in per_site)
    incident = np.zeros((n_sites, max_incident), dtype=np.int32)
    valid = np.zeros((n_sites, max_incident), dtype=np.bool_)
    for site, row in enumerate(per_site):
        if row:
            incident[site, : len(row)] = row
            valid[site, : len(row)] = True

    return incident, valid


__all__ = ["incident_cluster_table"]
