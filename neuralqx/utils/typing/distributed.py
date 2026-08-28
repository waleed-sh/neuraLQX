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


"""Distributed-execution public typing objects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from enum import unique
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeAlias

import jax
from jax.sharding import Mesh

if TYPE_CHECKING:
    from neuralqx.jax.distributed.coordinator import DistributedCoordinator


@unique
class DistributedAxis(Enum):
    """Logical mesh axes understood by distributed execution policies."""

    SAMPLES = "samples"
    OPERATORS = "operators"
    PARAMETERS = "parameters"


@unique
class DistributedPhase(Enum):
    """High-level execution phases with independent layout requirements."""

    SAMPLING = "sampling"
    OPERATOR_LOCAL_VALUES = "operator_local_values"
    GRADIENT_FORCE = "gradient_force"
    NETWORK_APPLY = "network_apply"


@unique
class DistributedLayoutKind(Enum):
    """Canonical distributed layouts selected by execution planners."""

    DISABLED = "disabled"
    REPLICATED = "replicated"
    SAMPLE = "sample"
    OPERATOR = "operator"
    OPERATOR_SAMPLE = "operator_sample"
    PARAMETER_REPLICATED = "parameter_replicated"


@unique
class DistributionRequest(Enum):
    """Explicit request state for a distributed execution phase."""

    CONFIG = "config"
    ENABLED = "enabled"
    DISABLED = "disabled"

    @classmethod
    def from_flag(cls, value: bool | None) -> DistributionRequest:
        """Convert historical ``None`` and ``bool`` flags to a request state.

        Args:
            value: Legacy request flag. ``None`` means the phase should follow
                global configuration, ``True`` means explicitly enabled, and
                ``False`` means explicitly disabled.

        Returns:
            The corresponding explicit distribution request.
        """
        if value is None:
            return cls.CONFIG
        return cls.ENABLED if bool(value) else cls.DISABLED


@dataclass(frozen=True, slots=True)
class DistributedWorkload:
    """Static workload facts used to choose a distributed layout.

    The coordinator receives only cheap shape-level information. It must not
    inspect large arrays or call model or operator kernels while planning. This
    keeps planning outside JIT traces and makes layouts stable across
    compilation boundaries.

    Attributes:
        request: User or configuration request state for the phase.
        sample_count: Number of samples available to the phase, if known.
        operator_costs: Per-operator cost estimates used for scheduling.
    """

    request: DistributionRequest = DistributionRequest.CONFIG
    """User or configuration request state for the phase."""

    sample_count: int | None = None
    """Number of samples available to the phase, if known."""

    operator_costs: tuple[int, ...] = ()
    """Per-operator cost estimates used for scheduling."""

    @classmethod
    def operator_local_values(
        cls,
        *,
        operator_costs: tuple[int, ...],
        sample_count: int | None,
        enabled: bool | None,
    ) -> DistributedWorkload:
        """Build a workload descriptor for local-value estimation.

        Args:
            operator_costs: Per-operator cost estimates. The default planner
                uses each operator's maximum connected-state count.
            sample_count: Number of samples used by the local-estimator call,
                if known at planning time.
            enabled: Legacy operator-sharding request flag.

        Returns:
            A normalized workload descriptor for the operator-local-values
            distributed phase.
        """
        return cls(
            request=DistributionRequest.from_flag(enabled),
            sample_count=sample_count,
            operator_costs=tuple(max(1, int(cost)) for cost in operator_costs),
        )


@dataclass(frozen=True, slots=True)
class DistributedPlan:
    """Concrete distributed layout selected for one execution phase.

    Attributes:
        phase: Execution phase for which the layout was selected.
        layout: Canonical layout kind selected by the planner.
        mesh: JAX mesh used by the layout, or ``None`` for disabled plans.
        axis_names: Mapping from logical axes to configured JAX mesh names.
        axis_partitions: Number of device partitions per logical axis.
        term_indices: Optional scheduled operator-term indices for the plan.
    """

    phase: DistributedPhase
    """Execution phase for which the layout was selected."""

    layout: DistributedLayoutKind
    """Canonical layout kind selected by the planner."""

    mesh: Mesh | None = None
    """JAX mesh used by the layout, or ``None`` for disabled plans."""

    axis_names: tuple[tuple[DistributedAxis, str], ...] = ()
    """Mapping from logical axes to configured JAX mesh names."""

    axis_partitions: tuple[tuple[DistributedAxis, int], ...] = ()
    """Number of device partitions per logical axis."""

    term_indices: jax.Array | None = None
    """Optional scheduled operator-term indices for the plan."""

    @classmethod
    def disabled(cls, phase: DistributedPhase) -> DistributedPlan:
        """Return a no-distribution plan for ``phase``.

        Args:
            phase: Execution phase that should run without distributed layout.

        Returns:
            A disabled plan for the requested phase.
        """
        return cls(phase=phase, layout=DistributedLayoutKind.DISABLED)

    @property
    def mode(self) -> str:
        """Historical operator-dispatch mode name for this layout."""
        if self.layout is DistributedLayoutKind.OPERATOR:
            return "operator_only"
        if self.layout is DistributedLayoutKind.OPERATOR_SAMPLE:
            return "operator_sample"
        return self.layout.value

    @property
    def uses_sample_axis(self) -> bool:
        """Whether the plan has a nontrivial sample axis."""
        return self.partitions(DistributedAxis.SAMPLES) > 1

    @property
    def operator_axis(self) -> str:
        """Configured JAX mesh axis name for operator partitions."""
        return self.axis_name(DistributedAxis.OPERATORS)

    @property
    def sample_axis(self) -> str | None:
        """Configured JAX mesh axis name for sample partitions, if present."""
        return self.axis_name_or_none(DistributedAxis.SAMPLES)

    @property
    def operator_partitions(self) -> int:
        """Number of operator-axis partitions."""
        return self.partitions(DistributedAxis.OPERATORS)

    @property
    def sample_partitions(self) -> int:
        """Number of sample-axis partitions."""
        return self.partitions(DistributedAxis.SAMPLES)

    def axis_name(self, axis: DistributedAxis) -> str:
        """Return the configured JAX mesh axis name for ``axis``.

        Args:
            axis: Logical distributed axis to look up.

        Returns:
            The configured JAX mesh axis name.

        Raises:
            KeyError: If ``axis`` is not present in this plan.
        """
        value = self.axis_name_or_none(axis)
        if value is None:
            raise KeyError(f"Plan {self.phase.value!r} has no {axis.value!r} axis.")
        return value

    def axis_name_or_none(self, axis: DistributedAxis) -> str | None:
        """Return the configured JAX mesh axis name for ``axis``, if present.

        Args:
            axis: Logical distributed axis to look up.

        Returns:
            The configured JAX mesh axis name, or ``None`` when the plan does
            not use that axis.
        """
        for key, value in self.axis_names:
            if key is axis:
                return value
        return None

    def partitions(self, axis: DistributedAxis) -> int:
        """Return the partition count for ``axis``.

        Missing axes are interpreted as replicated axes and therefore have one
        partition.

        Args:
            axis: Logical distributed axis to look up.

        Returns:
            Number of device partitions for the axis.
        """
        for key, value in self.axis_partitions:
            if key is axis:
                return int(value)
        return 1

    def require_mesh(self) -> Mesh:
        """Return the plan mesh.

        Returns:
            The concrete JAX mesh attached to this plan.

        Raises:
            ValueError: If this plan has no executable mesh.
        """
        if self.mesh is None:
            raise ValueError(f"Distributed plan {self.phase.value!r} has no mesh.")
        return self.mesh

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-friendly diagnostic summary.

        Returns:
            Dictionary containing phase, layout, configured axis names, and
            partition counts.
        """
        return {
            "phase": self.phase.value,
            "layout": self.layout.value,
            "axis_names": {axis.value: name for axis, name in self.axis_names},
            "axis_partitions": {
                axis.value: int(size) for axis, size in self.axis_partitions
            },
        }


Planner: TypeAlias = Callable[
    ["DistributedCoordinator", DistributedWorkload],
    DistributedPlan | None,
]
"""Callable signature for a distributed phase planner."""

WorkloadBuilder: TypeAlias = Callable[..., DistributedWorkload]
"""Callable signature for constructing workload descriptors at call boundaries."""


__all__ = [
    "DistributedAxis",
    "DistributedLayoutKind",
    "DistributedPhase",
    "DistributedPlan",
    "DistributedWorkload",
    "DistributionRequest",
    "Planner",
    "WorkloadBuilder",
]
