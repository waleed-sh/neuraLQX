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


"""Default distributed planning rules for neuraLQX phases."""

from __future__ import annotations

import jax
from neuralqx.config import config
from neuralqx.jax.mesh import create_device_mesh
from neuralqx.jax.mesh import logical_axis_name
from neuralqx.utils.typing import DistributedAxis
from neuralqx.utils.typing import DistributedLayoutKind
from neuralqx.utils.typing import DistributedPhase
from neuralqx.utils.typing import DistributedPlan
from neuralqx.utils.typing import DistributedWorkload
from neuralqx.utils.typing import DistributionRequest

from .coordinator import DistributedCoordinator
from .coordinator import distributed_planner
from .scheduling import schedule_balanced_slots


@distributed_planner(DistributedPhase.SAMPLING)
def _sampling_plan(
    _coordinator: DistributedCoordinator,
    workload: DistributedWorkload,
) -> DistributedPlan | None:
    """Plan sample generation over all available devices."""
    if workload.request is DistributionRequest.DISABLED:
        return None
    if (
        workload.request is DistributionRequest.CONFIG
        and not _sample_sharding_enabled()
    ):
        return None
    return _axis_plan(
        phase=DistributedPhase.SAMPLING,
        layout=DistributedLayoutKind.SAMPLE,
        axes=(DistributedAxis.SAMPLES,),
        axis_sizes={DistributedAxis.SAMPLES: int(jax.device_count())},
    )


@distributed_planner(DistributedPhase.GRADIENT_FORCE)
def _gradient_force_plan(
    _coordinator: DistributedCoordinator,
    workload: DistributedWorkload,
) -> DistributedPlan | None:
    """Plan covariance-force and score-function pullbacks over samples."""
    if workload.request is DistributionRequest.DISABLED:
        return None
    if (
        workload.request is DistributionRequest.CONFIG
        and not _sample_sharding_enabled()
    ):
        return None
    return _axis_plan(
        phase=DistributedPhase.GRADIENT_FORCE,
        layout=DistributedLayoutKind.SAMPLE,
        axes=(DistributedAxis.SAMPLES,),
        axis_sizes={DistributedAxis.SAMPLES: int(jax.device_count())},
    )


@distributed_planner(DistributedPhase.NETWORK_APPLY)
def _network_apply_plan(
    _coordinator: DistributedCoordinator,
    workload: DistributedWorkload,
) -> DistributedPlan | None:
    """Plan model application; parameter leaves are currently replicated."""
    if workload.request is DistributionRequest.DISABLED:
        return None
    if workload.request is DistributionRequest.CONFIG and not bool(
        config.get("parameter_sharding_enabled")
    ):
        return None
    return _axis_plan(
        phase=DistributedPhase.NETWORK_APPLY,
        layout=DistributedLayoutKind.PARAMETER_REPLICATED,
        axes=_active_parameter_axes(),
        axis_sizes={},
    )


@distributed_planner(DistributedPhase.OPERATOR_LOCAL_VALUES)
def _operator_local_values_plan(
    _coordinator: DistributedCoordinator,
    workload: DistributedWorkload,
) -> DistributedPlan | None:
    """Plan local-value operator work over operator and sample axes."""
    if workload.request is DistributionRequest.DISABLED:
        return None
    if workload.request is DistributionRequest.CONFIG and not bool(
        config.get("operator_sharding_enabled")
    ):
        return None
    if not workload.operator_costs:
        raise ValueError("Operator local-value planning requires operator costs.")

    device_count = int(jax.device_count())
    if device_count <= 1:
        return None

    operator_partitions, sample_partitions = _operator_axis_sizes(
        n_terms=len(workload.operator_costs),
        device_count=device_count,
        sample_count=workload.sample_count,
    )
    if sample_partitions > 1:
        axes = (DistributedAxis.OPERATORS, DistributedAxis.SAMPLES)
        layout = DistributedLayoutKind.OPERATOR_SAMPLE
        axis_sizes = {
            DistributedAxis.OPERATORS: operator_partitions,
            DistributedAxis.SAMPLES: sample_partitions,
        }
    else:
        axes = (DistributedAxis.OPERATORS,)
        layout = DistributedLayoutKind.OPERATOR
        axis_sizes = {DistributedAxis.OPERATORS: operator_partitions}

    plan = _axis_plan(
        phase=DistributedPhase.OPERATOR_LOCAL_VALUES,
        layout=layout,
        axes=axes,
        axis_sizes=axis_sizes,
    )
    if plan is None:
        return None
    return DistributedPlan(
        phase=plan.phase,
        layout=plan.layout,
        mesh=plan.mesh,
        axis_names=plan.axis_names,
        axis_partitions=plan.axis_partitions,
        term_indices=schedule_balanced_slots(
            workload.operator_costs,
            operator_partitions,
        ),
    )


def _axis_plan(
    *,
    phase: DistributedPhase,
    layout: DistributedLayoutKind,
    axes: tuple[DistributedAxis, ...],
    axis_sizes: dict[DistributedAxis, int],
) -> DistributedPlan | None:
    if not axes:
        return None
    logical_axes = tuple(axis.value for axis in axes)
    mesh = create_device_mesh(
        active_axes=logical_axes,
        axis_sizes={axis.value: size for axis, size in axis_sizes.items()},
    )
    if mesh is None:
        return None
    names = tuple((axis, logical_axis_name(axis.value)) for axis in axes)
    partitions = tuple(
        (axis, int(mesh.shape[logical_axis_name(axis.value)])) for axis in axes
    )
    return DistributedPlan(
        phase=phase,
        layout=layout,
        mesh=mesh,
        axis_names=names,
        axis_partitions=partitions,
    )


def _operator_axis_sizes(
    *,
    n_terms: int,
    device_count: int,
    sample_count: int | None,
) -> tuple[int, int]:
    if n_terms <= 0:
        raise ValueError("n_terms must be positive.")
    if n_terms >= device_count:
        return device_count, 1
    if sample_count is None or int(sample_count) <= 0 or not _sample_sharding_enabled():
        return device_count, 1

    candidates: list[tuple[int, int]] = []
    for operator_partitions in _divisors(device_count):
        if operator_partitions > n_terms:
            continue
        sample_partitions = device_count // operator_partitions
        if sample_partitions <= 1:
            continue
        if int(sample_count) % sample_partitions != 0:
            continue
        candidates.append((operator_partitions, sample_partitions))
    if not candidates:
        return device_count, 1
    return max(candidates, key=lambda item: (item[0], item[1]))


def _active_parameter_axes() -> tuple[DistributedAxis, ...]:
    axes = []
    if _sample_sharding_enabled():
        axes.append(DistributedAxis.SAMPLES)
    axes.append(DistributedAxis.PARAMETERS)
    if bool(config.get("operator_sharding_enabled")):
        axes.append(DistributedAxis.OPERATORS)
    return tuple(axes)


def _divisors(value: int) -> tuple[int, ...]:
    return tuple(
        candidate for candidate in range(1, int(value) + 1) if value % candidate == 0
    )


def _sample_sharding_enabled() -> bool:
    return bool(config.get("sample_sharding_enabled"))


__all__ = []
