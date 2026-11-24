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
from neuralqx.jax.tensor_parallel import network_sharding_enabled
from neuralqx.utils.errors import NeuralqxValueError
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
    if network_sharding_enabled():
        return _tensor_batch_plan(DistributedPhase.SAMPLING, workload)
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
    if network_sharding_enabled():
        return _tensor_batch_plan(DistributedPhase.GRADIENT_FORCE, workload)
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
    """Plan tensor-parallel model execution when enabled at startup.

    Sample batching has its own planner. Without tensor parallelism no separate
    model-layout plan is needed and the existing array placements are retained.
    """
    if workload.request is DistributionRequest.DISABLED:
        return None
    if network_sharding_enabled():
        return _tensor_batch_plan(DistributedPhase.NETWORK_APPLY, workload)
    return None


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
        raise NeuralqxValueError(
            "Operator local-value planning requires operator costs."
        )

    if network_sharding_enabled():
        return _tensor_operator_plan(workload)

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
        raise NeuralqxValueError("n_terms must be positive.")
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


def _divisors(value: int) -> tuple[int, ...]:
    return tuple(
        candidate for candidate in range(1, int(value) + 1) if value % candidate == 0
    )


def _sample_sharding_enabled() -> bool:
    return bool(config.get("sample_sharding_enabled"))


def _tensor_operator_plan(workload):
    from neuralqx.jax.tensor_parallel import tensor_axis_name
    from neuralqx.jax.tensor_parallel import tensor_mesh

    mesh = tensor_mesh()
    sample = logical_axis_name("samples")
    operator = logical_axis_name("operators")
    tensor = tensor_axis_name()
    op_count = int(mesh.shape[operator])
    if op_count <= 1:
        return None
    sample_count = int(mesh.shape[sample])
    use_samples = (
        sample_count > 1
        and workload.sample_count is not None
        and workload.sample_count % sample_count == 0
    )
    return DistributedPlan(
        phase=DistributedPhase.OPERATOR_LOCAL_VALUES,
        layout=(
            DistributedLayoutKind.OPERATOR_SAMPLE
            if use_samples
            else DistributedLayoutKind.OPERATOR
        ),
        mesh=mesh,
        axis_names=(
            (DistributedAxis.SAMPLES, sample),
            (DistributedAxis.OPERATORS, operator),
            (DistributedAxis.TENSOR, tensor),
        ),
        axis_partitions=(
            (DistributedAxis.SAMPLES, sample_count if use_samples else 1),
            (DistributedAxis.OPERATORS, op_count),
            (DistributedAxis.TENSOR, int(mesh.shape[tensor])),
        ),
        term_indices=schedule_balanced_slots(workload.operator_costs, op_count),
        batch_axes=(sample,) if use_samples else (),
    )


def _tensor_batch_plan(phase, workload):
    from math import prod

    from neuralqx.jax.tensor_parallel import tensor_mesh
    from neuralqx.jax.tensor_parallel.mesh import sample_spec

    mesh = tensor_mesh()
    axes = (DistributedAxis.SAMPLES, DistributedAxis.OPERATORS, DistributedAxis.TENSOR)
    spec = sample_spec()
    batch_axes = spec[0] if spec else ()
    count = prod(int(mesh.shape[name]) for name in batch_axes)
    if workload.sample_count is not None and workload.sample_count % count:
        batch_axes = ()
    return DistributedPlan(
        phase=phase,
        layout=(
            DistributedLayoutKind.SAMPLE_TENSOR
            if batch_axes
            else DistributedLayoutKind.TENSOR
        ),
        mesh=mesh,
        axis_names=tuple((axis, logical_axis_name(axis.value)) for axis in axes),
        axis_partitions=tuple(
            (axis, int(mesh.shape[logical_axis_name(axis.value)])) for axis in axes
        ),
        batch_axes=batch_axes,
    )


__all__ = []
