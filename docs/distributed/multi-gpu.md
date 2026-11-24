---
title: Multiple GPUs
---

# Multiple GPUs

## Installation and environment

Several NVIDIA GPUs on one Linux host use the same Python environment as [a single GPU](single-gpu.md#installation-and-environment). Install neuraLQX with the CUDA dependencies before launching the calculation:

```bash
python -m pip install --upgrade "neuralqx[cuda]"
python -m pip check
```

If the host requires a particular CUDA family, use the corresponding explicit JAX extra, such as `python -m pip install --upgrade neuralqx "jax[cuda12]"`, instead of choosing the generic CUDA extra. A local CUDA installation must also provide the compatible NCCL libraries used for GPU communication. Follow the [JAX installation requirements](https://docs.jax.dev/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-locally-harder) and the environment guidance from the previous page.

All participating GPUs must be visible to the same process. Check this within the allocation, not on a CPU-only login node:

```bash
JAX_PLATFORMS=cuda python - <<'PY'
import jax

print("Processes:", jax.process_count())
print("Local devices:", jax.local_devices())
print("Global devices:", jax.devices())
assert jax.process_count() == 1
assert jax.local_device_count() >= 2
assert all(device.platform == "gpu" for device in jax.devices())
PY
```

This guide uses **one Python process controlling all GPUs on the host**. It does not need `jax.distributed.initialize()` or `NEURALQX_DISTRIBUTED=1`. Those settings establish cooperation between processes and are introduced in the [multiple-node guide](multi-node.md).

## Change the launch, keep the simulation

Use the same [simulation.py](examples/simulation.py) as before. On an unscheduled workstation with four selected GPUs, launch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 JAX_PLATFORMS=cuda python simulation.py
```

In a scheduler allocation that already exposes the intended GPUs, use `JAX_PLATFORMS=cuda python simulation.py` and retain the scheduler's visibility setting. No change to the graph, state, objective, or driver is required. Sample sharding is already enabled by default.

The script should report one process and four local devices. Its sample array retains global shape `(64, 16, 4)`, but the chain dimension is partitioned. With four devices, each addressable shard contains 16 chains, or 256 retained configurations.

| Visible devices | Global chains | Global samples | Chains on each device | Samples on each device |
| --- | --- | --- | --- | --- |
| 1 | 64 | 1,024 | 64 | 1,024 |
| 4 | 64 | 1,024 | 16 | 256 |
| 8 | 64 | 1,024 | 8 | 128 |

Every sample still has all four edge labels. The standard model evaluation uses complete network parameters on each participating device. The expectation and gradient combine contributions from the global batch, so this remains one optimisation, rather than four independent models trained on unrelated data.

## Add tensor parallelism

To divide the MLP's weights and contractions between pairs of GPUs, keep the same script and change the launch policy:

```bash
NEURALQX_SHARDING=samples,tensor \
NEURALQX_MESH_TENSOR_AXIS_SIZE=2 \
JAX_PLATFORMS=cuda python simulation.py
```

On four visible GPUs, the mesh is `(samples=2, operators=1, tensor=2)`. Each sampling group handles 32 chains, and each tensor worker has a `(32,16,4)` copy of that group's sample shard. The first hidden weight of global shape `(4,32)` is column-partitioned into `(4,16)` blocks. The next weight of shape `(32,32)` is row-partitioned into `(16,32)` blocks. The small final head remains replicated.

For tensor-only execution on the same four GPUs:

```bash
NEURALQX_SHARDING=tensor \
NEURALQX_MESH_TENSOR_AXIS_SIZE=4 \
JAX_PLATFORMS=cuda python simulation.py
```

Now all four devices cooperate on each network evaluation. Each sees the complete sample batch, and the eligible weight dimensions divide four ways. No sample or operator group works on an independent portion of the batch. This can be appropriate when network memory dominates, but it gives up independent sampling work on those devices.

| Four-device layout | Independent chain groups | Tensor workers per group | Chains stored by each worker |
| --- | --- | --- | --- |
| `samples` | 4 | 1 | 16 |
| `samples,tensor` with tensor size 2 | 2 | 2 | 32 |
| `tensor` with tensor size 4 | 1 | 4 | 64 |

The global sample count stays 1,024 in every row. Repeated sample shards in a tensor group do not add observations. The [tensor parallelism guide](tensor-parallelism/index.md) explains the contractions, memory tradeoff, and automatic layout selection. The same startup policy works with [custom models made from the supplied layers](tensor-parallelism/custom-models.md).

## Fixed work and growing work

Keeping the global sample count fixed compares the cost of the same-sized update across layouts. This is often called strong scaling. Eventually each independent group has too little work to offset communication and launch overhead. The small demonstration can reach that point quickly.

To grow the statistical workload with the independent sampling groups, replace the sampler and the state's sample-count argument in `build_calculation()` as follows:

```python
import neuralqx as nqx
from simulation import build_calculation

original, _, _ = build_calculation()
sampler = nqx.sampler.MetropolisLocal(
    original.hilbert, n_chains_per_device=16,
)
requested_samples = 256 * sampler.sample_device_count
print("Independent sampling groups:", sampler.sample_device_count)
print("Requested global samples:", requested_samples)
```

Use `sampler` as the new state's sampler and `n_samples=requested_samples` in its constructor. This diagnostic constructs the sampler separately so the count can be inspected. Do not supply `n_chains` as well as `n_chains_per_device`.

Without tensor parallelism, four sample devices give 64 chains and 1,024 samples. With four GPUs arranged as two tensor pairs, this policy gives 32 chains and 512 samples. The retained chain length is 16 in either case. Multiplying by `jax.device_count()` in the latter case would also count the cooperating tensor workers, increasing work per independent group.

Growing the sample count changes the statistical calculation, so a timing comparison is no longer for identical global batches. The precision gained still depends on autocorrelation and mixing. Use [sampler diagnostics](../concepts/samplers/diagnostics.md) to assess that change.

## Add operator sharding

To distribute the example's four squared-charge terms as well as its chains, run:

```bash
NEURALQX_SHARDING=samples,operators \
JAX_PLATFORMS=cuda python simulation.py
```

During sampling, all four devices still receive different chains. During local-estimator evaluation, four computational terms on four devices produce an operator-only layout. Every device sees the batch needed for its assigned term, and the term contributions are combined. This can increase the batch data held by a device relative to sample-only evaluation. Operator sharding is therefore a policy to measure against your workload, not a general memory reduction switch.

The following complete diagnostic inspects the default operator plan for the same four one-connection terms. Run it in the same environment as the simulation:

```python
import jax
from neuralqx.jax.distributed import (
    DistributedPhase,
    DistributedWorkload,
    plan_distributed_execution,
)

plan = plan_distributed_execution(
    DistributedPhase.OPERATOR_LOCAL_VALUES,
    DistributedWorkload.operator_local_values(
        operator_costs=(1, 1, 1, 1), sample_count=1024, enabled=True
    ),
)
if plan is None:
    print("No multi-device operator plan on this device allocation")
else:
    print("Layout:", plan.layout.value)
    print("Mesh:", dict(plan.mesh.shape))
    print("Operator partitions:", plan.operator_partitions)
    print("Sample partitions:", plan.sample_partitions)
    print("Scheduled term slots:", jax.device_get(plan.term_indices))
```

[`plan_distributed_execution`](../api/public/neuralqx/jax/distributed/index.md#neuralqx.jax.distributed.plan_distributed_execution) describes a plan for the supplied workload. It does not itself evaluate the operator. With four devices the plan has four operator partitions and one sample partition. With eight devices it has four operator partitions and two sample partitions. The [sharding guide](sharding.md#combined-operator-and-sample-layouts) explains the factorisation and provides a comparison of actual local values with the policy enabled and disabled.

## Combine all three modes

On eight GPUs, the following explicitly allocates two devices to each tensor group, two operator coordinates, and two sample coordinates:

```bash
NEURALQX_SHARDING=samples,operators,tensor \
NEURALQX_MESH_SAMPLE_AXIS_SIZE=2 \
NEURALQX_MESH_OPERATOR_AXIS_SIZE=2 \
NEURALQX_MESH_TENSOR_AXIS_SIZE=2 \
JAX_PLATFORMS=cuda python simulation.py
```

Sampling uses four independent chain groups. Operator evaluation uses two term groups and two sample groups, with the same tensor pair cooperating inside each evaluation. The [phase diagram](sharding.md#combined-operator-and-sample-layouts) gives the sample and term counts. Unlike the non-tensor policy above, this path retains the specified mesh instead of selecting a new operator factorization from the term count.

On four GPUs you can choose `operators,tensor` with tensor size two. This gives `(1,2,2)`, with two term groups and replicated samples. Alternatively, `all` with tensor size two infers `(2,1,2)`, so there is no nontrivial operator partition. Inspect the plan instead of assuming every enabled mode receives multiple groups.

## A job with four GPUs

Download [multi-gpu.sh](examples/multi-gpu.sh) into the same directory as `simulation.py` and the prepared `.venv`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=nqx-gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=nqx-gpus-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda
export NEURALQX_DISTRIBUTED=0
export NEURALQX_SHARDING=samples
srun --ntasks=1 python -u simulation.py
```

Submit it with `sbatch multi-gpu.sh`, adding the site-specific account and partition. The key change from [the single GPU job](single-gpu.md#a-single-gpu-batch-job) is the request for four GPUs. `--ntasks=1` remains intentional. Launching four independent tasks without distributed initialization would create four independent simulations, and may also make them compete for the same GPUs if binding is incorrect. Slurm's [GPU binding options](https://slurm.schedmd.com/srun.html) determine which devices each task can see.

For operator sharding, set `NEURALQX_SHARDING=samples,operators`. For two tensor pairs, use the complete [multi-gpu-tensor.sh](examples/multi-gpu-tensor.sh) variant, which sets `NEURALQX_SHARDING=samples,tensor` and `NEURALQX_MESH_TENSOR_AXIS_SIZE=2`. Both jobs use the same simulation file. To spread the calculation across hosts, continue with [multiple nodes and GPUs](multi-node.md).
