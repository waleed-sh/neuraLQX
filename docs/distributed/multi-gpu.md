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

## Fixed work and growing work

Keeping the global sample count fixed makes it possible to compare the cost of the same-sized update across device counts. This is often called strong scaling. Eventually there is too little local work to hide communication and launch overhead. The small demonstration reaches that regime quickly.

For a larger calculation you may instead keep a fixed amount of sampling work per device. In `build_calculation()`, replace the sampler's `n_chains=64` argument with `n_chains_per_device=16`, and replace the state's `n_samples=1024` argument with `n_samples=256 * jax.device_count()`. On four devices this gives 64 chains and 1,024 samples. On eight devices it gives 128 chains and 2,048 samples, with retained chain length 16 in both cases.

Do not provide both chain-count arguments. Increasing device count under this second policy changes the statistical workload, so a time comparison is no longer a comparison of identical global batches. More samples can improve precision, but the gain still depends on autocorrelation and mixing. The [sampler diagnostics](../concepts/samplers/diagnostics.md) explain how to assess those properties.

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

To use operator sharding in this job, change its `NEURALQX_SHARDING` assignment to `samples,operators`. Keep the same simulation file. To spread the calculation across hosts, continue with [multiple nodes and GPUs](multi-node.md).
