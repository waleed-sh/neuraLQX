---
title: Single GPU
---

# Single GPU

## Installation and environment

This page uses an NVIDIA GPU on Linux. Start with a Python environment as in the [CPU guide](cpu.md#installation-and-environment), then install neuraLQX and a CUDA-enabled JAX build. The package's convenience extra is:

```bash
python -m pip install --upgrade "neuralqx[cuda]"
python -m pip check
```

The extra requests JAX's `cuda` dependencies. To choose a CUDA family explicitly, install neuraLQX and the matching JAX extra in the same command. For example, for CUDA 13:

```bash
python -m pip install --upgrade neuralqx "jax[cuda13]"
```

For a system requiring CUDA 12, use `jax[cuda12]` instead. The pip variants install the CUDA user-space libraries, but the machine still needs a compatible NVIDIA driver. JAX currently lists a Linux driver requirement of at least 580 for CUDA 13 and 525 for CUDA 12, together with GPU architecture requirements. Check the [JAX NVIDIA installation instructions](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu) against the actual hardware before choosing the variant.

If your cluster supplies CUDA and cuDNN through environment modules, follow JAX's [local CUDA installation route](https://docs.jax.dev/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-locally-harder), such as `jax[cuda13-local]`, with the versions required there. A loaded module's library paths can override pip-installed libraries, so choose a consistent installation route with your cluster's guidance.

Installing plain `neuralqx` on a machine that has a GPU does not by itself install a working GPU backend. Conversely, once a compatible CUDA backend is available, there is no `.to("cuda")` step for this example. JAX places the numerical arrays on its selected backend. These instructions do not establish support for AMD, Intel, or Apple GPU plugins. Consult the [JAX platform matrix](https://docs.jax.dev/en/latest/installation.html#supported-platforms) for those environments and test the neuraLQX operations you need.

## Check the allocated device

Run the following within your GPU allocation or on your GPU workstation. Set the environment before starting Python:

```bash
JAX_PLATFORMS=cuda python - <<'PY'
import jax

print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
assert all(device.platform == "gpu" for device in jax.devices())
PY
```

`JAX_PLATFORMS=cuda` makes an unavailable CUDA backend an error instead of allowing an unintended CPU run. Seeing a GPU in `nvidia-smi` is a useful driver check, but the JAX device check establishes whether the Python environment can use it.

On a workstation with several GPUs, `CUDA_VISIBLE_DEVICES` can restrict which ones the process sees. For example, `CUDA_VISIBLE_DEVICES=0` exposes one device. Inside a scheduler allocation, preserve the visibility assigned by the scheduler. Its logical GPU numbers may differ from the physical numbering on the machine.

## Run the same calculation

Use the unchanged [simulation.py](examples/simulation.py) introduced on the [CPU page](cpu.md#the-calculation-used-throughout-these-guides):

```bash
JAX_PLATFORMS=cuda python simulation.py
```

If you are selecting one GPU on an unscheduled workstation, use:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python simulation.py
```

The expected topology is one process and one visible device. The model, optimiser, and complete `(64, 16, 4)` sample batch fit on that device. neuraLQX's sample-sharding setting is enabled by default, but an axis of size one does not partition the batch. Operator sharding likewise has no cross-device work to distribute on a single GPU.

The Python control flow still runs on the host. Compiled sampling, model evaluation, and numerical kernels run on the GPU. Repeatedly converting large arrays to NumPy, printing arrays, or asking for a scalar during every update can introduce transfers and synchronisation. The example prints diagnostics only at the beginning and end.

## Memory and precision

neuraLQX enables JAX's 64-bit mode by default and uses `float64` and `complex128` as its configured real and complex defaults. GPU support for these dtypes does not imply equal performance across GPU models. Keep the precision required by the calculation and compare numerical results before reducing it. The [configuration page](configuration.md#precision-and-memory) explains the relevant settings.

JAX normally reserves a substantial part of GPU memory when it first executes work. If several applications share the device, its allocation policy can be adjusted before starting the process. For example:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda python simulation.py
```

Disabling preallocation changes memory allocation behaviour, not the memory needed by the arrays. It may also increase fragmentation. The [JAX memory guide](https://docs.jax.dev/en/latest/gpu_memory_allocation.html) explains the available policies. For an actual out-of-memory error, first identify whether model parameters, sample buffers, or connected-state evaluation dominate. [Chunking and operator streaming](sharding.md#chunking-and-streaming) address different temporary allocations.

## A single GPU batch job

Download [single-gpu.sh](examples/single-gpu.sh) beside `simulation.py`. Activate or prepare the GPU environment before submission, and adjust the account, partition, memory, and time limit to your cluster.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=nqx-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=nqx-gpu-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda
export NEURALQX_DISTRIBUTED=0
export NEURALQX_SHARDING=samples
srun --ntasks=1 python -u simulation.py
```

Submit with `sbatch single-gpu.sh`. Compared with the [CPU job](cpu.md#a-cpu-batch-job), this requests one GPU and selects CUDA. It still launches one Python process. The script deliberately relies on Slurm's device visibility rather than overwriting `CUDA_VISIBLE_DEVICES`.

For more devices on the same host, continue with [multiple GPU execution](multi-gpu.md). The simulation code stays the same, while the sample batch is divided among the visible devices.
