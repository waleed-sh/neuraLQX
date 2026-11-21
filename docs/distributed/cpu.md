---
title: CPU
---

# CPU

## Installation and environment

The standard installation provides the numerical dependencies for a CPU calculation. Create an environment with Python 3.11 or newer, then install the released package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade neuralqx
python -m pip check
```

These shell commands assume Linux or macOS. The [installation guide](../start/installation.md) explains the package extras and environment requirements. The CPU backend is also the starting point on a machine without a supported accelerator installation.

Set `JAX_PLATFORMS=cpu` in the launch command when you want the run to use CPUs even if another backend is installed. A single JAX CPU device can use several host cores through its compiled numerical operations. The device count is therefore not a count of CPU cores, and asking a scheduler for eight CPU cores does not create eight JAX devices.

## The calculation used throughout these guides

We will minimise a small squared-charge objective on a graph with four edges. Each edge has a U(1) label $q_e\in\{-2,-1,0,1,2\}$, and the operator is

$$
H=\sum_{e=0}^{3}Q_e^2,
\qquad
H|q\rangle=\left(\sum_{e=0}^{3}q_e^2\right)|q\rangle.
$$

This deliberately simple objective makes it possible to inspect the distribution of work without introducing a large physical model. Its minimum is zero at the configuration with all charges zero. It is an unconstrained example, rather than a gravitational Hamiltonian or a gauge invariant calculation. The [U(1) Hilbert space guide](../concepts/hilbert/u1.md) explains how to choose the constraints required by your research problem.

The graph, objective, network, optimiser, and total sample count stay the same in the [single GPU](single-gpu.md), [multiple GPU](multi-gpu.md), and [multiple node](multi-node.md) guides. Only the execution environment and launch change. The example is small enough to test on a laptop, so its timings should not be used as evidence that a larger device allocation is beneficial.

Download [simulation.py](examples/simulation.py) and save it in your working directory. The complete script is shown below.

```python
"""One VMC calculation used throughout the distributed simulation guides."""

# Import neuraLQX before Optax so configured distributed startup runs first.
# ruff: noqa: I001

import argparse

import jax
import neuralqx as nqx
import optax
from neuralqx.operator import SumOperator
from neuralqx.operator.prebuilt.wcl import NumberOperator


def build_calculation():
    """Construct the same physical problem and global batch on every process."""
    vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    edges = [(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
    graph = nqx.graph.Graph(edges, is_planar=True)
    hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=2)
    charges = tuple(NumberOperator(hilbert, edge=i) for i in range(4))
    objective = SumOperator(tuple(charge @ charge for charge in charges))

    sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=64)
    model = nqx.nn.MLP(hilbert.size, hidden_features=(32, 32), complex_output=False, rngs=17)
    state = nqx.vqs.VariationalState(
        sampler,
        model,
        n_samples=1024,
        n_discard_per_chain=16,
        sampler_seed=23,
        chunk_size=128,
    )
    driver = nqx.driver.VMC(objective, optax.sgd(learning_rate=0.01), variational_state=state)
    return state, objective, driver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    print(
        f"Process {jax.process_index()}/{jax.process_count()}: "
        f"{jax.local_device_count()} local, {jax.device_count()} global devices",
        flush=True,
    )
    state, objective, driver = build_calculation()
    driver.run(args.steps)

    # Every process participates in the expectation and any communication.
    result = state.expect(objective)
    mean = jax.device_get(result.mean)
    if jax.process_index() == 0:
        print(f"Mean squared-charge objective: {mean}")
        print(f"Global samples: {state.samples.shape}")
        print(f"Sample sharding: {state.samples.sharding}")

    # Only local shards are inspected here, so every process can print its own.
    for shard in state.samples.addressable_shards:
        print(
            f"Process {jax.process_index()}, device {shard.device.id}: "
            f"index={shard.index}, local shape={shard.data.shape}",
            flush=True,
        )


if __name__ == "__main__":
    main()
```

The [`NumberOperator`](../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.NumberOperator) measures one edge charge. Each `charge @ charge` is an ordinary matrix free product, and [`SumOperator`](../api/public/neuralqx/operator/index.md#neuralqx.operator.SumOperator) keeps the four contributions available as separate terms. This choice will let us demonstrate [operator sharding](sharding.md#operator-terms) without changing the mathematical objective.

[`MetropolisLocal`](../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisLocal) supplies 64 chains. The [`MLP`](../api/public/neuralqx/nn/index.md#neuralqx.nn.MLP) represents a real log amplitude, while [`VariationalState`](../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState) requests 1,024 samples across all chains. These are global counts. With 64 chains, the retained chain length is 16, and the sample array has shape `(64, 16, 4)`. The final dimension contains the complete edge configuration.

The model evaluation limit `chunk_size=128` controls how many configurations are processed in a model chunk. It does not select a device count or change the number of Markov chains. The [batching and chunking guide](../concepts/vqs/batching.md) explains the distinction.

## Run on a CPU

From the directory containing the script, run:

```bash
JAX_PLATFORMS=cpu python simulation.py
```

The script prints the process and device counts, performs ten VMC updates, evaluates the final objective, and reports the sample placement. On the usual single-device CPU backend, the global sample shape and local sample shape are both `(64, 16, 4)`. All configurations and model parameters reside on that device.

The reported objective is a Monte Carlo estimate after a short optimisation. Ten steps establish that the setup runs, rather than demonstrate convergence. Use `python simulation.py --steps 100` for a longer run, and follow the [driver monitoring guide](../concepts/drivers/monitoring.md) when deciding whether the state has converged.

The first iteration includes compilation and setup. When comparing execution times, separate that work from subsequent updates and wait for results to finish. See [timing and memory checks](troubleshooting.md#measure-completed-work).

## Inspect several logical CPU devices

You can exercise the same sample-sharding path without a GPU by starting a fresh process with four logical CPU devices:

```bash
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 python simulation.py
```

Now the global shape is still `(64, 16, 4)`, while each device owns `(16, 16, 4)`. All four devices belong to the same Python process and share the host's physical resources. This is useful for inspecting layouts and testing a parallel program. It does not simulate GPU performance or provide four additional CPUs.

Choose the logical device count before JAX initialises its backend. Changing it inside an already running notebook is too late. Restart the kernel with the intended environment when testing another topology. The [JAX parallelism guide](jax-parallelism.md) explains global shapes, meshes, and local shards.

## A CPU batch job

Download [cpu.sh](examples/cpu.sh) beside `simulation.py`. The job assumes that `.venv` already exists in the directory from which you submit it. Install packages before submission rather than having every job recreate the environment.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=nqx-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=nqx-cpu-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export JAX_PLATFORMS=cpu
export NEURALQX_DISTRIBUTED=0
export NEURALQX_SHARDING=samples
srun --ntasks=1 python -u simulation.py
```

Submit with `sbatch cpu.sh`, adding the account and partition required by your site. The resource numbers are example requests, not minimum requirements. This job launches one Python process with eight allocated host cores. The [Slurm batch-script reference](https://slurm.schedmd.com/sbatch.html) describes the scheduler directives and output filename substitutions.

To move the calculation to an accelerator, continue with [single GPU execution](single-gpu.md). You will keep `simulation.py` and install the appropriate JAX backend.
