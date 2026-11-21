---
title: Sharding in neuraLQX
---

# Sharding in neuraLQX

neuraLQX distributes work at several stages of a variational calculation. The most useful distinction is between distributing configurations and distributing operator terms. Both preserve the same mathematical objective, but they place different data on each device and incur different communication costs.

The [JAX parallelism page](jax-parallelism.md) introduces meshes and global arrays. This page describes the policies implemented in neuraLQX v2.0.0. The [configuration reference](configuration.md) gives their exact settings, while the hardware guides demonstrate them with [one shared simulation](cpu.md#the-calculation-used-throughout-these-guides).

## Sample and chain sharding

Sample sharding is enabled by default. It is a form of data parallelism, with different configurations evaluated using the same model parameters. For the supplied Metropolis samplers, the leading chain dimension is divided across the visible JAX devices. Each device evolves its own subset of chains, stores complete configurations, and evaluates the network on those configurations. The time dimension within a chain and the configuration width stay together.

For a retained sample array with shape

$$
(C,L,N),
$$

$C$ is the global number of chains, $L$ the retained length of each chain, and $N$ the Hilbert space configuration width. With $D$ sample devices and divisible $C$, each device owns `(C / D, L, N)`. The total number of retained configurations is $CL$, regardless of how many devices store them.

In the [shared example](cpu.md), `n_chains=64` and `n_samples=1024`. Four devices each own 16 chains and 256 retained configurations. Eight devices each own eight chains and 128 retained configurations. Neither launch multiplies the requested sample count by the number of devices.

The sampler rounds a requested `n_chains` upward when necessary to make it divisible by the participating device count. The sample request is then adjusted to the nearest positive multiple of the resolved chain count, choosing the larger value on a tie. Inspect the resolved shape instead of assuming that an arbitrary request was used unchanged. With `n_chains_per_device`, the total chain count deliberately grows with the number of devices. See [scaling the workload](multi-gpu.md#fixed-work-and-growing-work).

The sampler splits random keys across sample partitions. In a coordinated multi-process run, use the same initial model and sampler seeds on all processes so they describe the same global state. Do not independently add the process number to the model seed. Different device counts can change key allocation and floating-point reductions, so identical seeds do not imply identical trajectories across hardware layouts.

## Operator terms

For a sum $H=\sum_t c_t O_t$, the local estimator is

$$
H_{\mathrm{loc}}(\sigma)
=\sum_t c_t\sum_{\sigma'}
(O_t)_{\sigma,\sigma'}
\frac{\psi(\sigma')}{\psi(\sigma)}.
$$

The terms can be evaluated independently before their contributions are summed. Operator sharding distributes this term work. For supported [`ComputationalOperator`](../api/public/neuralqx/operator/index.md#neuralqx.operator.ComputationalOperator) terms, the device dispatch generates each assigned term's connections and evaluates its connected configurations on that device. The operators remain [matrix free](../concepts/operators/index.md#matrix-free-by-design).

Enable this policy before importing neuraLQX:

```bash
NEURALQX_SHARDING=samples,operators JAX_PLATFORMS=cuda python simulation.py
```

The expectation machinery flattens linear sums and operator sequences into terms. A graph edge is not automatically a separately scheduled term, and splitting a sum across devices does not divide every individual operator's connection dimension. The decomposition exposed by the [operator algebra](../concepts/operators/algebra.md) determines what can be scheduled.

The default planner estimates term cost from `max_conn_size`, assigns terms to balance those estimates, and pads the schedule with zero-work slots if necessary. This is a heuristic, not a measurement of each term's actual runtime. Different connection patterns and network evaluations can still produce imbalance. Non-computational or specialised estimator paths can use a fallback rather than the computational term-dispatch path.

## Combined operator and sample layouts

The layout is chosen for the execution phase. Sampling continues to use the devices as sample partitions even when operator sharding is enabled. Operator evaluation may then redistribute the batch onto an operator-only mesh or a mesh with both operator and sample axes.

For $T$ computational terms on $D>1$ devices, the default operator planner uses all $D$ devices as operator partitions when $T\ge D$. The full sample batch is then replicated across those partitions, and each partition evaluates its assigned terms. With fewer terms, the planner tries a factorisation $D=D_O D_S$ where $D_O\le T$, $D_S>1$, and the flattened sample count is divisible by $D_S$. It chooses the candidate with the largest $D_O$. This combined layout requires sample sharding to be enabled. If no suitable factorisation exists, it uses the operator-only layout with padding.

![During sampling, eight devices own eight different chain groups. During evaluation of four operator terms, the same devices form four operator partitions and two sample partitions.](../assets/images/distributed/phase-layouts.svg)

*The shared example on eight devices. Sampling partitions 64 chains eight ways. Operator evaluation partitions four terms and two halves of the flattened batch. Parameters are available on every device in both phases.*

For the example's four squared-charge terms and 1,024 samples, four devices use four operator partitions with the entire sample batch available to each. Eight devices use four operator partitions and two sample partitions, each containing 512 configurations. The operator axis combines four contributions for each configuration. The final expectation then reduces over the configurations.

This redistribution is why the [generic mesh-size settings](configuration.md#mesh-names-and-sizes) are not a promise that one fixed mesh is used everywhere. The current sampling and operator planners make phase-specific choices. Keep the size overrides unset for the standard execution paths unless you are deliberately using a lower-level mesh interface.

## Inspect and compare operator execution

Save [simulation.py](examples/simulation.py) in the current directory before running this example. It creates one sample batch and evaluates that same batch with operator sharding disabled and enabled. Keeping the samples fixed separates execution differences from Monte Carlo fluctuations.

```python
import jax
import numpy as np
from simulation import build_calculation

state, objective, _ = build_calculation()
samples = state.samples
ordinary = state.local_estimators(
    objective, samples=samples, operator_sharding=False
)
distributed = state.local_estimators(
    objective, samples=samples, operator_sharding=True
)
np.testing.assert_allclose(
    jax.device_get(ordinary.data),
    jax.device_get(distributed.data),
    rtol=1e-12,
    atol=1e-12,
)
print(distributed.to_stats().mean)
```

Use a single process with multiple visible devices for this comparison. Four logical CPU devices are sufficient to exercise the operator dispatch. A one-device run remains valid but cannot demonstrate cross-device scheduling. The tight tolerance here is appropriate for the example's diagonal integer-valued objective, not a universal tolerance for arbitrary operators.

The per-call `operator_sharding` option is accepted by [`local_estimators`](../api/public/neuralqx/vqs/estimators/index.md#neuralqx.vqs.estimators.local_estimators). It can request the computational operator path independently of the global default. A full VMC run uses the configuration because the driver calls the state without a per-call override. Related options can be grouped in [`LocalEstimatorOptions`](../api/public/neuralqx/vqs/estimators/index.md#neuralqx.vqs.estimators.LocalEstimatorOptions).

## Parameters and neural network parallelism

The current default execution uses complete model parameters for each sample or operator partition. Gradients combine information from the global calculation, and every process applies the same optimiser update. Parameter memory is therefore not divided by the sample-device count.

The setting `parameter_sharding_enabled` currently enables a placement helper that applies replicated `PartitionSpec()` placement. Despite the setting's name, it does not choose partitions of individual weight matrices or split a network into stages. Enabling `NEURALQX_SHARDING=all` does not make an oversized network fit by distributing its weights.

**Neural network tensor parallelism (TP) and pipeline parallelism (PP) are under development and are not part of v2.0.0.** TP would partition operations and tensors within network layers. PP would place different stages of the network on different devices. Layer axis annotations alone do not implement either execution strategy. The [custom network guide](../developers/networks/parameters.md) explains how model parameters enter the framework, but these guides do not prescribe a supported TP or PP configuration for this release.

## Chunking and streaming

Sharding divides work between devices. Chunking processes smaller pieces sequentially within an evaluation. They can be used together, but they address different limits.

`VariationalState.chunk_size` limits model evaluation batches. Operator streaming limits how many flattened input configurations are passed through connection generation at once. Its setting is `operator_streaming_chunk_size`, with a default of 1,024 when streaming is enabled. It is a count of input configurations, not operator terms or connected outputs.

For example, in a fresh run:

```bash
NEURALQX_OPERATOR_STREAMING_ENABLED=1 \
NEURALQX_OPERATOR_STREAMING_CHUNK_SIZE=64 \
JAX_PLATFORMS=cuda python simulation.py
```

This preserves the example's global sample count and sharding policy. It changes the size of connection-generation chunks. Streaming can reduce temporary workspace, but paths that assemble padded outputs still need storage for those outputs. Likewise, neither model chunking nor operator streaming removes the need to store the network parameters. See [batching and chunking](../concepts/vqs/batching.md) and [memory troubleshooting](troubleshooting.md#out-of-memory-errors) when choosing which allocation to reduce.

Proceed to the [CPU guide](cpu.md) for the complete simulation, or inspect the [parallelism configuration](configuration.md) for the supported launch and layout controls.
