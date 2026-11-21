---
title: Applying operators
---

# Applying operators

An operator provides matrix elements between basis configurations. A variational state combines those matrix elements with wavefunction amplitudes to estimate an expectation value. This page develops that connection, starting with the exact expression and ending with the methods you use in a simulation.

## From the expectation value to the local estimator

Let $\psi_\theta(\sigma)$ be the amplitude of configuration $\sigma$, with parameters $\theta$. Its norm is

$$
Z_\theta=\langle\psi_\theta|\psi_\theta\rangle
=\sum_\sigma|\psi_\theta(\sigma)|^2.
$$

The normalised expectation of $A$ is

$$
\begin{aligned}
\langle A\rangle_\theta
&=\frac{\langle\psi_\theta|A|\psi_\theta\rangle}{Z_\theta}\\
&=\frac{1}{Z_\theta}
\sum_{\sigma,\sigma'}
\psi_\theta(\sigma)^*
A_{\sigma,\sigma'}\psi_\theta(\sigma').
\end{aligned}
$$

For each configuration with nonzero amplitude, multiply and divide the inner expression by $\psi_\theta(\sigma)$. This gives

$$
\langle A\rangle_\theta
=\sum_\sigma
\underbrace{\frac{|\psi_\theta(\sigma)|^2}{Z_\theta}}_{p_\theta(\sigma)}
\underbrace{\sum_{\sigma'}A_{\sigma,\sigma'}
\frac{\psi_\theta(\sigma')}{\psi_\theta(\sigma)}}_{A_{\mathrm{loc}}(\sigma)}.
$$

The first factor is the Born probability distribution. The second is the **local estimator**,

$$
A_{\mathrm{loc}}(\sigma)
=\frac{(A\psi_\theta)(\sigma)}{\psi_\theta(\sigma)}
=\sum_{\sigma'}A_{\sigma,\sigma'}
\frac{\psi_\theta(\sigma')}{\psi_\theta(\sigma)}.
$$

Configurations where $\psi_\theta(\sigma)=0$ carry no Born weight and do not contribute to the original linear expectation. The ratio is evaluated on sampled configurations with nonzero amplitude. The [squared estimator](squared-estimators.md#wavefunction-zeros-and-support) needs an additional support consideration because its numerator measures the norm of a transformed state.

If the network returns the log amplitude $\ell_\theta=\log\psi_\theta$, the ratio is evaluated as

$$
\frac{\psi_\theta(\sigma')}{\psi_\theta(\sigma)}
=\exp\bigl[\ell_\theta(\sigma')-\ell_\theta(\sigma)\bigr].
$$

There is no need to know $Z_\theta$ to evaluate this expression. Sampling the Born distribution then turns the full basis sum into

$$
\langle A\rangle_\theta
\approx\frac1M\sum_{r=1}^{M}A_{\mathrm{loc}}(\sigma_r),
\qquad \sigma_r\sim p_\theta.
$$

The operator sum inside each local estimator still includes all of its relevant connected configurations. Sampling replaces the outer average over $\sigma$. It does not replace the operator by a random choice of one of its transitions.

For a diagonal observable $A_{\sigma,\sigma'}=a(\sigma)\delta_{\sigma,\sigma'}$, the amplitude ratio cancels and $A_{\mathrm{loc}}(\sigma)=a(\sigma)$. For an off-diagonal operator, the ratio carries the wavefunction information that cannot be obtained from the configuration alone. Local values can be complex even when the exact expectation of a Hermitian operator is real.

## Read a connected row

[`get_conn_padded`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.ComputationalOperator.get_conn_padded) returns the ingredients of the inner sum. The input is the row configuration $\sigma$, each output configuration is a column label $\sigma'$, and the accompanying element is $A_{\sigma,\sigma'}$.

For a single configuration of length $D$, given by `hilbert.size`, and padded connection capacity $K$, given by `operator.max_conn_size`, the outputs have shapes `(K, D)` and `(K,)`. Leading batch axes are preserved.

```python
import neuralqx as nqx
import jax.numpy as jnp
from neuralqx.operator.prebuilt.wcl import U1HolonomyOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
shift = U1HolonomyOperator(hilbert, edge=graph.edges[0])

connected, elements = shift.get_conn_padded(jnp.array([0], dtype=jnp.int32))
batch = jnp.stack((hilbert.all_states(), hilbert.all_states()))
batch_connected, batch_elements = shift.get_conn_padded(batch)
print(connected.tolist(), elements.tolist())
print(batch_connected.shape, batch_elements.shape)
```

```text
[[1]] [1.0]
(2, 3, 1, 1) (2, 3, 1)
```

This row contains $\langle0|R|1\rangle=1$. It does not say that $R|0\rangle=|1\rangle$. Ket action uses matrix columns, while the local estimator uses rows. The [U(1) holonomy convention](../lqx/u1.md#edge-and-loop-holonomies) explains the corresponding cyclic shifts in both descriptions.

### Padding and repeated targets

Zero matrix elements mark inactive entries. Several branches may reach the same target, and their amplitudes must be added. The padded capacity therefore need not equal the number of distinct nonzero targets.

Use `get_conn_coalesced` to merge repeated targets while retaining a fixed padded shape. Use `get_conn` for a single row with padding removed, or `n_conn` to count distinct nonzero targets. Removing padding produces a data-dependent shape, so `get_conn` is intended for inspection outside JIT. The [JAX guide to compilation](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html#just-in-time-compilation-with-jax-jit) explains why fixed shapes are useful in compiled calculations.

```python
import neuralqx as nqx
import jax.numpy as jnp
from neuralqx.operator.prebuilt.wcl import U1HolonomyOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
shift = U1HolonomyOperator(hilbert, edge=0)
cancelled = shift - shift

connected, elements = cancelled.get_conn(jnp.array([0], dtype=jnp.int32))
print(cancelled.max_conn_size)
print(cancelled.n_conn(hilbert.all_states()).tolist())
print(connected.shape, elements.shape)
```

```text
2
[0, 0, 0]
(0, 1) (0,)
```

The padded expression contains two branches, but their amplitudes cancel. The expectation engine handles these contributions as amplitudes, including their interference. It does not count branches as distinct physical outcomes.

## Check the derivation on a complete small basis

For three charge states $-1,0,1$, take the normalised amplitude vector $(1,2,1)/\sqrt6$. The following evaluates the local estimator of a cyclic holonomy and compares its Born average with direct matrix multiplication.

```python
import neuralqx as nqx
import jax.numpy as jnp
import numpy as np
from neuralqx.operator.prebuilt.wcl import U1HolonomyOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
operator = U1HolonomyOperator(hilbert, edge=0)
basis = hilbert.all_states()
psi = jnp.array([1.0, 2.0, 1.0]) / jnp.sqrt(6.0)

connected, elements = operator.get_conn_padded(basis)
columns = hilbert.states_to_numbers(connected)
local_values = jnp.sum(elements * psi[columns] / psi[:, None], axis=-1)
born_average = jnp.sum(jnp.abs(psi)**2 * local_values)
matrix_average = jnp.vdot(psi, operator.to_dense() @ psi)

np.testing.assert_allclose(born_average, matrix_average)
print(local_values.tolist())
print(round(float(born_average), 6))
```

```text
[2.0, 0.5, 1.0]
0.833333
```

This explicit sum is a verification on a small indexable space. A neural quantum state replaces the stored amplitude vector by network evaluations and the full Born average by sampling.

## Measure an operator with a variational state

[`VariationalState`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState) provides `expect`, `local_estimators`, and `expect_and_grad`. Here is a complete sampled calculation of a charge probability using a small [`RBM`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.RBM) and [`MetropolisLocal`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisLocal).

```python
import neuralqx as nqx
from neuralqx.operator.prebuilt import ChargeColoringOperator

a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
network = nqx.nn.RBM(hilbert.size, hidden_features=4, complex_output=False, rngs=0)
sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=4)
state = nqx.vqs.VariationalState(
    sampler, network, n_samples=256, n_discard_per_chain=16, seed=1
)
projector = ChargeColoringOperator(hilbert, site=0, charge=0)

local = state.local_estimators(projector)
estimate = state.expect(projector)
print(local.data.shape)
print(estimate)
```

The local values are zero or one according to whether the selected component has charge zero. Their mean estimates that charge's probability in the current wavefunction. The example measures the initial network, before any optimisation.

`local.data` follows the leading sample shape, here `(4, 64)`. `expect` returns [`Stats`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.Stats), including `mean`, `variance`, and `error_of_mean`. Inspect [sampling diagnostics](../samplers/diagnostics.md) and convergence rather than treating a finite batch as an exact result. For nonstandard sampling distributions, use the supported importance-sampling interface instead of applying an unweighted Born average to unrelated samples.

The state draws a batch when needed and caches it. Repeated expectations use that batch until it is replaced or invalidated. Call `state.sample()` when you intend to advance the chains and collect fresh configurations. This reuse is useful when comparing related observables on the same sample set.

Passing a list or tuple of operators to `state.expect` returns the expectation of their sum. To obtain separate statistics, evaluate each object individually, for example with a dictionary comprehension over your observable names. The same distinction applies when combining [penalty objectives](penalties.md#combine-objectives-and-optimise).

## Matrices, batches, and memory

[`to_sparse`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.DiscreteOperator.to_sparse) returns a JAX sparse `BCOO` matrix, and `to_dense` returns a dense JAX array. Both enumerate the entire basis and require an indexable Hilbert space. Their rows and columns follow `hilbert.all_states()` and its numbering methods. The `max_states` argument bounds basis enumeration, not the final matrix's byte size. Dense storage grows as the square of the Hilbert dimension.

For larger calculations, work with connectivity and expectations. `get_conn_padded(..., use_streaming=True, chunk_size=...)` evaluates the input batch in chunks. It still returns the requested connectivity arrays, so chunking reduces intermediate work rather than eliminating the final output allocation. When evaluating a state, `chunk_size` controls network evaluation, while `operator_streaming=True` and `operator_streaming_chunk_size` control operator work.

Ordinary observable gradients and factorised quadratic gradients require different estimators. Use `state.expect_and_grad(objective)` to let the implementation select the appropriate route. The next pages explain [operator algebra](algebra.md), the [squared estimator derivation](squared-estimators.md), and [expectation-level penalties](penalties.md).
