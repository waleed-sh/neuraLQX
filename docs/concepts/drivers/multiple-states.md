---
title: Multiple states
---

# Multiple states

Several independently initialised variational states can all converge towards the same low-energy state. When the purpose is to resolve several distinct states, an overlap penalty can discourage that collapse while the individual energies are reduced.

[`MultiVMCDriver`](../../api/public/neuralqx/driver/index.md#neuralqx.driver.MultiVMCDriver), also exported as [`MultiStateVMC`](../../api/public/neuralqx/driver/index.md#neuralqx.driver.MultiVMCDriver), coordinates this joint calculation. It takes one objective, one optimiser definition, and a sequence of variational states. All states must describe the same Hilbert space, although their networks and parameter trees can differ.

## The joint objective

Let $|\Psi_i\rangle$ denote the normalised wavefunction of state $i$, and let $E_i=\langle\Psi_i|H|\Psi_i\rangle$. With `orthogonality_strength` set to $\lambda$, the gradients correspond to

$$
\begin{aligned}
\mathcal J
&=\sum_{i=0}^{K-1}E_i
+\lambda\sum_{i<j}F_{ij},\\[4pt]
F_{ij}
&=|\langle\Psi_i|\Psi_j\rangle|^2.
\end{aligned}
$$

For positive $\lambda$, the overlap contribution favours distinct states. Setting it to zero removes pair-overlap calculations and leaves the individual objective gradients. The default strength is `1.0`, whose significance depends on the scale of the objective. It is not a universal choice for separating states.

This is a penalty method, not an exact orthogonalisation step. A small strength may allow states to remain too similar. A large one can make overlap optimisation dominate energy reduction. The driver neither guarantees exact orthogonality at a finite iteration nor assigns states to a sorted sequence of eigenvalues. Examine the individual expectations and overlaps when deciding what has been obtained.

The same interface can use a real constraint objective. In that case the interpretation is a collection of variational states with low constraint cost and reduced mutual overlap, rather than automatically a physical energy spectrum.

## How the overlap is sampled

For ordinary Born sampling, draw $x$ from state $i$ and $y$ independently from state $j$. Define

$$
R_{ij}(x,y)
=\frac{\psi_j(x)}{\psi_i(x)}
\frac{\psi_i(y)}{\psi_j(y)}.
$$

Then

$$
F_{ij}
=\mathbb E_{x\sim p_i,\,y\sim p_j}
\left[\operatorname{Re}R_{ij}(x,y)\right].
$$

The estimator evaluates each network on configurations drawn from both states. Under Born sampling, its gradient includes the effect of both the explicit amplitude ratios and the sampling distributions, following the distinction developed in [expectations and gradients](../vqs/expectations.md#why-differentiating-a-sampled-mean-is-not-sufficient).

The current pair routine pairs the two flattened batches and uses the smaller available sample count. Extra configurations in the longer batch do not enter that pair estimate. Similar sample counts avoid wasting this work. Use distinct random seeds and independently evolving chains for the different states.

## Importance sampling and normalization

The overlap routine also supports non-Born sampling. The correction is necessary because averaging $R_{ij}$ under different distributions does not generally give the squared normalized overlap. Each state may use a different proposal, provided that its proposal covers its own wavefunction's Born support.

Let $\widetilde q_i(x)$ and $\widetilde q_j(y)$ be the unnormalised densities that generated the two batches, and write $Q(x,y)=\widetilde q_i(x)\widetilde q_j(y)$. Define

$$
\begin{aligned}
A_{ij}(x,y)
&=\frac{\operatorname{Re}\!\left[
\psi_i(x)^*\psi_j(x)\psi_j(y)^*\psi_i(y)
\right]}{Q(x,y)},\\[6pt]
B_{ij}(x,y)
&=\frac{|\psi_i(x)|^2|\psi_j(y)|^2}{Q(x,y)}.
\end{aligned}
$$

The normalized overlap and its finite-batch estimate are then

$$
\begin{aligned}
F_{ij}
&=\frac{\mathbb E_{q_iq_j}[A_{ij}]}{\mathbb E_{q_iq_j}[B_{ij}]},\\[6pt]
\widehat F_{ij}
&=\frac{\sum_r A_{ij}(x_r,y_r)}{\sum_r B_{ij}(x_r,y_r)}.
\end{aligned}
$$

Unknown normalization constants in the proposal densities cancel. With ordinary Born sampling, $B_{ij}$ is constant and this reduces to the unweighted paired estimator above. With nonconstant weights, it is a self-normalized ratio estimator and has the usual finite-batch bias. Larger effective sample sizes reduce that bias, but importance sampling does not by itself guarantee lower variance.

[`fidelity_expect_and_grad_joint`](../../api/public/neuralqx/driver/multi_vmc/index.md#neuralqx.driver.multi_vmc.fidelity_expect_and_grad_joint) reads the frozen density attached to each [importance-sampled batch](../samplers/importance.md). This includes densities modified by operator blurring. If an adaptive exponent has changed since the batch was collected, the stored density still describes the batch actually being used. Without an explicit measure, the routine infers the density from that state's `machine_pow` and current wavefunction.

During differentiation, the proposal densities are held fixed and both physical numerator and normalization are differentiated,

$$
\partial_\theta\widehat F_{ij}
=\frac{\sum_r\partial_\theta A_{ij,r}
-\widehat F_{ij}\sum_r\partial_\theta B_{ij,r}}
{\sum_r B_{ij,r}}.
$$

This gives the Born result independently of the proposal exponent. Under Born sampling it is equivalent to the direct-plus-score gradient described above. An additional sampling-score term must not be added to this density-weighted derivative.

The implementation forms amplitude products before dividing by the sampling density. It therefore avoids multiplying zero Born weights by undefined wavefunction ratios when operator blurring samples an exact node. The model must still have well-defined derivatives on the configurations being evaluated, as discussed in the [importance sampling guide](../samplers/importance.md#reach-configurations-connected-by-an-operator).

For a joint calculation whose energies and overlaps both retain their Born interpretation, use the supported `ImportanceSampling` interface with the sampler's default `machine_pow=2`. Changing `machine_pow` alone is corrected by the overlap routine, but does not opt the individual energy estimators into the VQS importance-sampling machinery. Supply external importance batches with `state.set_samples(..., log_density=...)` so their generating density is available.

Although the exact fidelity lies between zero and one, a noisy paired estimate need not. With importance weights, reported variances and error bars describe the ratio estimator's influence values. The overlap routine also flattens the chain axes before constructing its statistics. Its reported uncertainty is therefore not a replacement for checking mixing and stability across larger batches and independent calculations.

## Run two states

This example jointly minimises an expected squared charge while penalising overlap. It deliberately runs only two iterations so that it is quick to execute. Those iterations establish that the calculation is configured, not that distinct eigenstates have converged.

```python
import jax.numpy as jnp
import neuralqx as nqx
import optax
from neuralqx.operator.prebuilt.wcl import NumberOperator

graph = nqx.graph.Graph([((0, 0, 0), (1, 0, 0))], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
states = tuple(
    nqx.vqs.VariationalState(
        nqx.sampler.MetropolisLocal(hilbert, n_chains=4),
        nqx.nn.MLP(1, hidden_features=(8,), complex_output=False, rngs=seed),
        n_samples=128,
        n_discard_per_chain=16,
        sampler_seed=seed + 20,
    )
    for seed in (10, 11)
)
charge = NumberOperator(hilbert, edge=0)
strength = 0.2
driver = nqx.driver.MultiVMCDriver(
    charge @ charge,
    optax.sgd(0.01),
    variational_states=states,
    orthogonality_strength=strength,
)

history = driver.run(2, obs={"Charge": charge})
record = history[-1]
joint_mean = (
    record["Energy"].mean
    + strength * record["orthogonality/0_1"].mean
)
print(driver.step_count)
print(len(driver.optimizer_states))
print(len(record["Charge"]))
print(bool(jnp.isfinite(joint_mean)))
```

```text
2
2
2
True
```

The optimiser definition is shared, but each state has its own optimiser state. An accepted joint iteration updates every member and increments the driver counter once. Rejecting that joint update through a callback skips all of its parameter updates.

An optional preconditioner is applied separately to each state's gradient. The built-in driver passes the same preconditioner object to these calls, so its most recent solver information belongs to the latest call. This is not a joint QGT containing cross-state parameter blocks.

## Read the joint record correctly

The driver's main loss statistics, including `driver.energy`, report the sum of the individual objective expectations. They do **not** include the overlap penalty, even though that penalty contributes to the gradients used for optimisation.

| Record key with the default `loss_name="Energy"` | Meaning |
| --- | --- |
| `Energy` | Sum of the individual energy statistics |
| `Energy/state_0`, `Energy/state_1`, ... | Each state's objective statistics |
| `orthogonality/0_1`, ... | Pair fidelity estimates before multiplying by the strength |
| A name supplied through `obs` | A list of observable statistics in state order |

The example explicitly reconstructs the joint objective's estimated mean. For more than two states, include every recorded pair. Do not reuse `Energy.error_of_mean` as the uncertainty of that penalised total, because it excludes the overlap terms and their correlations.

The summed energy statistics add the individual variances and combine their errors in quadrature. Their sample count is the minimum member count, and their `r_hat` is not a joint convergence diagnostic. Inspect each member's record instead. Pair entries are absent when the penalty is disabled.

Multi-state drivers retain history by default. Use `store_history=False` for a long run when a separate logger provides the record you need. Keys such as `"Energy/state_0"` are literal dictionary keys. With a runtime logger, access a field through `logger.series(("Energy/state_0", "Mean"))`, as explained in [logging and monitoring](../callbacks/logging.md#read-nested-and-multi-state-records).

## Cost, observables, and callbacks

There are $K(K-1)/2$ distinct overlap pairs. Increasing the number of states therefore adds pair work quadratically as well as adding each state's ordinary objective calculation. The network [chunk sizes](../vqs/batching.md) are used when evaluating the ratios, but they do not remove that growth in the number of pairs.

`driver.states` and `driver.state` both expose the tuple of states. A [callback](../callbacks/index.md) that assumes `context.state` is one VQS must be adapted before using it with a multi-state driver. Likewise, additional observables are lists of statistics, which should be separated into scalar entries if you want each member displayed in the live dashboard.

Use [checkpointing](checkpointing.md#custom-models-and-several-states) to save all members at a consistent update boundary. For final measurements, call `driver.estimate(...)` after the last update and inspect the returned state-ordered lists.
