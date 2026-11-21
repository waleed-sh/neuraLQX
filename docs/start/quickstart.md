---
title: Quick start
summary: Run a small SU(2) calculation on K5 with adaptive importance sampling and a neural wavefunction.
---

# Quick start

This guide builds a small variational calculation on the complete graph $K_5$. We will define a gauge invariant SU(2) Hilbert space, sample its spin networks, represent their amplitudes with a neural network, and minimise the standard quadratic Thiemann constraint. The example connects the main components of a simulation without requiring you to implement them yourself.

Follow the [installation guide](installation.md) first. The standard CPU installation is sufficient for this example. For an NVIDIA GPU, complete the [single GPU setup](../distributed/single-gpu.md) before running the code. Execute the Python blocks below in order in the same session, or combine them into one script.

## Define the graph and Hilbert space

[`K5Graph`](../api/public/neuralqx/graph/index.md#neuralqx.graph.K5Graph) connects every pair of its five vertices. It has ten edges, with four edges incident on each vertex. This gives us a small graph with nontrivial intertwiner degrees of freedom.

```python
import jax.numpy as jnp
import neuralqx as nqx
import optax

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(graph, two_j_max=1)

print(graph.n_vertices, graph.n_edges, hilbert.size)
```

```text
5 10 15
```

[`SU2Hilbert`](../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.SU2GaugeInvariantHilbert) constructs the gauge invariant space. The cutoff is given in doubled spins, so `two_j_max=1` permits $j_e=0$ and $j_e=\tfrac12$ on each edge. Admissible intertwiners couple the incident representations to an SU(2) singlet at every vertex. The Gauss constraint is therefore built into the basis, and we do not need to add a penalty for it.

Each configuration has 15 integer coordinates, consisting of ten doubled edge spins and five intertwiner channel labels. This is what `hilbert.size` measures, rather than the number of basis states. The [SU(2) Hilbert space guide](../concepts/hilbert/su2.md) explains the admissibility conditions, and the [state layout guide](../concepts/hilbert/state-layouts.md#the-su2-spin-network-layout) explains the stored labels.

## Choose the sampler

We need configurations on which to evaluate the wavefunction and its observables. [`MetropolisSU2`](../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSU2) generates them through moves compatible with the gauge invariant basis. We also enable [`ImportanceSampling`](../api/public/neuralqx/sampler/index.md#neuralqx.sampler.ImportanceSampling) so that the sampling distribution can adapt during optimisation.

```python
importance = nqx.sampler.ImportanceSampling(
    exponent=1.5,
    adaptive=True,
)

sampler = nqx.sampler.MetropolisSU2(
    hilbert,
    weights=(0.6, 0.0, 0.3, 0.1),
    stay_probability=0.05,
    n_chains=8,
    sweep_size=8,
    importance_sampling=importance,
)
```

The four `weights` choose among proposal families in the order **loop, edge, vertex, global**. Conditional on attempting a move rather than taking the explicit stay step, this example selects loop moves with probability 60%, vertex moves with probability 30%, and global proposals with probability 10%.

Loop moves change edge spins around a cycle and update affected intertwiner channels. Vertex moves change the channels at fixed edge spins. Global proposals provide access to configurations beyond a local sequence of moves. The edge family redraws a single spin within its integer or half-integer parity sector. At this cutoff there is no alternative spin in either sector, so we give that family zero weight. A positive global weight is required. The sampler includes the proposal corrections needed for detailed balance. See [SU(2) moves and mixture weights](../concepts/samplers/su2/moves.md) for the individual kernels.

The importance policy controls a different part of sampling. Starting from $\alpha=1.5$, the chains target

$$
q_\alpha(\sigma)\propto|\psi_\theta(\sigma)|^\alpha,
\qquad
w(\sigma)=|\psi_\theta(\sigma)|^{2-\alpha}.
$$

An exponent below two broadens the sampled distribution relative to the Born distribution. The importance weights $w$ correct expectations back to the physical state. They are distinct from the four proposal probabilities above. The variational state applies this correction to expectations, gradients, and stochastic reconfiguration, so we do not multiply the measurements by weights ourselves.

With `adaptive=True`, gradient evaluations periodically adjust the exponent for subsequent batches, using an estimate of gradient sampling variance. Leave the sampler's `machine_pow` at its default value of two when using this policy. The [importance sampling guide](../concepts/samplers/importance.md) explains the adaptation controls and diagnostics.

## Represent the wavefunction

The built-in [`MLP`](../api/public/neuralqx/nn/index.md#neuralqx.nn.MLP) receives a configuration and returns its complex log amplitude,

$$
\log\psi_\theta(\sigma)
=a_\theta(\sigma)+i\phi_\theta(\sigma).
$$

Its real part controls the amplitude magnitude, and its imaginary part controls the phase. We use two small hidden layers with 16 units each and real parameters. `complex_output=True` supplies separate outputs for the log magnitude and phase.

```python
network = nqx.nn.MLP(
    hilbert.size,
    hidden_features=(16, 16),
    activation="tanh",
    complex_output=True,
    dtype=jnp.float64,
    param_dtype=jnp.float64,
    rngs=7,
)
```

The input width includes both edge and intertwiner labels. Gauge invariance comes from the chosen Hilbert space, so this network does not need an additional Gauss constraint projector. `rngs=7` seeds the initial network parameters. The [MLP guide](../concepts/networks/built-in/mlp.md) describes the architecture and its options.

## Create the variational state

[`VariationalState`](../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState) combines the network, its parameters, and the sampler. It manages sample batches and provides the expectation values and derivatives used by the driver.

```python
state = nqx.vqs.VariationalState(
    sampler,
    network,
    n_samples=256,
    n_discard_per_chain=16,
    sampler_seed=11,
)
```

Here, 256 is the total number of retained configurations across all eight chains, giving 32 per chain. Each sampling step attempts eight proposals because we set `sweep_size=8`. Before retaining a fresh batch, each chain discards 16 such steps to allow it to adjust to the current wavefunction. The sampler seed is separate from the network seed.

These are modest settings for a first run. More retained samples can improve statistics, while more discard steps and different proposal weights can help equilibration. They address different issues. The [variational state guide](../concepts/vqs/state.md) explains this lifecycle, and the [sampling diagnostics](../concepts/samplers/diagnostics.md) help assess a longer calculation.

## Construct the Thiemann objective

[`LqxSU2`](../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxSU2) supplies the fixed graph Euclidean Thiemann constraint and geometric observables for this Hilbert space. Its [`thiemann_quadratic_constraint`](../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxSU2.thiemann_quadratic_constraint) method uses the standard ordering when `alternative=False`.

```python
geometry = nqx.lqx.LqxSU2(hilbert)
constraint = geometry.thiemann_quadratic_constraint(alternative=False)
```

Writing $H_v^E$ for the Euclidean constraint at vertex $v$, the objective is

$$
\begin{aligned}
\widehat C&=\sum_v H_v^E(H_v^E)^\dagger,\\
\mathcal L(\theta)
&=\frac{\langle\psi_\theta|\widehat C|\psi_\theta\rangle}
{\langle\psi_\theta|\psi_\theta\rangle}
=\sum_v\frac{\lVert(H_v^E)^\dagger\psi_\theta\rVert^2}
{\lVert\psi_\theta\rVert^2}\ge0.
\end{aligned}
$$

The method returns a list of local quadratic terms, which the driver interprets as their sum. There are no cross terms between different vertices. The ordering matters because $H_v^E$ is not generally Hermitian. A zero objective in this ordering means that each $(H_v^E)^\dagger$ annihilates the state.

The calculation uses matrix free operator actions and [squared estimators](../concepts/operators/squared-estimators.md), without constructing a matrix over the full Hilbert space. We retain the model's default dimensionless flux scale. The [SU(2) model guide](../concepts/lqx/su2.md) gives the operator definitions and conventions. This example uses the implemented Euclidean component, with no Lorentzian contribution.

## Run the optimisation

[`VMC`](../api/public/neuralqx/driver/index.md#neuralqx.driver.VMC) coordinates sampling, differentiation, and parameter updates. We combine [Optax SGD](../concepts/optimisers/optax.md) with [`SR`](../api/public/neuralqx/optimizer/index.md#neuralqx.optimizer.SR), the stochastic reconfiguration preconditioner. SR uses the geometry of the variational wavefunction to turn the gradient into an update direction. SGD then applies that direction with the chosen learning rate.

```python
optimizer = optax.sgd(learning_rate=0.01)
sr = nqx.optimizer.SR(diag_shift=0.05)

driver = nqx.driver.VMC(
    constraint,
    optimizer,
    variational_state=state,
    preconditioner=sr,
    loss_name="Constraint",
)

driver.run(20, show_progress=True)
```

The positive `diag_shift` regularises the sampled SR metric. See [stochastic reconfiguration](../concepts/optimisers/sr.md) for its equations and [choosing an optimiser with SR](../concepts/optimisers/sr-optimisers.md) for other combinations.

The first iteration includes [JAX compilation](https://docs.jax.dev/en/latest/jit-compilation.html#jit-and-caching), so it can take longer than subsequent iterations. The progress display reports the sampled constraint objective. It can fluctuate as the parameters and sample batches change. Twenty updates illustrate the workflow rather than establish convergence. Calling `driver.run` again continues from the current parameters and optimiser state.

## Measure the resulting state

We can now measure the constraint and the area of a surface punctured by the first edge. Increasing `n_samples` invalidates the cached batch, so these measurements use fresh samples from the updated wavefunction.

```python
state.n_samples = 1024
area = geometry.get_area(surface=[graph.edges[0]])

measurements = driver.estimate({
    "Constraint": constraint,
    "Area": area,
})

print("Constraint:", measurements["Constraint"])
print("Area:", measurements["Area"])
print("Sampling exponent:", float(state.sampling_exponent))
```

The estimates include the importance corrections. The area is reported in the model's flux units, and the printed exponent is the value used for the next sampling batch. The [expectations guide](../concepts/vqs/expectations.md) explains the returned statistics and their uncertainties.

For a longer study, check sampling and optimisation convergence before increasing the spin cutoff or network size. The [driver monitoring guide](../concepts/drivers/monitoring.md) shows how to track the calculation, and [saving and loading states](../concepts/vqs/saving.md) explains how to preserve the wavefunction. The [core concepts overview](../concepts/index.md) connects these components to the broader neuraLQX workflow.
