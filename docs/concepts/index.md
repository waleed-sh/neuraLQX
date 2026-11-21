---
title: Core concepts
---

# Core concepts

Working with neuraLQX means making two connected sets of choices. You specify a quantum system, including its graph, admissible states, and operators. You then choose how to represent a wavefunction on that system, sample it, and adjust its parameters. The core APIs give each of these choices a place in your calculation.

This guide introduces the relationship between those components. You do not need to understand their internal implementation before using them. What matters first is knowing which object describes the physics, which object holds the state you are studying, and which objects perform the numerical work.

## From a graph to a quantum state

In canonical loop quantum gravity, a graph supports the degrees of freedom used to describe quantum geometry. In a gauge invariant SU(2) spin network basis, edges carry representation labels, or spins, and vertices carry intertwiner labels that specify how incident representations couple. These ingredients belong to the kinematical description of the theory. The [review by Ashtekar and Lewandowski](https://arxiv.org/abs/gr-qc/0404018) gives a broader account of this construction and its relation to geometric operators and constraints.

The same distinction appears in neuraLQX. A [graph](graphs/index.md) records vertices, directed edges, their connectivity, and an embedding used for orientation conventions. A [Hilbert space](hilbert/index.md) determines the quantum labels that can live on that graph, their allowed values, and any restrictions on admissible configurations. Creating a graph alone does not assign spins, impose gauge invariance, or define a quantum state.

For a numerical calculation, you also choose a finite description. In an [SU(2) space](hilbert/su2.md), this includes an edge-spin cutoff and the corresponding admissible intertwiner labels. In a [U(1) space](hilbert/u1.md), the labels and their constraints follow the chosen Abelian model. The [state layout](hilbert/state-layouts.md) specifies how these labels are arranged in the arrays passed to networks, samplers, and operators.

It is useful to call one complete admissible assignment of these labels a *configuration*, denoted by $x$. A configuration identifies a basis state $|x\rangle$. A quantum state is a superposition of such basis states, with amplitudes that still need to be specified.

## Specify the theory and its operators

An [LQX model](lqx/index.md) connects a Hilbert space to the geometric observables and constraint operators of a particular formulation. For example, an SU(2) model can construct an area operator on a chosen set of edges. The operator describes how a quantity acts on states. Its expectation value also depends on the wavefunction in which you measure it.

The [operator API](operators/index.md) is the common interface for these actions. You can work with operators supplied by an LQX model, combine existing operators, or define an operator appropriate to your research. An operator may leave a configuration unchanged and multiply its amplitude, or connect it to other configurations in the Hilbert space. This is why the graph, state layout, and operator must describe the same system.

Keep the role of constraints explicit. A restriction built into a Hilbert space controls which configurations are admitted. A constraint included in an optimisation objective controls what the variational calculation tries to minimise. For instance, using a gauge invariant basis addresses the Gauss constraint at the level of the state space. It does not, by itself, solve the dynamics or the remaining constraints of the theory. The [Hilbert space constraints](hilbert/constraints.md) and [LQX model guides](lqx/structure.md) develop these choices separately.

## Represent and sample the wavefunction

A neural quantum state assigns a parametrised amplitude $\psi_\theta(x)$ to each configuration. The parameters $\theta$ are the trainable quantities in a [neural network](networks/index.md). neuraLQX models used by the variational-state interface return the log amplitude, $\log\psi_\theta(x)$, which can include both a modulus and a phase.

For a discrete orthonormal basis, the normalised state can be written as

$$
|\Psi_\theta\rangle = \frac{1}{\sqrt{Z_\theta}}
\sum_x \psi_\theta(x)|x\rangle,
\qquad
Z_\theta = \sum_x |\psi_\theta(x)|^2.
$$

The network gives you a way to evaluate amplitudes without storing a separate coefficient for every basis state. In a large Hilbert space, summing over all configurations is still impractical. A [sampler](samplers/index.md) therefore draws configurations from a target distribution, ordinarily

$$
p_\theta(x) = \frac{|\psi_\theta(x)|^2}{Z_\theta}.
$$

The proposal rule needs to suit the Hilbert space. In a constrained space, changing a single array entry arbitrarily can produce an invalid configuration. The [sampling guides](samplers/transition-rules.md) explain how proposals respect the structure of the state space and how to assess the resulting samples.

A [variational quantum state](vqs/index.md), usually shortened to VQS, brings together the network, its parameters, and the sampler. It manages samples and provides operations such as estimating [expectation values and gradients](vqs/expectations.md). The network defines the amplitude function, while the variational state supplies the interface used to carry out calculations with it.

Here is a small example that puts these pieces together and estimates an area in an initial, unoptimised state:

```python
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(graph, two_j_max=1)
geometry = nqx.lqx.LqxSU2(hilbert)
area = geometry.get_area(surface=[0])

model = nqx.nn.MLP(hilbert.size, hidden_features=(16, 16), rngs=0)
sampler = nqx.sampler.MetropolisSU2(hilbert, n_chains=4, sweep_size=8)
state = nqx.vqs.VariationalState(
    sampler=sampler,
    model=model,
    n_samples=64,
    n_discard_per_chain=16,
    seed=1,
)

estimate = state.expect(area)
print(estimate.mean)
```

The graph has five vertices and ten edges. The cutoff permits edge spins $0$ and $\tfrac{1}{2}$, and the Hilbert space includes the admissible intertwiner data. `hilbert.size` gives the number of configuration entries the network receives, rather than the dimension of the Hilbert space. The surface in this example crosses edge `0` once away from a vertex, with area expressed in the default units of $8\pi |\gamma|\ell_{\mathrm{P}}^2$.

The small network and sample count keep the example easy to run. The printed value is a Monte Carlo estimate for the initial parameters. It is not an optimised result, and the example does not establish that the chains have converged. A research calculation also needs [sampling diagnostics](samplers/diagnostics.md), uncertainty estimates, and checks that its numerical choices are adequate.

## Turn measurements into a variational calculation

To optimise the state, you choose an objective. This might be the expectation value of a Hamiltonian, or a suitable positive quadratic constraint when searching for states with a small constraint residual. The choice is part of the physics problem. Minimising an arbitrary operator is not automatically equivalent to solving the constraints you intended to study.

An [optimiser](optimisers/index.md) turns information about the objective into parameter updates. You can use [Optax](optimisers/optax.md) to choose the update rule and, where appropriate, [stochastic reconfiguration](optimisers/sr.md) to account for the geometry of the variational family. A [VMC driver](drivers/index.md) coordinates the repeated sampling, gradient estimation, and updates.

[Callbacks](callbacks/index.md) let you observe or control that run, for example by recording quantities or applying a stopping criterion. [Checkpointing](drivers/checkpointing.md) preserves the state of a calculation so it can be continued. During optimisation, the graph and Hilbert space normally stay fixed while the network parameters and the sampled configurations change.

## How to use the following sections

The sections in this guide follow the objects you will encounter in a simulation. This table is a map from a scientific or numerical choice to the API responsible for it.

| Your choice | Where it belongs |
| --- | --- |
| Vertices, directed edges, connectivity, and embedding | [Graphs](graphs/index.md) |
| Quantum labels, cutoffs, admissibility, and array layout | [Hilbert spaces](hilbert/index.md) |
| A particular LQG formulation and its supplied observables | [LQX models](lqx/index.md) |
| Actions on configurations and algebra between observables | [Operators](operators/index.md) |
| Proposals, Monte Carlo chains, and sample quality | [Samplers](samplers/index.md) |
| Parameter updates and stochastic reconfiguration | [Optimisers](optimisers/index.md) |
| The optimisation loop and its saved progress | [VMC drivers](drivers/index.md) |
| A wavefunction with parameters, samples, and estimators | [Variational states](vqs/index.md) |
| The parametrised log-amplitude function | [Neural networks](networks/index.md) |
| Monitoring and controlling a running calculation | [Callbacks](callbacks/index.md) |

Start with [graphs](graphs/index.md) and [Hilbert spaces](hilbert/index.md) to establish the representation of your system. Then follow the components needed for your observable or optimisation problem. The [first simulation tutorial](../tutorials/notebooks/first-simulation.ipynb) brings these choices together in a complete calculation, while [distributed simulations](../distributed/jax-parallelism.md) covers how the numerical work is placed on CPUs and GPUs.
