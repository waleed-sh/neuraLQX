---
title: Custom samplers
---

# Custom samplers

A sampler connects a variational wavefunction to the configurations used to estimate observables and gradients. Extending it is therefore a statistical task as well as a programming task. A transition can preserve every array shape, compile successfully, and still sample the wrong distribution. These guides explain both the interface and the conditions that make an implementation suitable for a simulation.

The [scientist sampling guide](../../concepts/samplers/index.md) introduces the supplied algorithms and their settings. Here the starting point is a new proposal, a new sampling algorithm, or additional state that the existing implementation does not provide.

## Choosing the extension point

For a new way of proposing configurations, implement an [`AbstractTransitionRule`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.AbstractTransitionRule) and pass it to [`MetropolisSampler`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.MetropolisSampler). The sampler already manages acceptance, random keys, repeated sweeps, and chain output. The rule supplies a candidate configuration and, when needed, the logarithmic correction for an asymmetric proposal. There is no registration decorator for this interface. Construction with `rule=your_rule` selects the implementation.

Use [`FunctionalRule`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.FunctionalRule) when those operations are conveniently expressed as functions. A rule class is useful when a family of proposals has validated settings or reusable helpers. Both approaches use the same transition contract.

Implement [`AbstractSampler`](../../api/public/neuralqx/sampler/index.md#neuralqx.sampler.AbstractSampler) only when the algorithm itself changes. Independent draws from an enumerated distribution, an autoregressive sampling procedure, or a method with a different state evolution can require this level of control. Such an implementation also takes responsibility for initialisation, reset behaviour, sample ordering, and the lifetime of its random state.

## The distribution being sampled

For a log wavefunction $\ell_\theta(x)=\log\psi_\theta(x)$, the ordinary target is

$$
\begin{aligned}
\pi_\theta(x)
&=\frac{\exp\!\left(m\,\operatorname{Re}\ell_\theta(x)\right)}{Z_\theta},\\
Z_\theta&=\sum_{x\in\mathcal H}
\exp\!\left(m\,\operatorname{Re}\ell_\theta(x)\right).
\end{aligned}
$$

The usual Born distribution uses `machine_pow=2`, so its weight is $|\psi_\theta(x)|^2$. The sampler evaluates the supplied callable through `log_probability`. This returns the unnormalised log weight, without $\log Z_\theta$. The normalisation cancels in a Metropolis ratio.

Keep that callable as the source of the target. A [variational state](../../concepts/vqs/state.md) can provide a sampling function adapted to an [importance measure](../../concepts/samplers/importance.md). Reaching through it to a particular neural network would bypass that choice. Likewise, the argument conventionally named `parameters` may contain a framework variable tree rather than only trainable weights.

## Validity, reachability, and efficiency

The configuration space is determined by the Hilbert space, including its local domains and constraints. The Metropolis core does not perform a general `hilbert.is_valid` check on every candidate from a custom rule. A rule must produce valid configurations or deliberately turn an invalid proposal into a self transition. The [Hilbert proposal guide](../hilbert/proposals.md) explains the related domain-level interface.

Validity alone is insufficient. For example, exchanging two labels preserves their sum. This is useful for a fixed-charge sector, but it cannot explore a space containing several charge sectors. Establish which configurations communicate under the proposal and whether that is the intended physical space. For a finite chain, irreducibility on the target support and aperiodicity provide the usual convergence conditions. Detailed balance establishes stationarity, but does not establish either of those properties by itself.

Only after these checks does acceptance rate become a useful performance diagnostic. A high rate can accompany tiny changes or frequent self transitions, with strongly correlated measurements. Compare observable autocorrelations and effective sample size using the [sampling diagnostics](../../concepts/samplers/diagnostics.md), rather than optimising acceptance in isolation.

## How the guides fit together

The [sampler contract](contracts.md) describes the full interface and implements a small independent sampler. The [transition-rule guide](transition-rules.md) constructs an asymmetric binary proposal that works with the existing Metropolis implementation. [Detailed balance and proposal corrections](detailed-balance.md) derives its acceptance factor and verifies the complete transition matrix on a finite space.

The final page covers [sampler state](state.md), including random keys, parameter changes, rule state, and the additional restrictions imposed by distributed execution. Read it before retaining caches or adding adaptation to a proposal, since the acceptance decision does not automatically select or reject the rule's returned state.
