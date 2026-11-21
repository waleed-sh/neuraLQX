---
title: Detailed balance and proposal corrections
---

# Detailed balance and proposal corrections

A proposal rule defines how a candidate is generated. The Metropolis acceptance step corrects the resulting bias so that the intended distribution is stationary. This page derives that correction and gives a finite-state verification that can be adapted to a new rule.

Write the target as $\pi(x)=\exp[w(x)]/Z$, where `sampler.log_probability` returns $w(x)$. For ordinary sampling, $w(x)=m\operatorname{Re}\log\psi(x)$. The proposal probability $q(y\mid x)$ includes all choices made while generating $y$ from $x$, including site selection, channel selection, and any normalisation over available moves.

## Acceptance and detailed balance

For distinct configurations, the transition probability is

$$
\begin{aligned}
P(y\mid x)&=q(y\mid x)\,a(x,y),\\
a(x,y)&=\min\left(1,
\frac{\pi(y)q(x\mid y)}{\pi(x)q(y\mid x)}\right).
\end{aligned}
$$

The probability of remaining at $x$ includes both direct self proposals and rejected moves. For a pair with positive forward and reverse probability,

$$
\begin{aligned}
&\pi(x)P(y\mid x)\\
&\quad=\min\!\left(\pi(x)q(y\mid x),\pi(y)q(x\mid y)\right)\\
&\quad=\pi(y)P(x\mid y).
\end{aligned}
$$

Summing this equality over $x$ establishes stationarity. In logarithmic form the acceptance argument is

$$
\begin{aligned}
\Delta(x,y)&=w(y)-w(x)+c(x,y),\\
c(x,y)&=\log q(x\mid y)-\log q(y\mid x).
\end{aligned}
$$

The [transition rule](transition-rules.md) returns only $c(x,y)$. A positive correction favours a move whose reverse proposal is more likely than its forward proposal. Reversing this sign generally gives the wrong stationary distribution.

The implementation compares a uniform variate with $\exp(\Delta)$. Since the variate lies below one, this implements the same acceptance probability as explicitly taking the minimum with one for finite weights. A proposal with no reverse path has zero acceptance from a positive-weight state under this construction. If the target assigns zero weight to a candidate, that candidate must not enter the retained target support.

## Verifying a complete transition matrix

For the independent binary rule, $q(y\mid x)=q(y)$. With two sites, all four states can be enumerated, so detailed balance can be checked without sampling error. Rows in the matrix below are current states and columns are proposed destinations.

```python
import itertools
import numpy as np

basis = np.asarray(list(itertools.product((0, 1), repeat=2)))
theta = np.asarray([0.2, -0.3])
proposal_probabilities = np.asarray([0.8, 0.3])

log_target = 2 * (basis @ theta)
target = np.exp(log_target - log_target.max())
target /= target.sum()
log_q = np.sum(
    basis * np.log(proposal_probabilities)
    + (1 - basis) * np.log1p(-proposal_probabilities),
    axis=-1,
)
q = np.exp(log_q)

correction = log_q[:, None] - log_q[None, :]
log_ratio = log_target[None, :] - log_target[:, None] + correction
acceptance = np.exp(np.minimum(0.0, log_ratio))
transition = q[None, :] * acceptance
transition[np.diag_indices_from(transition)] += 1 - transition.sum(axis=1)

flux = target[:, None] * transition
np.testing.assert_allclose(transition.sum(axis=1), 1.0)
np.testing.assert_allclose(flux, flux.T, atol=1e-14)
np.testing.assert_allclose(target @ transition, target, atol=1e-14)
assert np.all(transition > 0)
print("Detailed balance and stationarity verified for all four states")
```

```text
Detailed balance and stationarity verified for all four states
```

This checks the proposal's mathematical definition. Also test that the implemented rule generates that proposal, computes the same correction, and preserves its state contract. For a small stochastic rule, empirical proposal frequencies with fixed inputs can expose an implementation that disagrees with the analytical $q$ even when the acceptance derivation is correct.

## State-dependent numbers of moves

Suppose $M(x)$ is a set of allowed neighbours and a rule chooses uniformly from it. If $y\in M(x)$ and $x\in M(y)$, then

$$
\begin{aligned}
q(y\mid x)&=\frac{1}{|M(x)|},\\
c(x,y)&=\log|M(x)|-\log|M(y)|.
\end{aligned}
$$

Using `None` merely because the underlying adjacency is undirected would omit this degree correction. Uniform selection among neighbours is symmetric only when the relevant degrees agree.

If several elementary choices lead to the same final configuration, add their probabilities when defining the marginal $q(y\mid x)$. A rule that instead treats the chosen move as an auxiliary variable needs a consistent reverse move and an acceptance derivation on that augmented space. Do not mix a marginal forward probability with a single-path reverse probability.

## Conditioning a proposal on validity

Let $q_0$ be a symmetric proposal and $\mathcal C$ the valid set. Drawing once and replacing an invalid candidate with $x$ preserves the off-diagonal symmetry within $\mathcal C$. The rejected probability is added to the diagonal.

By contrast, repeating the draw until it is valid produces

$$
\begin{aligned}
q_{\mathcal C}(y\mid x)
&=\frac{q_0(y\mid x)}{A(x)},\\
A(x)&=\sum_{z\in\mathcal C}q_0(z\mid x).
\end{aligned}
$$

Its correction is $\log A(x)-\log A(y)$, even though $q_0$ was symmetric. If $A(x)$ is difficult to compute, a constraint-preserving move or explicit rejection is often easier to establish correctly. The [custom constraint](../hilbert/constraints.md) and [proposal-move](../hilbert/proposals.md) guides describe how these choices interact with the Hilbert space.

## Mixtures, adaptation, and equilibration

A fixed, state-independent mixture of individually valid Metropolis kernels preserves the target. Choose a component with a fixed probability and apply that component's own acceptance rule. This differs from first combining proposal probabilities and then applying a single marginal Metropolis correction, although both constructions can be valid.

If component weights depend on $x$, the fixed-mixture argument no longer applies. Derive the full forward and reverse probabilities, or include the component choice in a justified auxiliary-variable construction. Similarly, tuning probabilities from the chain history produces an adaptive process. A counter in rule state does not by itself supply a convergence argument. A straightforward approach is to adapt during a separate warmup stage and freeze the proposal before collecting measurements.

Finally, detailed balance is not a mixing diagnostic. A reversible rule can remain in a conserved sector, and a deterministic flip at uniform target can be periodic. Check communicating classes and self-transition probabilities on small examples, then study [autocorrelations and effective sample size](../../concepts/samplers/diagnostics.md) at the intended simulation scale.
