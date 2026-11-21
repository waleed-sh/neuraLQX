---
title: Accuracy and convergence
---

# Accuracy and convergence

A reliable volume calculation needs two numerical checks. The shifted systems must be solved accurately, and the finite quadrature must approximate the desired matrix function accurately. These are separate questions. A small solve residual does not establish that the quadrature order and spacing are sufficient.

This page uses the notation from [the SRQ derivation](srq.md). The dimensionless density is $Q$, its scale bound is $B$, and $A=Q^2/B^2$. The volume action is $Vb=c_V\sqrt B\,A^{1/4}b$. Keep the graph, incident spins, units, and input vector fixed while checking its numerical evaluation. Changing the physical spin cutoff or the number of Monte Carlo samples answers a different convergence question.

## What the shifted residuals measure

For shift $t_\ell$, let $\widetilde y_\ell$ be the computed solution of

$$
(A+t_\ell I)y_\ell=Ab.
$$

The diagnostic recomputes the true residual of that system,

$$
r_\ell=Ab-(A+t_\ell I)\widetilde y_\ell,
\qquad
\rho_\ell=\frac{\|r_\ell\|_2}{\|Ab\|_2}.
$$

It does not rely only on the convergence estimate carried by the CG recurrence. For a zero right-hand side, the implementation uses a denominator of one, and the zero solution has zero residual.

The connection to action error follows from

$$
\widetilde y_\ell-y_\ell
=-(A+t_\ell I)^{-1}r_\ell.
$$

Since $A$ is positive semidefinite and $t_\ell>0$,

$$
\left\|
\widetilde V_{K,h}b-V_{K,h}b
\right\|_2
\leq
c_V\sqrt B
\sum_{\ell=-K}^{K}
\frac{\omega_\ell}{t_\ell}\|r_\ell\|_2.
$$

Here $V_{K,h}$ denotes the finite quadrature with exact solves. The bound concerns the error introduced by the solves, not the difference between that quadrature and the true volume. It can also be very pessimistic for tiny shifts. A setting such as `solve_rtol=1e-12` is therefore not a promise of relative volume accuracy $10^{-12}$.

### Inspect a connected row

[`VolumeOperator.row_with_diagnostics`](../../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.VolumeOperator) returns connected configurations, matrix elements, and residuals for one input configuration. Select `method="srq"` explicitly to obtain the shifted residual array.

```python
import jax.numpy as jnp
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(
    graph, two_j_max=4, random_init="vacuum"
)
model = nqx.lqx.LqxSU2(hilbert)
volume = model.get_volume(
    graph.vertices[0],
    method="srq",
    srq_order=128,
    srq_step=0.75,
    solve_rtol=1e-12,
    solve_maxiter=128,
)

configuration = hilbert.complete_channels_for_edges([4] * graph.n_edges)
assert configuration is not None
connected, elements, residuals = volume.row_with_diagnostics(
    jnp.asarray(configuration)
)

print(residuals.shape)
print(bool(jnp.all(jnp.isfinite(elements))))
print(bool(jnp.max(residuals) < 1e-9))
```

```text
(257,)
True
True
```

There are $2\times128+1$ residuals. Inspect their maximum when deciding whether the solves are sufficiently converged. The final comparison above checks the implementation's usual failure threshold for this tolerance. It does not establish an error target for the volume.

With `method="spectral"` the residual array is empty. **The `method="auto"` row interface also returns an empty residual array, even when its internal policy selects SRQ.** An empty array from automatic evaluation therefore says nothing about which branch was used or its accuracy. Construct an explicit SRQ operator for this diagnostic.

Runtime SRQ rows reject solves whose largest relative residual exceeds $\max(100\,\texttt{solve\_rtol},10^{-9})$, or whose residual is nonfinite, by producing nonfinite amplitudes. A failed SRQ solve during Hamiltonian table construction raises an error instead. These checks catch numerical failures. They do not replace a convergence study. In particular, the public [`VertexVolumeAction.apply`](../../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.VertexVolumeAction) returns its residuals for you to check and does not apply this rejection threshold itself.

## Compare with a fixed-sector dense reference

Dense diagonalisation is useful when the selected local intertwiner space is small. You do not need to enumerate the full Hilbert space or build spectral volume tables for every spin assignment.

The following example uses [`fixed_sector_action`](../../../api/public/neuralqx/operator/prebuilt/su2/index.md#neuralqx.operator.prebuilt.su2.VolumeOperator.fixed_sector_action) to compare the two methods for the same graph, incident spins, and volume scale. The coefficient vector is expressed in `reference.channels` order. It is a vector in the local intertwiner space, not a complete spin network configuration.

```python
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(
    graph, two_j_max=4, random_init="vacuum"
)
volume = nqx.lqx.LqxSU2(hilbert).get_volume(graph.vertices[0])
configuration = hilbert.complete_channels_for_edges([4] * graph.n_edges)
assert configuration is not None

reference = volume.fixed_sector_action(configuration, method="spectral")
vector = jnp.arange(1, len(reference.channels) + 1, dtype=jnp.float64)
vector = vector / jnp.linalg.norm(vector)
dense_action, _ = reference.apply(vector)
reference_norm = float(jnp.linalg.norm(dense_action))
assert reference_norm > 0

print("order  step  relative action error  largest solve residual")
for order, step in (
    (32, 0.75),
    (64, 0.75),
    (96, 0.75),
    (128, 0.75),
    (192, 0.50),
    (256, 0.50),
):
    action = volume.fixed_sector_action(
        configuration,
        method="srq",
        order=order,
        step=step,
        rtol=1e-12,
        maxiter=128,
    )
    assert action.channels == reference.channels
    approximate_action, residuals = action.apply(vector)
    absolute_error = float(jnp.linalg.norm(approximate_action - dense_action))
    relative_error = absolute_error / reference_norm
    largest_residual = float(jnp.max(residuals))
    assert np.isfinite(relative_error) and np.isfinite(largest_residual)
    print(f"{order:5d}  {step:.2f}  {relative_error:21.3e}  {largest_residual:.3e}")
```

The dense result is a floating-point reference. In a 64-bit CPU check, the first three quadrature orders gave relative action errors of approximately $3\times10^{-3}$, $8\times10^{-6}$, and $2\times10^{-8}$. Their largest solve residuals were already below $10^{-12}$. The quadrature was still changing while the shifted solves were well converged.

For a nonzero reference action, the reported quantity is

$$
\epsilon_{\mathrm{rel}}(b)
=\frac{\|\widetilde Vb-V_{\mathrm{dense}}b\|_2}
{\|V_{\mathrm{dense}}b\|_2}.
$$

Also retain the absolute numerator. A relative error is undefined for an exact zero action and can be misleading for a nearly vanishing one. Use an absolute error target in the chosen volume units in those cases.

One vector in one sector is a demonstration, not a certification of every volume row. Repeat the comparison across relevant spin assignments, vertices, and input vectors. Include the larger local spaces encountered by the simulation and sectors with small nonzero volume eigenvalues. Where feasible, testing a complete local basis checks all columns of that local operator.

## Refine the interval and grid separately

First hold $h$ fixed and increase $K$. This extends the integration interval and tests the effect of the omitted tails. In the example, the sequence from $(K,h)=(32,0.75)$ to $(128,0.75)$ does precisely this.

Next decrease $h$ while increasing $K$ enough to preserve the interval. The change from $(128,0.75)$ to $(192,0.50)$ keeps $Kh=96$, so it primarily checks the logarithmic grid spacing. Finally, extending to $(256,0.50)$ checks the tails again at the finer spacing. Error need not decrease at every step because discretisation, truncation, and roundoff contribute together.

Refine the solve independently by increasing `maxiter` and tightening `rtol`, then inspect the true residuals and the resulting action. Longer quadrature intervals include smaller shifts, which can make the shifted systems more difficult. Increasing `order` alone does not guarantee that the existing iteration limit is sufficient. Conversely, tightening `rtol` cannot repair a quadrature interval that is too short.

When a local spectrum is available, the quadrature error has a particularly direct interpretation. Define

$$
R_{K,h}(x)
=\sum_{\ell=-K}^{K}\omega_\ell\frac{x}{x+t_\ell},
\qquad
x_r=\frac{\lambda_r(Q)^2}{B^2}.
$$

With exact shifted solves, the local operator norm error is

$$
\|V_{K,h}-V\|_2
=c_V\sqrt B\,
\max_r\left|R_{K,h}(x_r)-x_r^{1/4}\right|.
$$

This separates the scalar quadrature approximation from the linear solver. The [scalar example](srq.md#see-the-approximation-on-individual-eigenvalues) illustrates that rational function. The formula is not a quantity automatically returned by the solve diagnostics.

If dense diagonalisation is too expensive, compare several independent refinements of the action and the observables of interest. Agreement between refinements is useful evidence of convergence, but it is not a certified bound on the entire spectrum. There is no single `volume_error` argument that supplies such a guarantee.

The volume kernels use 64-bit arithmetic under the package's default precision configuration. Retain that precision for tight comparisons. Once refinement reaches a roundoff floor, smaller tolerances or more shifts may not improve the answer. The [dense zero-eigenvalue convention](index.md#dense-spectral-evaluation) also matters when comparing near a kernel.

## Separate numerical error from sampling error

For a fixed normalised wavefunction, a volume approximation changes the expectation by

$$
\delta\langle V\rangle
=\langle\psi|(\widetilde V-V)|\psi\rangle,
\qquad
|\delta\langle V\rangle|
\leq\|\widetilde V-V\|_2.
$$

This is a deterministic change in the observable. Increasing the sample count reduces statistical uncertainty but does not remove this numerical bias.

When comparing two volume settings in a [`VariationalState`](../../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState), hold its parameters fixed and reuse the same cached sample batch. Consecutive `state.expect(volume_a)` and `state.expect(volume_b)` calls reuse that batch if you do not resample or invalidate it between calls. This makes their difference easier to interpret. You can also compare the paired [local estimators](../../operators/application.md#from-the-expectation-value-to-the-local-estimator) on those configurations.

Choose the numerical tolerance in relation to the precision needed for the observable. For a Monte Carlo measurement, the change under refinement should be comfortably below the statistical uncertainty and any physical accuracy target. A short, noisy run can hide a numerical bias, so perform the deterministic sector comparisons as well. [Sampling diagnostics](../../samplers/diagnostics.md) address the separate question of whether the retained configurations adequately represent the wavefunction.

## The volume inside a Hamiltonian

The Euclidean constraint uses volume actions inside a holonomy commutator. Its intermediate spaces include gauge covariant total-spin-$\tfrac12$ sectors, and intermediate edge spins can reach $j_{\max}+\tfrac12$. A test of the standalone gauge invariant vertex volume does not cover all of those blocks.

Set the shared policy on [`LqxSU2`](../../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxSU2) when you want a refinement to affect both observables and the volumes used by subsequently constructed constraints.

```python
import neuralqx as nqx

graph = nqx.graph.K5Graph()
hilbert = nqx.hilbert.SU2Hilbert(
    graph, two_j_max=1, random_init="vacuum"
)
model = nqx.lqx.LqxSU2(
    hilbert,
    volume_method="srq",
    srq_order=192,
    srq_step=0.50,
    srq_solve_rtol=1e-12,
    srq_maxiter=256,
    hamiltonian_backend="sector",
)

volume = model.get_volume(graph.vertices[0])
constraint = model.thiemann_regularized_constraint(graph.vertices[0])
print(volume.method)
print(volume.srq_order)
```

```text
srq
192
```

These are example refinement settings, not a universal accuracy prescription. Construct the corresponding constraint for each policy and compare its action or expectation with fixed wavefunction parameters and, for sampled comparisons, the same configurations. If the optimisation uses a [quadratic constraint](../structure.md#quadratic-constraints), check that objective as well. Its sensitivity need not match the standalone volume's relative error.

An override such as `model.get_volume(vertex, srq_order=192)` changes only that returned volume. Assigning a different observable to `model.volume` does not reconfigure the Hamiltonian. Use the shared model settings when changing its internal volume evaluation, and reconstruct the operators you want to compare.

Keep numerical volume convergence, physical cutoff convergence, and sampling convergence as distinct checks. Together they establish whether a reported observable is stable under the approximations used in the calculation.
