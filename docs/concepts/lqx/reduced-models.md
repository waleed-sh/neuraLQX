---
title: Reduced models
---

# Reduced models

The reduced models describe settings in which the degrees of freedom and Hamiltonian have already been specialised. Their state labels can be stored using a [U(1) Hilbert space](../hilbert/u1.md), but their physical interpretation and operator action are determined by the reduction. Sharing a numerical storage type does not make these models interchangeable with the [weak coupling models](u1.md).

Two constructions are supplied. [`LqxQR`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxQR) describes a single vertex quantum reduction with three nonnegative spin labels. [`LqxSphericalBojowaldSwiderski`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LqxSphericalBojowaldSwiderski), also exported as `LqxSphericalBoSw`, describes radial and transverse labels in spherical symmetry.

## Which reduction matches your calculation?

| Choice | Single vertex quantum reduction | Spherical symmetry |
| --- | --- | --- |
| Physical labels | Three reduced spin values | Radial edge labels $k$ and transverse labels $\mu$ |
| Graph representation | Three base edges representing the reduced directions | A half ladder with an ordered radial chain |
| Local values | Nonnegative, with one gauge component | Integer, with one gauge component |
| Shift at the cutoff | Amplitude vanishes beyond the boundary | Labels wrap cyclically |
| Supplied dynamics | Euclidean and Lorentzian constraints | The implemented Bojowald–Swiderski vertex constraint |
| Guide | [Single vertex quantum reduction](qr.md) | [Spherical symmetry](spherical.md) |

These cutoff conventions affect the finite operator. In particular, a QR shift cannot be replaced by a shared U(1) cyclic holonomy without changing its action at the boundary. Conversely, the spherical Hamiltonian uses cyclic shifts whose size is set by its regularisation parameter.

## The same model interface

Both models use the [standard LQX structure](structure.md). Factories construct operators, properties remember your selections, and the base class assembles the total and quadratic objectives. The available observables differ. The QR model supplies reduced flux powers and bounded shifts but no default area, volume, or mass. The spherical model supplies a local volume and radial triad observable, but no default area or mass.

The QR model's total includes its Euclidean and Lorentzian terms. In the spherical model, the implemented vertex Hamiltonian is registered through the `euclidean_constraint` factory and there is no separate supplied Lorentzian component. Read the model equations before interpreting these common interface names as a particular split of a continuum constraint.

## Reading the model equations

Each of the following pages introduces the state labels, constructs a compatible space, and gives the action of every operator exposed by the model. The formulas state the normalisation and cutoff used in the code, so they can also serve as a reference when comparing with an analytic calculation or a different regularisation.

Begin with [single vertex quantum reduction](qr.md) for the bounded three-label construction, or [spherical symmetry](spherical.md) for a radial chain and its neighbouring-vertex Hamiltonian. If you need a different reduced construction, [creating a model](custom-models.md) shows how to reuse the interface while making your own physical choices explicit.
