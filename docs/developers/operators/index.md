---
title: Custom operators
---

# Custom operators

A custom operator describes a new observable, constraint, or dynamical term on a neuraLQX Hilbert space. Its implementation must tell the package what the basis configurations mean, which configurations are connected, and what their matrix elements are. Once those pieces agree, the standard estimator machinery can evaluate it on a [variational state](../../concepts/vqs/index.md).

The operator infrastructure is matrix free. A computational operator calculates the required matrix rows from configurations without allocating the full Hilbert space matrix. Small local matrices, recoupling tables, and precomputed graph data can still be useful ingredients. Matrix free refers to avoiding the global matrix, whose dimensions grow with the complete Hilbert space.

If your expression can already be built from the [supplied operators](../../concepts/operators/prebuilt.md), use their algebra first. A new class becomes useful when the elementary action is missing, or when a fused implementation can evaluate an existing expression more directly.

## Choose the interface that matches the action

| Base class | What you implement | What it is useful for |
| --- | --- | --- |
| [`AbstractOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.AbstractOperator) | Hilbert space, matrix-element dtype, Hermiticity information, and an adjoint when needed | Observables with a separately registered direct estimator |
| [`DiscreteOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.DiscreteOperator) | The above, a connection capacity, and batched padded connectivity | A custom discrete representation with its own batching implementation |
| [`ComputationalOperator`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.ComputationalOperator) | The operator metadata and one JAX kernel for a single configuration | Most new matrix free operators, with batching, streaming, and algebra supplied by the base |

These interfaces build on [structs and pytrees](../structs/index.md). Declare array coefficients as numerical fields and fixed choices such as a site index as static fields. Construct topology and shape metadata before numerical evaluation. The generated constructor and `__post_init__` are usually sufficient, so a custom Python constructor is rarely needed.

## Define the matrix row, not just the ket action

For input configuration $x$, connectivity describes the entries

$$
A_{xy}=\langle x|A|y\rangle.
$$

Those entries produce

$$
(A\psi)(x)=\sum_y A_{xy}\psi(y),
\qquad
A_{\mathrm{loc}}(x)=\frac{(A\psi)(x)}{\psi(x)}.
$$

The input is the row label. This matters for an operator usually introduced through its ket action. If $a^\dagger|n\rangle=\sqrt{n+1}|n+1\rangle$, row $n$ of $a^\dagger$ connects to column $n-1$ with element $\sqrt n$. The [connectivity guide](connectivity.md) implements that example and checks its dense matrix.

For a diagonal observable, every nonzero connection returns to the input configuration. For a local off-diagonal observable, only a few labels change. A constraint built from several local terms may return the same target more than once, and those contributions must add.

## Decide how the action treats boundaries and constraints

A finite local cutoff is part of the operator's mathematical definition. A shift can vanish at a boundary, wrap cyclically, or act through a prescribed projection. Those choices describe different operators. State the chosen convention in the class documentation and test the extreme basis labels.

Likewise, an operator on a constrained space must respect that space or explicitly represent a restriction. Masking transitions outside a subspace implements a projected action. It can change products and constraint norms, so it is not an interchangeable substitute for a full-space action. The [restriction guide](../../concepts/operators/restricted-spaces.md) explains embeddings and residuals when contributions outside the variational subspace must be retained.

## From a kernel to a usable observable

[Operator contracts](contracts.md) develops a local matrix operator that works with arbitrary scalar local labels. [Connectivity](connectivity.md) explains padding, repeated targets, shapes, and a bounded creation operator. [Composition](composition.md) then connects custom operators to sums, products, adjoints, squared estimators, and variational measurements.

Each stage should have a small independent reference. Check that connectivity reproduces the intended dense matrix, that the adjoint is its conjugate transpose, and that composed expressions act in the correct order. For a Hamiltonian used in optimisation, also check an expectation and gradient against exact differentiation on a tiny space. The [custom estimator gradient guide](../estimators/gradients.md) explains what that comparison needs to establish.
