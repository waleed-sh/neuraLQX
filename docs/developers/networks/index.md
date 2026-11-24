---
title: Custom networks
---

# Custom networks

A custom network defines the variational family through its log amplitudes. The scientist guide to [custom models](../../concepts/networks/custom-models.md) shows how to express that function with Flax NNX, Flax Linen, or a JAX callable. These developer guides address what is required to make a reusable model behave correctly under sampling, differentiation, chunking, and framework adaptation.

The essential mathematical object remains

$$
\ell_\theta(x)=\log\psi_\theta(x).
$$

The network returns one scalar, real or complex, for each configuration. Samplers use its real part to construct probabilities. Operators evaluate it on connected configurations to form amplitude ratios. Gradient estimators differentiate it with respect to the trainable parameter tree. A model extension must give all three consumers a consistent function.

## Choosing the level of extension

An ordinary NNX or Linen module, or an explicit apply function, is sufficient for a new architecture. There is no model-registration requirement for passing one to [`VariationalState`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState). Use the [built-in layers](../../concepts/networks/layers.md) and [activation functions](../../concepts/networks/layers.md#select-an-activation) where their conventions suit the model, and implement additional operations with JAX.

[`LogWavefunction`](../../api/public/neuralqx/nn/index.md#neuralqx.nn.LogWavefunction) is the NNX base used by the supplied wavefunction models. A subclass can implement only `__call__`. It also offers an optional interface for [connected-state evaluation](connections.md), allowing an operator's target configurations to reuse intermediate work from their source configurations without changing the represented wavefunction.

A new [`AbstractModelFramework`](../../api/public/neuralqx/utils/frameworks/index.md#neuralqx.utils.frameworks.AbstractModelFramework) is appropriate when an existing model system needs a different arrangement of variables or a different apply interface. It is usually unnecessary for a new neural layer or architecture. The [framework guide](parameters.md) explains the variable contract and provides a small complete adapter.

## What must remain invariant

The value assigned to a configuration must not change when it appears in a larger batch, in a different chunk, or next to different configurations. Model parameters must not change during evaluation. Any fixed model state needed to define the wavefunction must be carried explicitly so that the same variables reproduce the same values.

These requirements are stronger than merely returning an array of the right shape. Batch-dependent normalisation, active dropout, or hidden random features would make a Metropolis ratio compare evaluations of different functions. They would also invalidate the assumption that an operator's connected-state evaluation uses the same wavefunction as its source evaluation.

Complex output introduces a separate distinction. Returning a complex log amplitude does not establish holomorphic dependence on complex parameters. The [complex-model guide](../../concepts/networks/complex.md) explains the available differentiation choices. The developer [model contract](contracts.md) connects those choices to implementation and verification.

## Developing and testing an extension

Begin with ordinary forward evaluation and a small space whose configuration layout is explicit. Verify shapes, batch independence, parameter gradients, and round trips through the chosen framework adapter. Only then introduce optimisations such as connected-state reuse.

For an optimised implementation, compare both values and derivatives against the ordinary forward path. A cache can preserve forward values while dropping parameter dependence from the gradient. This is especially important for [squared and non-Hermitian estimators](../estimators/gradients.md), where differentiation can pass through the connected wavefunction evaluations themselves.

The pages that follow separate these responsibilities. [Model contracts](contracts.md) covers the numerical function, input domain, and projection behaviour. [Parameters and frameworks](parameters.md) covers ownership, fixed state, and adapter construction. [Connected-state evaluation](connections.md) develops an exact sparse update and tests its gradients and fallback behaviour.

## Tensor partitioning

Reusable architectures can supply [tensor partition rules](tensor-rules.md) that describe their contraction dimensions without replacing the model's global JAX function. The [custom tensor model guide](../../distributed/tensor-parallelism/custom-models.md) demonstrates supplied layers, explicit layouts, normalization, and fixed-input gradient checks.
