---
title: Projectors
---

# Projectors

A projector constructs a restricted wavefunction from an underlying model. It can remove selected basis amplitudes or combine amplitudes into a symmetry sector. The resulting parameters still belong to the base network, while the projection specifies which variational family those parameters describe.

For a fixed linear projector $P$,

$$
|\phi_\theta\rangle=P|\psi_\theta\rangle,
\qquad
P^2=P.
$$

An orthogonal projector also satisfies $P^\dagger=P$. The network wrapper returns $\log\phi_\theta(\sigma)$. **Projection acts on amplitudes**, so it is generally incorrect to apply $P$ directly to the values of $\log\psi_\theta$.

neuraLQX supplies the [nonvacuum projector](#remove-the-vacuum-amplitude) and base classes for defining your own masks or coherent row expansions. It does not infer a symmetry group or a physical gauge projection from the name of a model.

## Remove the vacuum amplitude

[`NonVacuum`](../../api/public/neuralqx/nn/projectors/index.md#neuralqx.nn.projectors.NonVacuum) implements

$$
P_{\ne\Omega}=I-|\Omega\rangle\langle\Omega|.
$$

The default vacuum is the configuration with every coordinate equal to zero. In the usual U(1) encoding it is the zero-charge configuration. In the SU(2) spin network encoding it has zero edge spins and zero channels. Configurations with only some zero labels remain allowed.

The registered shortcut `nqx.nn.projectors.nonvacuum` can wrap an existing model, decorate a model class, or configure a projector with keyword arguments. This example decorates a constructor.

```python
import jax.numpy as jnp
import neuralqx as nqx

@nqx.nn.projectors.nonvacuum
class NonVacuumRBM(nqx.nn.RBM):
    pass

network = NonVacuumRBM(4, hidden_features=8, rngs=0)
configurations = jnp.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, -1, 0],
], dtype=jnp.int32)
values = network(configurations)
print(jnp.isneginf(values.real).tolist())
print(bool(jnp.exp(values[0]) == 0))
```

```text
[True, False, False]
True
```

For an existing instance, use `nqx.nn.projectors.nonvacuum(base_model)` or `NonVacuum()(base_model)`. Wrapping an initialised NNX model retains its parameter objects without reinitialising them. A decorated class instead becomes a constructor returning the corresponding wrapper.

`vacuum=0` can be replaced by another repeated scalar label or a vector of length `hilbert.size`, for example `NonVacuum(vacuum=(0, 1, 0, 1))`. This excludes exactly that basis configuration. It does not exclude an arbitrary vacuum superposition.

The projection is exact and needs no overlap penalty coefficient. It also changes the variational problem. Excluding a physical vacuum is appropriate only when the remaining sector is the one you intend to study.

## Support, normalisation, and sampling

Projection does not normalise the resulting state,

$$
\|\phi_\theta\|^2
=\langle\psi_\theta|P^\dagger P|\psi_\theta\rangle.
$$

That norm must be nonzero for a valid variational state. The sampler must target the projected log amplitudes. Sampling the original model and simply deleting forbidden samples does not provide the same simulation interface or establish equilibration on the remaining support.

Pass the wrapped model to [`VariationalState`](../../api/public/neuralqx/vqs/index.md#neuralqx.vqs.VariationalState). The current Metropolis implementation searches for finite support when a recognised projection leaves initial chains at zero amplitude. The search uses the configured proposal rule and raises if it cannot find support within its bound. It does not replace thermalisation or prove that the retained support is connected under those proposals.

A Hilbert constraint and a wavefunction projector are distinct. The constraint determines the configurations admitted by the Hilbert space. A projector can set some otherwise admissible amplitudes to zero or combine amplitudes coherently. It does not rewrite the sampler's move set or the operator's connections. See [sampling diagnostics](../samplers/diagnostics.md) when assessing the resulting chains.

The base model's direct autoregressive sampler is not inherited by a projected wrapper. Its original conditional probabilities generally describe a different distribution.

## Define a diagonal projector

[`DiagonalProjector`](../../api/public/neuralqx/nn/projectors/index.md#neuralqx.nn.projectors.DiagonalProjector) is the simplest extension. Implement a Boolean `keep(states)` with output shape `states.shape[:-1]`. If $m(\sigma)$ is that mask,

$$
(P\psi)(\sigma)=m(\sigma)\psi(\sigma).
$$

A zero-or-one mask is automatically Hermitian and idempotent. Excluded amplitudes are represented with real log amplitude `-inf`.

```python
import jax.numpy as jnp
import neuralqx as nqx
from neuralqx.nn.projectors import DiagonalProjector

class TotalChargeZero(DiagonalProjector, name="total_charge_zero"):
    def keep(self, states):
        return jnp.sum(states, axis=-1) == 0

base_model = nqx.nn.MLP(3, hidden_features=(8,), rngs=1)
network = nqx.nn.projectors.total_charge_zero(base_model)
configurations = jnp.array([[1, -1, 0], [1, 0, 0], [0, 0, 0]])
print(jnp.isfinite(network(configurations).real).tolist())
```

```text
[True, False, True]
```

This example imposes one total-label condition. It is not the vertex-by-vertex U(1) Gauss law, which depends on graph orientation and the chosen gauge convention. For a physical gauge invariant basis, prefer the [Hilbert space's own constraint construction](../hilbert/constraints.md) when it expresses the intended sector directly.

Concrete subclasses register automatically. `name=` chooses the shortcut, while the default is the class name converted to snake case. [`get_projector`](../../api/public/neuralqx/nn/projectors/index.md#neuralqx.nn.projectors.get_projector) constructs a registered projector by name, and [`available_projectors`](../../api/public/neuralqx/nn/projectors/index.md#neuralqx.nn.projectors.available_projectors) lists the current registry. Use `register=False` for a private class or an experimental definition that does not need a global shortcut.

Annotated fields define projector arguments. Array data is fixed model state rather than trainable parameters. Discrete construction options can use [`field`](../../api/public/neuralqx/nn/projectors/index.md#neuralqx.nn.projectors.field) with `static=True`. Replacing an existing registered name requires an explicit replacement request.

## Define a coherent symmetry projection

[`RowProjector`](../../api/public/neuralqx/nn/projectors/index.md#neuralqx.nn.projectors.RowProjector) represents a finite expansion of each projector row,

$$
(P\psi)(x)=\sum_{a=1}^{K}c_a(x)\psi(y_a(x)).
$$

`connections(states)` returns configurations of shape `(*batch, K, features)` and coefficients of shape `(*batch, K)`. Coefficients are matrix-row entries, with no implicit conjugation. The row width is static, and zero coefficients can pad unused entries.

For a unitary finite-group action, the invariant projector is $P=|G|^{-1}\sum_{g\in G}U_g$. The group action must be defined on the actual basis, including any label permutations, signs, or representation factors. The library does not establish those physical conventions for you.

A simple example is charge conjugation $C\psi(x)=\psi(-x)$ on a symmetric charge domain. Since $C^2=I$, the even and odd projections are

$$
P_\pm=\frac12(I\pm C).
$$

```python
import jax.numpy as jnp
import neuralqx as nqx
import numpy as np
from neuralqx.nn.projectors import RowProjector, field

class ChargeParity(RowProjector, register=False):
    parity: int = field(static=True, default=1)

    def __post_init__(self):
        super().__post_init__()
        if self.parity not in (-1, 1):
            raise ValueError("parity must be +1 or -1")

    def connections(self, states):
        points = jnp.stack((states, -states), axis=-2)
        coefficients = jnp.broadcast_to(
            jnp.array([0.5, 0.5 * self.parity]),
            states.shape[:-1] + (2,),
        )
        return points, coefficients

base_model = nqx.nn.MLP(
    2, hidden_features=(8,), complex_output=False,
    dtype=jnp.float64, param_dtype=jnp.float64, rngs=2
)
odd_model = ChargeParity(parity=-1)(base_model)
configurations = jnp.array([[1, 0], [-1, 1], [0, 0]])
amplitudes = jnp.exp(odd_model(configurations))
reflected = jnp.exp(odd_model(-configurations))

np.testing.assert_allclose(reflected, -amplitudes, atol=1e-12)
print(bool(jnp.isneginf(odd_model(configurations[-1]).real)))
```

```text
True
```

The odd sector vanishes at the all-zero configuration because the two contributions cancel. Negative or complex coefficients must be combined coherently, which can produce a complex result even from a real base model. Averaging the log amplitudes would produce a different wavefunction.

Use `evaluation_chunk_size` to bound base-network batches when a row expansion is large. It controls evaluation memory without changing the projection. Nonzero connections must be in the base model's domain.

## Mathematical responsibilities of a custom projector

For an arbitrary row expansion, you are responsible for $P^2=P$ and any claimed Hermiticity or symmetry sector. A general linear combination of transformations is not automatically a projector. Composing two projectors also need not give a projector unless their mathematical relationship permits it, for example when they commute.

The more general [`Projector`](../../api/public/neuralqx/nn/projectors/index.md#neuralqx.nn.projectors.Projector) base accepts a pure JAX `project(log_value, states)` implementation. It must preserve the batch axes and apply the intended operation to amplitudes. Prefer a diagonal mask or row expansion when either already expresses the construction.

Exact cancellation nodes receive `-inf+0j` and a zero log-derivative convention in the row implementation. A logarithmic derivative is mathematically undefined at a parameter-dependent node. A calculation that specifically needs the derivative of the amplitude at such a node requires an amplitude-level treatment.

## Keep operator action outside the support

If $\phi=P\psi$ has zero amplitude on some configurations, an operator $A$ can still map it into those configurations. The full squared objective is

$$
\frac{\|A\phi\|^2}{\|\phi\|^2}.
$$

Discarding the part of $A\phi$ outside the represented support would change this objective. Do not remove operator output connections merely because the projected wavefunction vanishes there.

The variational state recognises native projection wrappers and uses an output-sampling residual estimator for non-diagonal `Squared` objectives. Its default uses one auxiliary output draw per source. To control that additional variance, use [`FullResidual`](../../api/public/neuralqx/operator/index.md#neuralqx.operator.FullResidual) with an identity [`BasisEmbedding`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.BasisEmbedding) and an explicit `n_probes`. `n_probes=None` sums reachable outputs instead of drawing them. The [restricted spaces and residuals guide](../operators/restricted-spaces.md) explains the estimator and its distinction from source-state Monte Carlo error.

NNX, Linen, and explicit-function wrappers are supported. For Linen, initialise the wrapped module so its parameter scopes and fixed projection collection are present. An existing unwrapped variables dictionary is not automatically the variables dictionary of the wrapped module.
