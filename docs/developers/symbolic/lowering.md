---
title: JAX lowering and execution
---

# JAX lowering and execution

The JAX lowerer turns a validated symbolic definition and its analyses into a numerical connection function. The resulting [`CompiledOperator`](../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.CompiledOperator) is a native computational operator. It participates in batching, streaming, algebra, expectations, and gradients through the existing neuraLQX interfaces.

This integration is what makes a symbolic definition useful beyond a small demonstration. Estimators receive the same kind of connected configurations and matrix elements that they receive from a directly implemented operator.

## The connection contract

For configurations with shape `(..., N)`, the public padded interface returns targets with shape `(..., K, N)` and matrix elements with shape `(..., K)`. Here $N$ is the configuration size and $K$ is the operator's static `max_conn_size`.

The final coordinate axis uses the Hilbert space representation. Matrix elements use the operator dtype, subject to JAX's enabled precision. Leading batch dimensions are preserved. The native wrapper replaces targets associated with exactly zero matrix elements by the input configuration, so padding remains a valid input for a wavefunction on a constrained space.

This replacement concerns inactive slots. It does not project active targets onto the Hilbert space. A nonzero connection outside the physical domain is still an operator-definition error.

## Reference and planned execution

The reference path builds a runner for each term. It binds iterator indices, evaluates guards, copies the source into a working target, applies the ordered update program, and evaluates the coefficient using source and completed-target values. Term outputs are assembled and passed through the connectivity plan when deduplication is enabled.

The planned path uses the [kernel plan](kernels.md) to share scalar expression templates and construct target representatives directly from coordinate maps. Both paths use the same matrix-row convention and static output bound. Disabling `optimize_kernels` selects reference execution without disabling deduplication.

Neither path builds a dense operator matrix during ordinary execution. A matrix appears only when the user deliberately requests an operation such as `.to_dense()` on a sufficiently small indexable space.

## Test batching, streaming, and JIT

This example builds a diagonal operator, checks two leading batch dimensions, and compares streamed execution with direct execution. It also passes the operator through a JAX transformation as a native pytree.

```python
import jax
import jax.numpy as jnp
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))],
    is_planar=True,
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
operator = (
    s.SymbolicOperatorBuilder(hilbert, "batched_charge", hermitian=True)
    .for_each_site("e")
    .emit(matrix_element=s.site("e").value.pow(2))
    .compile()
)
states = jnp.asarray(hilbert.all_states(), dtype=hilbert.dtype).reshape(3, 3, 2)
targets, matrix_elements = operator.get_conn_padded(states, use_streaming=False)
streamed_targets, streamed_elements = operator.get_conn_padded(
    states, use_streaming=True, chunk_size=4
)

assert targets.shape == (3, 3, 1, 2)
assert matrix_elements.shape == (3, 3, 1)
np.testing.assert_array_equal(targets, states[..., None, :])
np.testing.assert_allclose(matrix_elements[..., 0], np.sum(np.asarray(states)**2, axis=-1))
np.testing.assert_array_equal(streamed_targets, targets)
np.testing.assert_allclose(streamed_elements, matrix_elements)

@jax.jit
def evaluate(op, configurations):
    return op.get_conn_padded(configurations, use_streaming=False)

jitted_targets, jitted_elements = evaluate(operator, states)
jitted_elements.block_until_ready()
np.testing.assert_array_equal(jitted_targets, targets)
np.testing.assert_allclose(jitted_elements, matrix_elements)
```

Streaming partitions the sample batch into fixed-size chunks. It does not change the symbolic action count or divide one operator row into smaller connection blocks. See [batching and chunking](../../concepts/vqs/batching.md) for how operator work relates to the rest of a variational calculation.

## Static definitions and dynamic arrays

The compiled callable, capacity, name, dtype, Hermiticity flag, and update-support information are static struct fields. They describe the operation rather than a trainable array. Configurations are numerical inputs. Network parameters enter through the estimator and variational state, not through the symbolic builder's Python callbacks.

In particular, a named symbolic coefficient with a default is embedded in the definition. It is not a runtime argument automatically differentiated by the compiled operator. Reconstruct the definition when changing such a fixed coefficient, or implement an explicitly parameterised native operator if that is the required contract.

The [structs and pytrees guide](../structs/index.md) explains the underlying distinction. The [custom estimators guide](../estimators/index.md) explains how derivatives with respect to wavefunction parameters use operator connections.

## Conditional evaluation under transformations

The lowerer uses JAX conditionals to retain scalar guard regions where possible. This does not justify arbitrary undefined arithmetic inside inactive branches. JAX traces both branches, and applying `vmap` over predicates can transform a conditional into a selection. The [JAX conditional reference](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html) describes these transformation semantics.

When developing an optimisation, compare scalar, batched, and differentiated execution for expressions with restricted domains. A finite final value can coexist with a nonfinite inactive derivative. The planner's conservative regions and reference fallbacks are part of preserving those numerical behaviours.

## Adjoint construction is a separate operation

Compiling an operator does not generally produce its adjoint kernel at the same time. A Hermitian declaration permits the native operator to return itself. Otherwise, generic adjoint construction searches candidate predecessor configurations over the inferred local update supports, tests forward connections, and conjugates their accumulated matrix elements.

The current generic path limits the total local candidate count to 65,536. Dynamic destinations can force a large support and make that path impractical. Supplying an explicitly compiled reverse action with `.with_adjoint(...)` avoids that search, but the author must supply the mathematical adjoint. The [scientist algebra guide](../../concepts/operators/symbolic/algebra.md) demonstrates this with ladder operators.

## Measure the right stage

The symbolic pass reports measure Python-side compiler work. The first numerical call can additionally trace and compile JAX execution. Later calls with compatible shapes, dtypes, and static inputs can reuse that device program. JAX's [compilation and caching guide](https://docs.jax.dev/en/latest/jit-compilation.html#jit-and-caching) explains that reuse.

Warm up the numerical path before measuring steady-state execution, and synchronise an output with `.block_until_ready()` at the timing boundary. When comparing plans, keep the source configurations, precision, and estimator workload fixed. Report symbolic compilation, first execution, and repeated execution separately when each is relevant.

Use the [profiling guides](../profiling/index.md) for traces of a complete calculation. A connection kernel that is faster in isolation need not make an expectation faster if it changes the number of wavefunction evaluations or the memory pressure downstream.
