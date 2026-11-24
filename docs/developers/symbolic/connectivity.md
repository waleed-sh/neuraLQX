---
title: Connectivity analysis
---

# Connectivity analysis

A symbolic definition describes contributions, while an estimator needs connected configurations and their accumulated matrix elements. These are different quantities. Ten diagonal rules have ten contributions but only one target for a given source. Conversely, two different update expressions can coincide only on particular configurations.

Connectivity analysis separates identities that can be established before execution from collisions that must be resolved at runtime. Its result is a [`ConnectivityDeduplicationPlan`](../../api/public/neuralqx/operator/symbolic/compiler/core/connectivity/index.md#neuralqx.operator.symbolic.compiler.core.connectivity.ConnectivityDeduplicationPlan), consumed by both the reference and planned JAX kernels.

## Raw contributions and target classes

For a term with $M_t$ iterator rows and $E_t$ emissions, its raw contribution count is $M_tE_t$. Without a manual capacity hint, the first analysis pass sums these counts. For a particular source, some contributions can be inactive or have zero amplitude, but the raw shape cannot depend on those runtime values.

The deduplication pass instantiates each emission with its static iterator bindings. It symbolically executes the ordered update program, representing unchanged coordinates as source reads and changed coordinates as expressions. Targets with identical representations belong to the same class.

If class $C$ contains several contributions to target $T_C$, its amplitude is

$$
b_C(\sigma)=\sum_{r\in C}p_r(\sigma)a_r(\sigma).
$$

Grouping a class therefore saves target construction and output slots while retaining all its contributing amplitudes. Statically inactive branches and literal zero contributions can be removed. Two unconditional, exactly opposite finite constants at the same target can also cancel. The analysis deliberately does not treat arbitrary floating-point reassociation as an exact cancellation proof.

## A diagonal example

This definition emits two half-contributions for each of two edge charges. Four raw branches reduce to one diagonal target, with coefficient $q_0^2+q_1^2$.

```python
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))],
    is_planar=True,
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
half_charge = 0.5 * s.site("e").value.pow(2)
definition = (
    s.SymbolicOperatorBuilder(hilbert, "two_halves", hermitian=True)
    .for_each_site("e")
    .emit(matrix_element=half_charge, tag="first_half")
    .emit(matrix_element=half_charge, tag="second_half")
    .build()
)

artifact = s.SymbolicCompiler().compile(definition)
plan = artifact.metadata_map()["connectivity_deduplication"]
assert plan["raw_size"] == 4
assert plan["target_class_count"] == 1
assert plan["output_size"] == 1

raw = definition.compile(deduplicate_connected_components=False)
assert raw.max_conn_size == 4
basis = np.asarray(hilbert.all_states())
expected = np.diag(np.sum(basis**2, axis=-1))
np.testing.assert_allclose(artifact.compiled_operator.to_dense(), expected)
np.testing.assert_allclose(raw.to_dense(), expected)
```

The dense matrix is only an independent check on this nine-state example. Neither compilation path needs that matrix to execute on a larger space.

## Prove disjointness where possible

Different target expressions need not collide. If one rule always changes coordinate 0 and leaves coordinate 1 unchanged, while another always changes coordinate 1 and leaves coordinate 0 unchanged, their outputs cannot be equal. The analysis first uses such changed-coordinate information to avoid unnecessary pair comparisons.

Further checks use affine integer bounds and bounded exhaustive evaluation over finite local domains. A proof considers the conditions under which both actions can be active. Unknown expressions, dynamic indices, unavailable domain information, and exhausted analysis budgets remain conservative. Failure to prove disjointness creates a runtime obligation rather than permission to discard a contribution.

These proofs use local domains. A global constraint can make fewer configurations physically available, but the compiler does not generally solve that constraint to obtain the tightest possible bound. This is why a capacity can be correct without being minimal.

## Resolve the remaining collision components

Imagine a graph whose vertices are target classes, with an edge whenever two classes might coincide. Each connected component can be processed independently. Coordinates that are identical across the component do not need to appear in its comparison key.

Where coordinate ranges are proven integer ranges, keys are packed into 32-bit words without losing information. This is an injective encoding, not a probabilistic hash. When that proof is unavailable, the implementation retains full coordinate keys.

For a component of $n$ targets and key width $w$, the current strategy is selected from bounded temporary-work estimates.

| Strategy | Current selection and execution |
| --- | --- |
| Direct-address workspace | At least 8 targets, at most 12 packed key bits, and a key domain no larger than $4n$. Accumulate amplitudes by exact key bucket |
| Pairwise comparison | Otherwise, $n^2\max(1,w)\leq16384$. Compare keys and accumulate each class into its first occurrence |
| Stable lexicographic sort | Larger components. Sort exact keys, identify equal-key runs, and scatter their sums back to first occurrences |

These thresholds are implementation choices rather than user tuning parameters. They bound assembly work and temporary storage. They do not change the mathematical equivalence relation between targets.

## Derive the padded capacity

For a collision component $C$, finite-image analysis can bound the number of distinct targets by $K_C\leq|C|$. If there are $S$ isolated target classes, the final static capacity is

$$
K=\max\left(1,\ S+\sum_C K_C\right).
$$

The lower bound of one preserves the native padded interface even for the zero operator. Runtime assembly sums colliding amplitudes and stably compacts nonzero entries when needed. Unused native output slots have zero matrix element and the source configuration as their target.

A user-supplied `.max_conn_size(...)` hint affects the raw analysis. It does not replace the proven deduplicated bound. An underestimated hint is unsafe in a raw execution path because output can be truncated. Prefer the inferred capacity, and treat a capacity change as something to verify through aggregated matrix elements rather than array shape alone.

## Respect numerical proof assumptions

The plan records integer ranges used in its proofs. At execution, the coalescer checks that the configuration dtype can represent those intermediate integers exactly. If it cannot, the implementation compares full targets conservatively instead of relying on unsupported disjointness or packed-key assumptions.

Floating-point amplitude reductions still follow numerical arithmetic. Exact equality of targets does not imply that every reassociation of their coefficients is bitwise identical. Regression tests should compare the aggregated action with an appropriate tolerance and include cancellation, low-precision, and boundary cases where those behaviours matter.

## Interpret assembly costs in context

`artifact.metadata_map()["connectivity_cost"]` records estimated assembly work, and `NQXDSL-W401` can report a substantial increase over raw assembly. This model excludes neural network evaluation and is not a hardware benchmark.

An expensive coalescing step may still reduce total estimator time by removing many repeated wavefunction evaluations. Compare the complete expectation or gradient workload, using the [profiling guides](../profiling/index.md), after establishing equivalent operator actions. The next page explains how [kernel planning](kernels.md) can reduce target construction before runtime coalescing begins.
