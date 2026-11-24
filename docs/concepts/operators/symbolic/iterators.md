---
title: Iterators and graph structure
---

# Iterators and graph structure

An iterator specifies the locations at which a symbolic term is evaluated. Its rows are fixed when the operator is constructed. The values stored at those locations, and the conditions determining whether a row contributes, are evaluated later for each configuration.

For a domain $D_t$ of site tuples, a term has the form

$$
A_t=\sum_{r\in D_t} A_{t,r}.
$$

An iterator row supplies the label bindings for one $A_{t,r}$. It does not itself create an interaction or an update. Those come from the [emissions](emissions.md) attached to the term.

## Available iteration domains

The methods below belong to [`SymbolicOperatorBuilder`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicDiscreteJaxOperator). Here $N=\texttt{hilbert.size}$ and $M$ is the number of supplied rows.

| Iterator | Visits | Typical use |
| --- | --- | --- |
| [`globally()`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicDiscreteJaxOperator.globally) | One visit with no labels | A complete update specified by fixed indices |
| [`for_each_site("i")`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicDiscreteJaxOperator.for_each_site) | $N$ single indices | Local diagonal costs or single-coordinate changes |
| [`for_each_pair("i", "j")`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicDiscreteJaxOperator.for_each_pair) | $N^2$ ordered pairs, including $i=j$ | All-pairs expressions that include self terms |
| [`for_each_distinct_pair("i", "j")`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicDiscreteJaxOperator.for_each_distinct_pair) | $N(N-1)$ ordered pairs | Directed pair moves without self pairs |
| [`for_each(labels, over=rows)`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicDiscreteJaxOperator.for_each) | $M$ explicit tuples | Selected edges, neighbouring coordinates, or loops |
| [`for_each_triplet(..., over=rows)`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicDiscreteJaxOperator.for_each_triplet) | $M$ three-tuples | Oriented triples supplied by a model |
| [`for_each_plaquette(..., over=rows)`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.SymbolicDiscreteJaxOperator.for_each_plaquette) | $M$ four-tuples | Four-edge loops or other four-coordinate actions |

The triplet and plaquette methods require explicit rows. Their names do not make the compiler search a graph for triangles or plaquettes. The general `for_each` supports any fixed tuple length, including one.

## Use the Hilbert layout's indices

In neuraLQX, graph vertices and configuration sites are not interchangeable. For a scalar U(1) edge layout, the sites label edges. For a vector layout, one edge contributes several coordinates. An SU(2) spin network also stores intertwiner channels. Passing a list of graph endpoint pairs directly to an iterator is therefore usually incorrect.

The following diagonal interaction couples consecutive edge coordinates along a two-edge path. Its single tuple `(0, 1)` refers to those coordinates, not the path's three vertices.

```python
import jax.numpy as jnp
import numpy as np
import neuralqx as nqx
from neuralqx.operator import symbolic as s

graph = nqx.graph.Graph(
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0))],
    is_planar=True,
)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
pairs = [(0, 1)]

interaction = (
    s.SymbolicOperatorBuilder(hilbert, "edge_correlation", hermitian=True)
    .for_each(("left", "right"), over=pairs)
    .emit(
        matrix_element=s.site("left").value * s.site("right").value
    )
    .compile()
)
_, coefficients = interaction.get_conn_padded(
    jnp.array([1, -1], dtype=hilbert.dtype)
)
np.testing.assert_allclose(coefficients, [-1])
```

The row list is consumed during construction. Each row must contain one valid integer index for every label. Changing the geometry or layout requires constructing the corresponding new domain. For physical edge identifiers, use the space's [edge-to-site mapping](spaces.md), then form the tuples from those indices.

## Ordered pairs and repeated rows

Distinct pairs are still ordered. Both `(i, j)` and `(j, i)` occur. For a symmetric diagonal pair interaction $V(q_i,q_j)=V(q_j,q_i)$,

$$
\sum_{i\ne j}V(q_i,q_j)
=2\sum_{i<j}V(q_i,q_j).
$$

Choose explicit unordered pairs, include a factor of one half, or restrict the domain with `site("i").index < site("j").index`. A directed hopping operator can instead need both orientations. Decide from the mathematical sum rather than from the word “distinct”.

Repeated rows are retained. Supplying `[(0, 1), (0, 1)]` adds the same interaction twice. Deduplication combines their target configurations by adding coefficients, which preserves that factor of two. It does not remove a repeated physical contribution as though the row list were a set.

## Global terms

A global term has one visit and no bound site names. Use static indices in its updates and [`source_index`](../../../api/public/neuralqx/operator/symbolic/index.md#neuralqx.operator.symbolic.source_index) in its coefficient.

```python
exchange = (
    s.SymbolicOperatorBuilder(hilbert, "exchange", hermitian=True)
    .globally()
    .emit(s.swap(0, 1), matrix_element=1)
    .compile()
)
targets, coefficients = exchange.get_conn_padded(
    jnp.array([1, -1], dtype=hilbert.dtype)
)
np.testing.assert_array_equal(targets, [[-1, 1]])
np.testing.assert_allclose(coefficients, [1])
```

This term exchanges two coordinates once. A site iterator with the same fixed-index update would repeat that exchange $N$ times, multiplying its matrix element by $N$.

## Terms are added, not nested

Each new iterator call finishes the current term. Consecutive `.for_each_site(...)` calls do not create nested Python loops. To bind two labels simultaneously, use a pair or explicit tuple iterator. To add another contribution, start another iterator after the first term has an emission.

With $M$ visits and $E$ emissions per visit, the raw candidate count is $ME$. Conditions and repeated targets can reduce the number of physical connections, and the compiler can often prove a smaller output capacity. The [emission guide](emissions.md#repeated-targets-and-output-capacity) explains how that affects padded output. A large all-pairs domain still describes all of its visits, so use a sparse graph-derived domain when the physics only couples selected coordinates.
