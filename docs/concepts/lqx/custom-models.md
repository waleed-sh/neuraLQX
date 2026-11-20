---
title: Creating a model
---

# Creating a model

A custom LQX model is useful when a physical construction has become more than a few lines in a script. Perhaps several studies share the same constraint, or several observables depend on the same couplings. A subclass gives those choices a consistent interface and lets the standard model machinery assemble the total and quadratic constraints.

You do not need to write a new operator kernel to create a model. This page builds models entirely from existing operators. Implementing a new numerical action is a separate task covered by the [custom operator section](../../developers/operators/index.md).

## A minimal subclass

Subclass [`LQXModel`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LQXModel), call its constructor, and implement the factories your model provides. The inherited properties handle selection and caching. Start with a factory that can be called without arguments if you want the corresponding property to have a default.

The following example packages a charge penalty on a single component [U(1) space](../hilbert/u1.md),

$$
H = \lambda \sum_e \hat q_e^2.
$$

It is a simple example of the interface, not an LQG Hamiltonian constraint. We put it in the Euclidean component to demonstrate how the standard total is assembled. The mathematical operator is built from [`NumberOperator`](../../api/public/neuralqx/operator/prebuilt/wcl/index.md#neuralqx.operator.prebuilt.wcl.NumberOperator), operator products, and a scalar coefficient.

```python
import neuralqx as nqx
import numpy as np
from neuralqx.operator.prebuilt import wcl


class ChargePenaltyModel(nqx.lqx.LQXModel):
    def __init__(self, hilbert, *, coupling=1.0):
        super().__init__(hilbert)
        if hilbert.gauge_dimensions != 1:
            raise ValueError("This model needs one U(1) component.")
        self.coupling = float(coupling)

    def get_euclidean_constraint(self):
        total = wcl.IdentityOperator(self.hilbert, const=0)
        for edge in self.graph.edges:
            charge = wcl.NumberOperator(self.hilbert, edge=edge)
            total = total + self.coupling * (charge @ charge)
        return total


a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
model = ChargePenaltyModel(hilbert, coupling=0.5)

states = np.asarray(hilbert.all_states())
expected = 0.5 * np.sum(states**2, axis=-1)
np.testing.assert_allclose(model.constraint.to_dense(), np.diag(expected))
print(model.constraint is model.constraint)
```

```text
True
```

The example verifies the full matrix against the expected diagonal on a small space. It also demonstrates that repeated property access returns the selected object. The subclass does not need to reimplement `constraint`, `quadratic_constraint`, or their setters.

An unimplemented Lorentzian component is inherited from the base class and omitted from the automatic total. If you override an optional factory and need to declare it unavailable for some configuration, use `self._not_implemented(name, detail)`. This is how the base class distinguishes an intentionally absent component from a failure inside an implemented one. Ordinary errors in a factory should remain visible.

## Preserving local constraints

An ordinary sum is sufficient when the global operator is your intended object. If your calculation imposes a separate condition at each vertex, preserve that information with [`sum_constraint_terms`](../../api/public/neuralqx/lqx/index.md#neuralqx.lqx.LQXModel.sum_constraint_terms).

For illustration, group the previous charge penalty by the source vertex of each oriented edge,

$$
H_v = \lambda\sum_{e\,\text{outgoing from }v}\hat q_e^2.
$$

Every edge is counted once in the total. More importantly, the model can now construct $\sum_v H_vH_v^\dagger$ rather than the square of the global sum. This grouping is a modelling choice. For a physical constraint, use the actual local decomposition from its definition.

```python
import neuralqx as nqx
import numpy as np
from neuralqx.operator.prebuilt import wcl


class LocalChargePenaltyModel(nqx.lqx.LQXModel):
    def __init__(self, hilbert, *, coupling=1.0):
        super().__init__(hilbert)
        if hilbert.gauge_dimensions != 1:
            raise ValueError("This model needs one U(1) component.")
        self.coupling = float(coupling)

    def get_euclidean_constraint(self):
        terms = {}
        for vertex in self.graph.vertices:
            local = wcl.IdentityOperator(self.hilbert, const=0)
            for edge in self.graph.connectivity[vertex]["outgoing"]:
                charge = wcl.NumberOperator(self.hilbert, edge=edge)
                local = local + self.coupling * (charge @ charge)
            terms[vertex] = local
        return self.sum_constraint_terms(terms)


a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
graph = nqx.graph.Graph([(a, b), (b, c), (c, a)], is_planar=True)
hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=1)
model = LocalChargePenaltyModel(hilbert)

local_terms = model.get_constraint_terms()
matrices = [np.asarray(term.to_dense()) for term in local_terms.values()]
local_quadratic = sum(matrix @ matrix.conj().T for matrix in matrices)
states = np.asarray(hilbert.all_states())

# This triangle has one outgoing edge at each vertex.
np.testing.assert_allclose(local_quadratic, np.diag(np.sum(states**4, axis=-1)))
objective = model.quadratic_constraint
print(len(local_terms))
```

```text
3
```

Use the canonical objects from `self.graph.vertices` as the dictionary keys. If you implement both Euclidean and Lorentzian factories, use the same keys in both. The base class combines contributions with matching labels before constructing each local quadratic term. It does not pair terms by their position in a list.

The explicit matrices above are only a check on this small example. The selected `objective` retains its factorisation for [expectation estimation](../vqs/expectations.md). It does not require you to build `local_quadratic` in a simulation.

## Adding observables and parameters

A geometric factory follows the same pattern. Implement `get_volume(vertex)`, for example, and return an operator on `self.hilbert`. Keeping the vertex explicit is useful when there is no preferred region. A caller can construct several local volumes or select one through `model.volume = model.get_volume(vertex)`.

For a parameter that defines the whole model, a constructor argument is often the clearest choice. For a parameter that varies between individual operator constructions, a factory argument is convenient. In either case, explain the units, default, and scope. A lapse used by one factory does not automatically multiply other components.

When a constructor parameter changes, create a new model or explicitly rebuild and assign affected operators. Merely changing `model.coupling` in these examples would leave an already cached constraint unchanged. The [selection rules](structure.md#what-changes-when-you-make-a-new-selection) explain which assignments invalidate dependent properties.

## Checking a scientific construction

Before using a new model in a large calculation, compare its operators with equations on a small Hilbert space. Check diagonal values, signs, cutoff behaviour, and the intended adjoint. If local conditions matter, inspect `get_constraint_terms()` and verify that the quadratic objective uses the desired grouping and ordering.

The [supplied model guides](index.md#choosing-a-supplied-model) provide examples of these decisions in physical constructions. The [operator API](../../api/public/neuralqx/operator/index.md) is the reference for the algebra available when composing your own model.
