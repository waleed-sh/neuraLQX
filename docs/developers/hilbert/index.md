---
title: Custom Hilbert spaces
---

# Custom Hilbert spaces

A Hilbert space defines the basis configurations on which the rest of a simulation operates. It establishes what each stored label means, which combinations of labels are allowed, and how those configurations can be generated and indexed. Operators use that representation to construct connected states, samplers use it to initialise and propose configurations, and the variational model assigns an amplitude to each configuration it receives.

These guides are for developers introducing a new local domain, a different graph layout, a constrained sector, or a genuinely new space implementation. If you only need to select a supplied U(1) or SU(2) construction, begin with the [scientist guide to Hilbert spaces](../../concepts/hilbert/index.md). Extending a supplied generic space is usually sufficient when the new physics fits its representation.

## Local domains, a layout, and a constraint

For $N$ scalar degrees of freedom with finite local domains $D_i$, the unconstrained configurations form

$$
\mathcal X=\prod_{i=0}^{N-1}D_i,
\qquad
\mathcal H=\operatorname{span}\{|x\rangle:x\in\mathcal X\}.
$$

If all $D_i$ are identical, the space is homogeneous. If they differ in their labels or local dimensions, the space is heterogeneous. A constraint $C(x)$ selects

$$
\mathcal X_C=\{x\in\mathcal X:C(x)=\mathrm{True}\},
\qquad
\mathcal H_C=\operatorname{span}\{|x\rangle:x\in\mathcal X_C\}.
$$

Homogeneity and constraints are separate decisions. Either a homogeneous or heterogeneous space can be constrained. The number of stored labels also need not equal the number of independent physical degrees of freedom. A constrained representation may retain dependent labels because operators and networks need the complete configuration.

In a graph-based model, the layout assigns those scalar labels to edges, vertices, and their components. An edge can carry several charges, while a vertex can carry a channel label. A component count is not a local Hilbert space dimension. For example, three independent components with $d$ possible values each have $d^3$ combinations, while occupying three scalar positions in the state array.

## Choose the smallest extension that expresses the space

[`HomogeneousDiscreteHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HomogeneousDiscreteHilbert) attaches the same scalar or vector domain to all enabled entities. [`HeterogeneousDiscreteHilbert`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.HeterogeneousDiscreteHilbert) supports different edge and vertex defaults, per-entity overrides, and mixed component widths. A custom constraint can be attached to either.

Subclass these implementations when their representation is suitable but a specialised initialiser or another operation needs to change. Implement [`DiscreteHilbertSpace`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.DiscreteHilbertSpace) directly when the graph-backed representation itself does not fit. The [construction guide](construction.md) demonstrates both routes, including a complete graph-free binary register.

[`AbstractHilbertSpace`](../../api/public/neuralqx/hilbert/index.md#neuralqx.hilbert.AbstractHilbertSpace) is a smaller foundational interface. Implementing only its finiteness and dimension properties does not make a new representation compatible with the discrete operator and sampler stack. The remaining contracts must match the consumers you intend to use.

## Keep the responsibilities distinct

| Responsibility | Question it answers |
| --- | --- |
| Local domains and conversions | Which labels belong at each site, and which local indices represent them? |
| Layout | Where is each physical degree of freedom stored? |
| Constraint | Does a locally admissible configuration belong to the chosen sector? |
| Random generation | How can a valid initial configuration be drawn? |
| Proposal move | How can a current configuration be changed while preserving the required domain? |
| Indexing | Can configurations be placed in a stable, enumerable basis order? |

A validity predicate does not automatically provide an efficient initialiser or an ergodic proposal. Knowing the dimension does not automatically provide constrained basis indexing. These distinctions become especially important for gauge invariant spaces, where local changes can violate relations between several labels.

## A practical development order

Start by [defining the local domains and space](construction.md), then make the [state representation](states.md) explicit. Implement and test [constraints](constraints.md) independently of sampling. Next supply [random state generation](random.md) and choose [proposal moves](proposals.md) that can explore the intended sector. Add [indexing and enumeration](indexing.md) when exact checks or array representations require them.

The [U(1)](../../concepts/hilbert/u1.md) and [SU(2)](../../concepts/hilbert/su2.md) guides show how these responsibilities are combined in the supplied physical spaces. Their specialised reconstruction and proposal algorithms are not automatically inherited by a generic user-defined constraint. A new physical construction needs its own argument for validity, sampling support, and consistency with its operators.
