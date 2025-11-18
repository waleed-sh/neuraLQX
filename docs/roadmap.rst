=========
Roadmap
=========

This page sketches where neuraLQX is today and where it is headed. This is *not* a binding promise or release plan. Instead,
think of it as a high-level map: what problems we are trying to solve for loop quantum gravity (LQG), why the current design
looks the way it does, and which directions we expect the project to grow to.

For the motivations and design principles behind these choices, see the `philosophy <./philosophy.html>`_ page.

-------------------
Where we are today
-------------------

The current version of neuraLQX focuses on the Abelian :math:`U(1)^N` models of LQG. This is the "simple playground" where
we can prototype ideas relevant to the full :math:`SU(2)` LQG while leveraging NetKet's mature infrastructure. At the moment,
neuraLQX provides:

- A NetKet-based implementation of :math:`U(1)^N` models for LQG-like systems


- Two complementary ways of defining operators, and thus defining models, which are:

    - local operators, built on NetKet's :code:`LocalOperator` machinery, where operators are expressed as sums of local operators

    - computational operators, built on the more abstract NetKet :code:`DiscreteOperator` interface, where an operator is defined by *how it acts* on a basis state, rather than by storing an explicit matrix


- Support for **fixed graphs only**. All constraints and operators are graph-non-enlarging: the underlying graph is held fixed and edges carry :math:`U(1)^N` degrees of freedom.


- Full reliance on NetLet's existing strengths:

    - Hilbert space representation and sampling

    - Variational states

    - Optimisers, drivers, etc.


You can think of the current neuraLQX as a LQG specific layer on top of NetKet. It speaks the language of LQG, but is 
still structurally constrained by NetKet's abstractions.

--------------------------------
Why start with :math:`U(1)^N`?
--------------------------------

Working with :math:`U(1)^N` as a start is a delibrate first step. It keeps group theory and representation theory manageable,
making it easier to focus on a first iteration of how to organise LQG Hilbert spaces in code, how to represent constraints as
operators, how to interface those structures with neural network based variational methods. It also provides a clean testbed for
operator representations (matrix-based local operators vs. functional computational operators). It allows us to learn where NetKet's
abstractions are a perfect fit, and where LQG-specific needs start pushing us beyond those abstractions.


-------------------
The road to come
-------------------

The lessons learned from this Abelian playground directly inform the design of the next stages of neuraLQX. In what follows, 
we outline some of the things we already are working on, and what we are working towards.


Short-term roadmap
-------------------

The short-term focus is what we are working on right now, and that is to stabalise and deepen the :math:`U(1)^N` implementation
while keeping the public-facing API as clean and future-proof as possible. For this, the existing models which neuraLQX
ships with can be implemented as computational (based on :code:`ComputationalOperator` s, or local (based on :code:`LocalOperator` s).
This is because we know that in the :math:`SU(2)` theory, there will be operators which cannot be implemented as local operators,
and to keep the interface clean, for any defined model we, by default, offer the two implementations when available.


Medium-term roadmap
--------------------

The next major qualitative step is to move towards the full :math:`SU(2)` theory. This requires us to introduce a custom Hilbert API
that is no longer limited by NetKet's current abstractions. Unlike the Abelian theory, in the :math:`SU(2)` theory the edges carry
:math:`SU(2)` representation labels and the vertices are computationally non-trivial. They host :math:`SU(2)`-invariant tensors
that impose local gauge invariant. Operators such as the Hamiltonian constraint and volume operators probe this richer structure.

To support this, neuraLQX will introduce its own Hilbert space API which is designed from the ground up for such models, and
supports "backward compatibility" to "simpler" models (where vertices are computationally trivial) such as the existing
Abelian model. Of course, this requires that the new API be fully compatible with NetKet, so that as long as you stay within
the supported feature set, you can still use NetKet's optimisers, drivers, and so on, without changing your neuraLQX-level code.

The guiding principle is that the **public neuraLQX interface should remain as stable as possible, even as the internal implementation shifts from Netket-centric to LQG-specific**. In
this phase, NetKet becomes one backend among others, rather than *the* structural core. The goal is to keep what works
brilliantly in NetKet while allowing neuraLQX to grow the LQG-specific machinery that does not fit naturally into NetKet's
abstractions.

Long-term roadmap
------------------

The longer-term vision is to move beyond fixed graphs and tackle graph changing constraints, essential for the full :math:`SU(2)`
theory. This requires a conceptual shift:

- Today, sampling is over configurations on a fixed graph. Constraints can *remove* edges by setting their degrees of freedom to zero, but computationally, the graph remains fixed
- In the future, we want to sample over (graph, configuration) pairs, subject to LQG constraints that may create/remove edges/loops.

The key ingredients to this path will include many things. For example, we already are working on extending our Graph API
to include graph *mutators* and graph *validators*. Mutators will perform primitive operations which change the graph, such as
adding edges, removing edges, graph-refinements, attaching loops, etc. Meanwhile, Validators are rules that check whether a graph
is admissible. For example, if one wants to impose a "cutoff" on how many loops a constraint can add at any given vertex, this
can be achieved via a Validator. Therefore, they can check local and global admissibility of the graph.

Of course, once we can mutate a graph, we need to generate a kinematical Hilbert space on it. Therefore, the Hilbert API
will see yet another rewriting which allows it to be more dynamic, enabling us to quickly generate spaces on-the-fly, but
remain friendly to our underlying machinery (aka JAX).

The vision then is once completed, the last step is to tie the sampling and the VMC process together to accommodate for this. Sampling
is then done in extended configuration space where states are now pairs of (graph, labels) and constraints/operators may act by
modifying only the labels on a fixed graph (the current graph non-enlarging implementation) or changing the graph itself. In this regime,
neuraLQX will will need samplers that understand proposal moves for both labels and graphs and a VMC driver that puts this to work. This
also means that operators will need to expose the ability to also *act on the graph* and not only on a given basis element.

----------------------------
Relation to NetKet
----------------------------

Throughout this roadmap, we expect the relationship to NetKet to evolve. NetKet provides an **amazing** toolbox for
quantum systems. However, our demands on the long run will start to ask for more than what NetKet is designed to provide. As
such

- **Right now:** neuraLQX is tightly built on top of NetKet. NetKet's Hilbert spaces, operators and variational machinery are the backbone of neuraLQX. We simply use it to introduce LQG specific strucutre on top

- **Next:** we want to provide all APIs and LQG operators in a neuraLQX native manner which *still remains compatible with NetKet* but can handle, natively, the demands of LQG. For example, future neuraLQX Hilbert spaces should remain compatible with NetKet.

- **Later:** as graph dynamics are implemented, the core of neuraLQX will be largely independent. We expect to use NetKet where it fits, but neuraLQX abstractions will be driven primarily by LQG requirements rather than by the limits of existing libraries.


The user-facing philosophy is to protect your investment in neuraLQX-level code: the goal is that scripts written against high-level neuraLQX API should continue to work and require only minor updates as the backend develops.


----------------------------
How to contribute
----------------------------

This roadmap is **intentionally** high-level. Details, priorities, and timelines will evolve as we learn from experiments, user feedback, and the broader LQG community. If you are
interested in contributing, testing ideas or aligning research projects with neuraLQX, please visit the `contributing <./contribute.html>`_ page.