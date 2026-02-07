.. _philosophy:

===========================
Our Philosophy
===========================

neuraLQX is built on a clear and ambitious vision: to lead the way toward **variational loop quantum gravity (vLQG)**,
a computational and conceptual framework in which the fundamental structures of LQG can be studied, approximated,
and optimised using modern variational and machine learning techniques.

Our guiding philosophy is to **bridge the gap** between the highly formal world of canonical LQG and the practical
tooling of variational quantum simulation. The idea is not to re-implement NetKet for a niche use case, but to
provide a *layer of physical meaning* on top of it, one that speaks the language of graphs, holonomies,
constraints, and gauge invariance.

The result is a toolkit that is simultaneously theoretical and practical, rigorous enough to respect the mathematical
structure of LQG, but accessible enough to support exploration and experimentation in real numerical settings.

----------------------------
Why neuraLQX exists
----------------------------

At its heart, neuraLQX aims to answer a central question:

*Can we represent and optimise candidate states of quantum geometry directly, using neural-network-based variational
approaches, without giving up the core structure of LQG?*

To do this, neuraLQX brings together three worlds:

1. **Loop Quantum Gravity**, where states are defined on graphs with edges labelled by group representations and vertices
   by intertwiners.
2. **Variational Quantum Simulation**, where trial states are parameterised by neural networks or other flexible ansätze.
3. **High-performance numerical backends**, like JAX and NetKet, which provide automatic differentiation, Monte Carlo
   sampling, and stochastic optimisation.

In practice, this means we treat the LQG Hilbert space as a *variational manifold*, a space of representable states
whose parameters are tuned to minimise constraint violations and capture low-energy or diffeomorphism-invariant
structures.

----------------------------
Design principles
----------------------------

From its inception, neuraLQX is guided by several simple but strict principles:

**1. Physics-first, abstractions-second.**
   All abstractions, from graphs to solvers, exist to express physics cleanly. The framework is designed so that LQG
   objects (graphs, holonomies, constraints, operators) are the *native* building blocks.

**2. Compatibility over duplication.**  
   NetKet has an **amazing** suite of capabilities. We reuse as much of NetKet's mature infrastructure as possible, but
   extend it only where LQG’s needs go beyond it. This means users can mix NetKet-native components (optimisers, samplers)
   with neuraLQX ones without friction.

**3. Incremental realism.**  
   We start from the Abelian :math:`U(1)^N` sector, the "Abelian playground", not because it is simple, but because it
   allows for fast iteration. Every API choice here is made with a clear path to the full :math:`SU(2)` case in mind.

**4. Transparency and reproducibility.**  
   Every object (graph, operator, solver) is inspectable, serialisable, and meant to be logged. A simulation should be
   fully reconstructible from its logs.

**5. Scalability and modularity.**  
   The codebase is structured so that the same high-level abstractions, graphs, Hilbert spaces, solvers, can be reused
   for different levels of theory, from toy models to near-full LQG simulations.


-------------------------------
What "variational LQG" means
-------------------------------

The term *variational loop quantum gravity* refers to a new computational viewpoint:
treating the physical Hilbert space of LQG as the target of a **variational optimisation**.

Instead of working with fixed basis states or explicit spin network sums, neuraLQX allows you to:

* define a graph and Hilbert space consistent with a chosen truncation or cutoff,
* express the quantum constraints (Hamiltonian, Gauss, diffeomorphism) as operators,
* define a parameterised neural-network quantum state,
* and *train* that state to minimise constraint expectation values or other physically motivated losses.

Symbolically:

.. math::

   |\psi_\theta\rangle = \arg\min_{\psi_\theta} \big[
       \langle \psi_\theta | \hat{C}^2 | \psi_\theta \rangle
       + \lambda_\mathrm{penalty} \, \Phi[\psi_\theta]
   \big]

where :math:`\hat{C}` denotes a constraint operator and :math:`\Phi` optional penalties or regularisers
(e.g., orthogonality, gauge averaging, or symmetry enforcement).

This allows LQG research to be phrased as *optimisation problems* on neural manifolds rather than purely symbolic
constraint equations. It opens a door to quantitative experiments with quantum geometry.

----------------------------
Where neuraLQX fits in
----------------------------

neuraLQX is neither a toy simulator nor a black-box ML model. It is a *research platform* for the next generation
of loop quantum gravity simulations.

The package currently serves three distinct but related goals:

1. **Numerical experimentation:**  
   Provide a sandbox where researchers can implement and test new constraint forms, penalty terms, and ansätze.

2. **Conceptual unification:**  
   Offer a shared language between LQG theorists and computational physicists, one that abstracts away technical
   details (sampling, JAX tracing, MPI) so they can focus on ideas.

3. **Long-term infrastructure:**  
   Build the backbone for variational LQG, where future non-Abelian, graph-changing, and coupled models can live.

----------------------------
The path forward
----------------------------

Our philosophy does not end with the current release. We see neuraLQX as a living project that will evolve with
its community and the needs of quantum gravity research. The road ahead includes:

* moving from :math:`U(1)^N` to full :math:`SU(2)` support,
* implementing dynamic (graph-changing) Hamiltonians and diffeomorphism constraints,
* and enabling training over ensembles of graphs rather than a single one.

Ultimately, our goal is for neuraLQX to make **variational LQG** a mature and reproducible research programme,
a space where neural methods are used not as black boxes, but as *tools of understanding*.
