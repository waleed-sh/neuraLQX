.. _guides:

========
Guides
========

This section contains the hands-on documentation for neuraLQX. The goal of these pages is to help you
*use the library effectively* by understanding the core abstractions, the data model, and the workflows
you will use in real experiments.

If you are new to the package, the best reading order is:

1. **Graphs**: what the kinematics live on (vertices/edges, minimal loops, embeddings).
2. **Hilbert spaces**: what a configuration is and how degrees of freedom are encoded.
3. **Gauge groups**: how Gauß constraints and gauge copies are represented and evaluated.
4. **LQX models**: how physical operators and constraints are built from the kinematics.
5. **Solver**: the end-to-end VMC workflow, logging, checkpoints, and reproducibility.

You can jump directly to any guide below.

.. note::

   These guides are written as API-facing documentation, not as a paper. They are meant to be read
   while you implement models, run sweeps, and debug training runs.


-------------------------------------------
Quick links
-------------------------------------------

The sections below provide direct links to the most commonly referenced guides.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Graphs
      :link: graphs
      :link-type: doc

      Graph construction, multiedges, embeddings, and minimal loops.
      Use this when defining kinematics or building loop-based operators.

   .. grid-item-card:: Hilbert spaces
      :link: hilbert
      :link-type: doc

      Configuration encoding, cutoffs, gauge copies, and constrained (gauge-invariant) representations.

   .. grid-item-card:: Gauge groups
      :link: gauge_groups
      :link-type: doc

      Gauge-group descriptors and Gauß constraint operators (local, computational, and JAX backends).

   .. grid-item-card:: LQX models
      :link: lqx/index
      :link-type: doc

      Model interfaces, constraints, and the Euclidean / spherical / QRLG families.

   .. grid-item-card:: Solver
      :link: solvers
      :link-type: doc

      End-to-end optimisation pipeline: samplers, networks, schedules, checkpoints, MPI, and analysis.



.. toctree::
   :hidden:
   :maxdepth: 2

   graphs
   hilbert
   operators/index
   gauge_groups
   lqx/index
   vqs/index
   solvers
   errors
   debugging
   profiling
   structs