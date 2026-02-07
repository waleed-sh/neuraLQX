.. image:: _static/logo.png
   :align: center
   :alt: neuralQX logo
   :width: 400px
   :class: no-scaled-link

.. raw:: html

   <div style="margin-bottom: 24px;"></div>



.. raw:: html

   <div style="text-align:center; margin-top: 10px; margin-bottom: 10px;">
     <div style="font-size: 36px; font-weight: 800; letter-spacing: -0.02em;">

     </div>
     <div style="font-size: 16px; opacity: 0.85; margin-top: 6px;">
       High-performance variational simulations for canonical Loop Quantum Gravity, built on NetKet &amp; JAX.
     </div>
   </div>

.. raw:: html

   <div style="max-width: 980px; margin: 0 auto;">

.. raw:: html

   <div style="display:flex; flex-wrap:wrap; gap:10px; justify-content:center; margin: 18px 0 8px 0;">
     <a href="quickstart.html" class="no-underline">
       <span style="text-decoration: none; display:inline-block; padding:10px 14px; border-radius:999px; border:1px solid rgba(0,0,0,0.15); font-weight:600;">
         🚀 Quickstart
       </span>
     </a>
     <a href="installation.html" class="no-underline">
       <span style="display:inline-block; padding:10px 14px; border-radius:999px; border:1px solid rgba(0,0,0,0.15); font-weight:600;">
         📦 Install
       </span>
     </a>
     <a href="guides/index.html" class="no-underline">
       <span style="display:inline-block; padding:10px 14px; border-radius:999px; border:1px solid rgba(0,0,0,0.15); font-weight:600;">
         ⚙️ Guides
       </span>
     </a>
     <a href="tutorials/index.html" class="no-underline">
       <span style="display:inline-block; padding:10px 14px; border-radius:999px; border:1px solid rgba(0,0,0,0.15); font-weight:600;">
         🧠 Tutorials
       </span>
     </a>
     <a href="documentation/index.html" class="no-underline">
       <span style="display:inline-block; padding:10px 14px; border-radius:999px; border:1px solid rgba(0,0,0,0.15); font-weight:600;">
         📘 API Docs
       </span>
     </a>
   </div>

.. raw:: html

   </div>

.. attention::
    The package is currently under development and will be released at a later date. Once available, it can be
    installed via pip. Check back here for the release data later.

    **Release Date:** TBA


What is neuraLQX?
===================

**neuraLQX** is an open-source Python package for **high-performance variational simulations of canonical loop quantum gravity**.
It is designed to let you work directly with LQG-native objects (graphs, Hilbert spaces, gauge groups, constraints, projectors)
while leveraging a battle-tested variational backend.

Under the hood, neuraLQX builds on **NetKet** and **JAX**. In practice, this means you can use state of the art Monte Carlo sampling,
automatic differentiation, and scalable optimisation workflows, while writing simulations in a vocabulary that matches LQG.

.. raw:: html

   <div style="height: 12px;"></div>


Start here
============

If you are new to the project, the fastest path is:

1. Read :doc:`philosophy` to understand the design goals and the long-term direction.
2. Follow :doc:`quickstart` to run your first end-to-end workflow.
3. Use the :doc:`guides/index` pages when you want to go deeper into the core building blocks.

.. raw:: html

   <div style="height: 12px;"></div>


In the meantime, explore:
===========================

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 📘 Documentation
      :link: documentation/index.html
      :text-align: center

      Core modules, API concepts, and reference pages.

   .. grid-item-card:: ⚙️ Guides
      :link: guides/index.html
      :text-align: center

      Practical, "how-to" pages for graphs, Hilbert spaces, gauge groups, models, and solvers.

   .. grid-item-card:: 🧠 Tutorials
      :link: tutorials/index.html
      :text-align: center

      Worked examples and end-to-end workflows.

   .. grid-item-card:: 💡 Getting Started
      :link: installation.html
      :text-align: center

      Installation and configuration basics.


Key features
==============

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 🧩 Modular by design
      :text-align: left

      neuraLQX is organised around clearly separated modules (graphs, Hilbert spaces, gauge groups, operators, projectors,
      solver/driver orchestration). This keeps workflows inspectable and makes it easier to extend or swap components.

   .. grid-item-card:: 🕸️ Graph-native LQG workflows
      :text-align: left

      Define the discrete geometric backbone explicitly and build consistent kinematics and operators on top.
      Graphs are treated as first-class objects that drive indexing, orientation conventions, and operator structure.

   .. grid-item-card:: 🧮 Constraint-driven optimisation
      :text-align: left

      Express Gauß, Hamiltonian, and diffeomorphism-relevant structures as operators and run variational optimisation loops
      that target constraint satisfaction via expectation values, penalties, and projectors.

   .. grid-item-card:: ⚡ Performance-first backend
      :text-align: left

      JAX + NetKet provide fast sampling, differentiation, and optimisation. neuraLQX focuses on LQG-specific structure
      while staying compatible with NetKet's variational ecosystem.

.. raw:: html

   <div style="height: 8px;"></div>


What you can do today
======================

The current focus is the Abelian :math:`U(1)^N` playground: a clean environment where we can test operator constructions,
Hilbert-space layouts, gauge-invariant sampling strategies, and end-to-end solver workflows.

This includes (among other things):

* fixed-graph models where edges carry :math:`U(1)^N` degrees of freedom,
* multiple operator backends (local-operator style vs. "act-on-state" computational operators),
* kinematical and reduced (gauge-invariant) Hilbert-space constructions,
* projectors and penalty-based workflows that integrate into variational Monte Carlo,
* and a solver that coordinates models, samplers, networks, and logging in a reproducible way.

.. raw:: html

   <div style="height: 14px;"></div>


Suggestions
============

If you have ideas (for example, operators or models you would like to see implemented), please share them by opening a new post
in the ideas discussion board:

`Ideas & suggestions <https://github.com/waleed-sh/neuraLQX/discussions/new?category=ideas>`_

P.S. you will need a GitHub account.


Examples
=========

The package is still under development, but its core functionality has already been used in real workflows.
For a taste of the kind of systems neuraLQX targets, see the papers linked from the repository and the tutorial material.

You can also find more hands-on examples in the tutorials area. During development, some tutorial content may live as notebooks
in the GitHub repository before it is fully integrated into Read the Docs.


License
========

This package will be available under the Apache 2.0 license.
You can read the license text here:

`Apache 2.0 License <https://github.com/waleed-sh/neuraLQX/blob/main/LICENSE>`_


.. raw:: html

   <div style="margin-top: 18px; opacity: 0.75; font-size: 13px; text-align: center;">
     Built with ❤️ for reproducible variational LQG research.
   </div>

