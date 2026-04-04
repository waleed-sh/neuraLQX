.. _experimental:

==================
Experimental API
==================

This section documents **experimental** neuraLQX functionality.

The experimental namespace is where new ideas land first: prototype drivers, new variational-state
implementations, alternative estimators, and workflow abstractions that are already useful for research,
but are **not yet part of the stable public API**.

You should use this section when you explicitly want early access to features that are a significant extensions of the
standard neuraLQX public API.

.. warning::

   Experimental features come with **no stability guarantee**.

   Concretely, anything under ``neuralqx.experimental`` may:

   * change behaviour between releases,
   * change names or signatures without deprecation warnings,
   * break in subtle ways for some parameter regimes,
   * be removed entirely if it turns out to be a dead end,
   * change checkpoint/log formats (and thus old files may stop loading),
   * and may have performance characteristics that shift as implementations evolve,
   * are not guaranteed to function as intended or give correct results.

   If you are writing long-lived or production-grade code, or using neuraLQX for research, prefer the stable API first, and
   only adopt experimental features after pinning neuraLQX to a specific version and validating your workflow.


-------------------------------------
Enabling experimental features
-------------------------------------

neuraLQX gates experimental functionality behind the ``EXPERIMENTAL`` configuration flag.
There are two reliable ways to enable it.

Enable via ``cfg.update`` (recommended for scripts and Jupyter)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Call ``cfg.update`` immediately after importing neuralqx, **before** importing anything
from ``neuralqx.experimental``:

.. code-block:: python

   import neuralqx as nqx
   nqx.cfg.update("EXPERIMENTAL", True)

   # now safe to import experimental
   import neuralqx.experimental as nqxx

This works regardless of import order and is the only reliable method in Jupyter notebooks
or any environment where ``neuralqx`` may have been imported before your code runs.

Enable via shell environment variable (recommended for batch jobs and scripts)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``NQX_EXPERIMENTAL=1`` in your shell **before launching Python**:

.. code-block:: bash

   # one-off
   NQX_EXPERIMENTAL=1 python your_script.py

   # or export it for the whole session
   export NQX_EXPERIMENTAL=1
   python your_script.py

This works because the environment variable is present before Python starts, so it is read
correctly when ``neuralqx`` is first imported.

.. warning::

   **Do not** set ``os.environ["NQX_EXPERIMENTAL"] = "1"`` inside Python as a substitute.
   The configuration system reads environment variables **once**, at the moment
   ``neuralqx`` is first imported into the process. In Jupyter kernels, IPython sessions,
   or anywhere another package has already imported ``neuralqx`` transitively, the value
   is already frozen, subsequent writes to ``os.environ`` are silently ignored.

   .. code-block:: python

      # This looks correct but is unreliable, do not rely on it
      import os
      os.environ["NQX_EXPERIMENTAL"] = "1"
      import neuralqx as nqx  # configs may already be initialized

   Use ``nqx.cfg.update("EXPERIMENTAL", True)`` instead.


-----------------------
What lives here?
-----------------------

The experimental namespace typically includes:

* Prototype **drivers** (for example, multi-state VMC),
* Prototype **variational states** and estimators,
* Prototype **neural network architectures**,
* Early **operator backends** or kernel variations,
* Convenience tooling that has not yet been hardened into the stable public API.

A good mental model is: if it is under ``neuralqx.experimental``, it is allowed to move fast.


-----------------------
How to import
-----------------------

Once ``NQX_EXPERIMENTAL=1`` is set, you can import the experimental namespace like this:

.. code-block:: python

   import neuralqx.experimental as nqxx

You can also import concrete objects directly:

.. code-block:: python

   from neuralqx.experimental.driver import MultiStateVMC
   from neuralqx.experimental.vqs import MultiMCState
   from neuralqx.experimental.solver import MultiSolver


-----------------------
Reading order
-----------------------

Experimental features in neuraLQX almost always extend *one* part of the stable workflow rather than replacing
everything. The recommended order is:

1. Read the stable **Solver** and **VQS / MCState** documentation first (so you understand the baseline flow).
2. Read the experimental feature page you need (for example multi-state VMC).
3. Start from a minimal example, then integrate into your full model.

This reduces debugging time because you always know which part of the pipeline is "standard" and which part is
"experimental".

.. hint::

   The experimental API in neuraLQX mirrors the public API in its structure (i.e. a new experimental VMC driver will be
   available under ``neuralqx.experimental.driver`` since in the public API, drivers are found under ``neuralqx.driver``.


-----------------------
Pages
-----------------------

.. toctree::
   :maxdepth: 1
   :caption: Experimental API

   mvmc


-------------------------------------------
Quick links
-------------------------------------------

.. grid:: 4
   :gutter: 2

   .. grid-item-card:: Experimental API Reference
      :link: api/index
      :link-type: doc

      The reference for the complete experimental API in neuraLQX.


   .. grid-item-card:: Multi-state VMC
      :link: mvmc
      :link-type: doc

      Jointly optimise several variational states with an orthogonality (fidelity) penalty.
      Includes ``MultiMCState``, ``MultiStateVMC``, and the ``MultiSolver`` wrapper.

   .. grid-item-card:: Single-Trunk Multi-state VMC
      :link: stmh_vmc
      :link-type: doc

      Jointly optimise several variational states with an orthogonality (fidelity) penalty, using one neural network.

   .. grid-item-card:: Structs and data containers
      :link: structs_guide
      :link-type: ref

      Immutable, JAX-native data containers with explicit field kinds, derived
      fields, serialisation, and pytree registration for arbitrary classes.

   .. grid-item-card:: Symbolic Operator DSL
      :link: symbolic_operators
      :link-type: ref

      Define matrix-free operators declaratively (iterators, predicates,
      emissions) and compile them into executable JAX-backed operator kernels.

   .. grid-item-card:: Stable guides
      :link: guides
      :link-type: ref

      Go back to the stable documentation set (graphs, hilbert, gauge groups, models, solver) that the experimental
      features build on.

