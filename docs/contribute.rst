.. _contribute:

=========================
Contributing to neuraLQX
=========================

neuraLQX is an open research software project aimed at making **variational loop quantum gravity (vLQG)**
practical by offerring a platform for reproducible simulations, inspectable abstractions, and performance that scales with modern
accelerators and parallel hardware.

We welcome contributions of all sizes, from fixing a typo to adding a new operator backend, and we try to
keep the contribution process straightforward and friendly.

.. note::

   neuraLQX is still evolving rapidly. Some parts of the codebase (especially under ``neuralqx.experimental``)
   may change without deprecation. If you plan a larger contribution, it helps to align early via an issue
   or a discussion thread so your work lands smoothly.


-----------------------
Ways to contribute
-----------------------

There are many valuable ways to help, and not all of them require writing new features.

**Help other users**
  Answer questions, share minimal examples, or explain how you solved a specific setup problem.

**Improve documentation**
  Clarify a guide, add missing conceptual context, improve docstrings, or add a small "gotcha" note where
  users usually get stuck.

**Add or refine tests**
  Add regression tests for bugs, add small correctness checks for operators on tiny Hilbert spaces, or add
  sanity checks for gauge invariance/diffeomorphism invariance.

**Contribute code**
  Fix bugs, improve performance, add new models/operators, extend the solver workflow, or harden the
  experimental API into stable form.

**Provide benchmarks / validation data**
  Small exact-diagonalisation checks, reference values for tiny graphs, convergence comparisons, timing
  measurements (including on HPC).

**Share LQG domain knowledge**
  Many contributions are "physics contributions": a clear mathematical definition, a clean mapping to code,
  and a correctness argument. Even if you are not implementing the code, writing down precise specs is
  extremely valuable.


---------------------------------
Before you open a pull request
---------------------------------

For small changes (typos, minor doc fixes, tiny bug fixes), it is totally fine to open a PR directly.

For anything non-trivial, please do one of the following first:

* **Open a GitHub Issue** (bug report, feature request, API change proposal).
* **Start a Discussion** (design questions, "is this the right approach?", roadmap alignment).

Why this matters:

* It prevents duplicate work.
* It gives maintainers a chance to confirm the direction before you invest time.
* It helps ensure API/design consistency across modules (graphs, Hilbert, gauge groups, operators, solver).

.. tip::

   If you are unsure whether something is "small" or "large": open a short discussion thread first.
   A 5-minute alignment message often saves hours later.


-----------------------
Development setup
-----------------------

The exact dependency layout can evolve as neuraLQX matures, but the following workflow is generally robust.

1) Fork and clone
---------------------------

Fork the repository on GitHub, then clone your fork:

.. code-block:: bash

   git clone https://github.com/<your-username>/neuraLQX.git
   cd neuraLQX

Add the upstream remote so you can pull updates:

.. code-block:: bash

   git remote add upstream https://github.com/waleed-sh/neuraLQX.git
   git fetch upstream

2) Create an environment
----------------------------------

Use any environment manager you like. Example with ``venv``:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -U pip

3) Install in editable mode
-------------------------------------

From the repository root:

.. code-block:: bash

   pip install -e .

If the repository provides development extras (optional), install them too:

.. code-block:: bash

   pip install -e ".[dev]"


4) Enable pre-commit hooks (required)
-------------------------------------

neuraLQX uses automated formatting to keep the codebase consistent.
Formatting is handled by **Black** and enforced via a pre-commit hook.

After installing the development dependencies, enable the hooks once:

.. code-block:: bash

   pre-commit install

From this point on:

* Code is automatically formatted on ``git commit``.
* Commits will fail if formatting cannot be applied cleanly.
* This avoids formatting-only review comments in pull requests.

.. important::

   If you skip this step, your pull request will likely fail CI due to
   formatting checks. Installing the hook locally saves time for everyone.


-----------------------
Running tests locally
-----------------------

Most contributions should include tests, and you should run the relevant subset before opening a PR. A lot of the
tests in neuraLQX require specific feature sets from both NetKet and neuraLQX which may be experimental.

Therefore, a common default is:

.. code-block:: bash

   NQX_TESTING=1 pytest -q

If you only want to run a subset:

.. code-block:: bash

   NQX_TESTING=1 pytest -q tests/<path_or_pattern>

.. important::

    If you modified code that may require MPI, you have to run the tests as follows

    .. code-block:: bash

        NQX_MPI_TESTS=1 NQX_MPI=1 NQX_TESTING=1 pytest -q

.. important::

   Many failures that appear "random" in variational Monte Carlo are actually reproducibility problems.
   When you write tests, aim for invariants and small deterministic checks (tiny graphs, tiny cutoffs,
   controlled seeds), not long stochastic convergence curves.



-------------------------------------------
Code style and quality expectations
-------------------------------------------

We try to keep the codebase approachable and consistent.
Code formatting is handled automatically using **Black** and enforced both
locally (via pre-commit) and in CI.

As a contributor, you normally do not need to run formatters manually —
just commit your changes with the pre-commit hook enabled.


As a general guideline:

**Write readable code**
  Prefer clarity over cleverness, especially in physics-heavy kernels.

**Type hints**
  Add type hints for public-facing functions where it improves readability and catches mistakes early.

**Docstrings**
  Public API should have docstrings that explain *what it does*, *what the inputs mean physically*, and
  *what the output represents*. For operator-like objects, include the defining equation where possible.

**Avoid hidden global state**
  Prefer explicit configuration via arguments or solver config objects.

**Be careful with JAX tracing**
  Avoid Python-side data-dependent control flow inside JIT kernels. Keep shapes static where feasible.
  Treat host callbacks and implicit conversions as performance hazards.

**MPI safety**
  Do not print from every rank. Ensure collective operations do not deadlock. Keep I/O on master ranks.


----------------------------------------
Experimental vs stable API
----------------------------------------

neuraLQX separates "stable" and "experimental" namespaces.

* ``neuralqx.*`` (stable): intended to be reasonably stable and documented as the public API.
* ``neuralqx.experimental.*``: opt-in features that may change or be removed without warning.

If your contribution touches experimental modules:

* keep experimental imports behind the experimental gate (``NQX_EXPERIMENTAL=1``),
* document the risk clearly,
* and avoid making stable modules depend on experimental ones unless there is an explicit migration plan.


-----------------------
Pull request workflow
-----------------------

1) Create a branch
----------------------------

Create a descriptive branch name:

.. code-block:: bash

   git checkout -b fix-gauge-sampler-reset
   # or: git checkout -b feature-computational-operator-speedup

2) Make changes in focused commits
--------------------------------------------

Small, focused commits are easier to review. If your change is large, consider splitting it into:

* refactor PR (no behaviour change),
* feature PR (new behaviour),
* docs PR (guide + examples),
* performance PR (optimisation with benchmarks).

3) Add tests and docs
-------------------------------

A good PR usually includes:

* tests for new behaviour,
* docstring updates,
* and (when relevant) a short documentation snippet or guide update.

4) Run checks locally
-------------------------------

At minimum:

.. code-block:: bash

   NQX_TESTING=1 pytest -q

If you touched docs:

.. code-block:: bash

   make -C docs html

5) Open the PR
------------------------

When opening a PR, include:

* what changed and why,
* any relevant issue/discussion link,
* how to test it,
* any performance impact (good or bad),
* and any API implications.

6) Address review feedback
------------------------------------

Reviews are collaborative: expect iteration, especially for core abstractions (Hilbert, operators, solver).
If there is a disagreement, we prefer to resolve it by writing down a precise expected behaviour and adding
a small test for it.


-------------------------------------------
What makes a great PR (checklist)
-------------------------------------------

**Correctness**
  * Behaviour is correct on small known cases (tiny graphs, tiny cutoffs).
  * Operator action matches its mathematical definition.
  * Gauge invariance/diffeomorphism invariance is respected where expected.

**Reproducibility**
  * Results are stable under fixed seeds where relevant.
  * Tests do not rely on long stochastic convergence.

**API design**
  * Public API changes are intentional and documented.
  * Experimental features remain gated and clearly marked.
  * Names match existing conventions across modules.

**Performance**
  * Avoids accidental Python loops in hot paths.
  * Uses vectorisation/JIT appropriately.
  * Does not introduce shape polymorphism pitfalls unless justified.

**Docs**
  * New user-facing features are documented.
  * Docstrings include physical meaning and usage examples where helpful.

**Logging / UX**
  * Errors are informative.
  * Solver-facing changes surface meaningful configuration in logs.


--------------------------------------------------
Contributing physics: operators and models
--------------------------------------------------

Some of the most important contributions to neuraLQX are physics contributions: adding or improving operators,
constraints, projectors, or model definitions.

When you contribute a new physics object, please include:

**1) A precise definition**
  Provide the mathematical expression, including conventions (graph orientation, index ordering, adjoints).

**2) A minimal correctness strategy**
  For example:
  * check hermiticity on a tiny Hilbert space,
  * compare local estimator vs explicit matrix on a tiny graph if possible,
  * verify gauge invariance numerically,
  * verify symmetry under graph automorphisms where applicable.

**3) A usage example**
  A short snippet in docs or a small test is often enough.

**4) A clear placement**
  Explain whether the object belongs as a ``LocalOperator``-like implementation, a ``ComputationalOperator``-like
  implementation, or both. If both are possible, it is often useful to implement both to keep model APIs
  uniform.


-----------------------
Reporting bugs
-----------------------

Good bug reports are a major contribution.

When reporting a bug, include:

* neuraLQX version or commit hash,
* Python version,
* JAX + NetKet versions,
* CPU/GPU details if relevant,
* minimal reproduction code (small graph + small cutoff),
* and the full error traceback.

If the issue depends on runtime configuration, include the environment variables you used, for example:

.. code-block:: bash

   export NQX_VERBOSE=True
   export NQX_MPI=0
   export NQX_EXPERIMENTAL=0


-----------------------
Community expectations
-----------------------

We want neuraLQX to be a welcoming place for collaboration across physics, numerics, and ML.
Be respectful, assume good faith, and keep discussions focused on ideas and evidence.
If the repository includes a formal Code of Conduct, please follow it.


---------------------------------
Where to ask questions
---------------------------------

If you are unsure where something belongs, or how to design a contribution:

* open a GitHub Discussion,
* or open a small draft PR early and ask for direction.

Early questions are encouraged — they save time for everyone.


Thanks
----------------

neuraLQX stands on the shoulders of the open scientific software ecosystem (especially NetKet and JAX).
Contributions of any size help move vLQG forward, and we appreciate your time and care.

