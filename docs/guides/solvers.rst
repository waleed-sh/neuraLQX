.. _solver:

=========
Solvers
=========

The :class:`neuralqx.solver.Solver` is the orchestration layer that turns a fully specified
physical model (an :class:`neuralqx.lqx.AbstractLqxInterface` instance) into a reproducible
variational Monte Carlo (VMC) training workflow. It is the object you interact with when
you want to:

* bind together the **physical definition** (Hilbert space, graph, gauge group, constraints/operators),
* choose **algorithmic components** (ansatz, sampler, optimiser, optional preconditioners),
* run a consistent optimisation loop with logging/monitoring,
* checkpoint, resume, and analyse results without rebuilding ad-hoc scripts around NetKet.

The solver is intentionally **thin**. The numerically heavy work (sampling, stochastic reconfiguration, driver loops)
is delegated to NetKet machinery. The solver provides a guarded configuration protocol, constructs
compatible VMC objects, manages runtime bookkeeping, and offers uniform export/import
and diagnostics across local runs and MPI runs.

This page documents the solver family as an API you can rely on in experiments, sweeps, and HPC jobs.
Use :class:`neuralqx.solver.Solver` for one variational state, and
:class:`neuralqx.solver.MultiSolver` for the public multi-state workflow.


Single-state Solver
-------------------

Conceptual model
^^^^^^^^^^^^^^^^^^^

A solver instance is stateful. Conceptually it owns two categories of objects:

**(A) Physical model references** (they come from the LQX model you pass in):

* ``solver.lqx``: the active :class:`neuralqx.lqx.AbstractLqxInterface` instance.
* ``solver.hilbert``: the Hilbert space associated with that model as a subclass of :class:`neuralqx.hilbert.AbstractHilbertInterface`
* ``solver.graph``: the underlying graph (kinematics / combinatorics of degrees of freedom).
* ``solver.gauge_group``: the gauge group object used in the kinematical layer.

These are *not* optional. If the physical model changes, you are solving a different problem.

**(B) Algorithmic objects** (you configure them explicitly):

* a sampler (Markov chain strategy and its hyperparameters),
* an optimiser (and optional stochastic reconfiguration settings),
* a neural network ansatz (a :mod:`flax.linen` module),
* a variational state (neuralqx MCState),
* a VMC driver (NetKet variational driver) that performs the optimisation loop.

The solver is guarded on purpose. It separates *configuration* from *execution* to avoid
ambiguous semantics (for example, "changing the sampler" mid-run without rebuilding
dependent objects).


The guarded configuration protocol
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The solver follows a staged protocol:

1. **Attach a sampler** via :meth:`~neuralqx.solver.Solver.set_sampler`.
2. **Attach an optimiser** via :meth:`~neuralqx.solver.Solver.set_optimizer`.
3. **Attach an ansatz** via :meth:`~neuralqx.solver.Solver.set_network`.

Only after those are set can the solver initialise the underlying VMC state/driver.

This protocol matters because the constructed VMC objects depend on the combination of
Hilbert space, sampler, optimiser, and model. It also makes "resume from disk" behaviour
predictable: loading a checkpoint is meaningful only if the same full system exists again.


Constructing a Solver
^^^^^^^^^^^^^^^^^^^^^^^

Minimal construction:

.. code-block:: python

   import neuralqx as nqx

   # lqx is any concrete model implementing the AbstractLqxInterface
   solver = nqx.solver.Solver(lqx)

At construction time, the solver also manages output paths, run identifiers, and seeding.

Output directory layout
~~~~~~~~~~~~~~~~~~~~~~~

By default, a solver creates a dedicated output directory under a default ``Output Data``
folder (relative to your script). You can change this behaviour:

.. code-block:: python

   solver = nqx.solver.Solver(
       lqx,
       output_path="my_results",
       auxiliary_path="scan/run_001",
   )

With this pattern, *all* artifacts for that solver instance go to
``my_results/scan/run_001``. This is intentionally boring, the point is that you can run
large sweeps and keep the filesystem stable and predictable.

Seeding rules
~~~~~~~~~~~~~

The solver accepts an optional integer seed. That seed is used for the variational state
(initial network parameters), while a deterministic integer transformation is used for the
sampler seed (so that parameters and sampling are reproducibly tied together without
requiring you to seed everything manually).

.. code-block:: python

   solver = nqx.solver.Solver(lqx, seed=42)

Cleanup behaviour
~~~~~~~~~~~~~~~~~

An optional ``clean_up`` flag can remove empty directories under ``output_path``.
This is mostly helpful in sweep scripts that create many runs but sometimes exit early.

.. code-block:: python

   solver = nqx.solver.Solver(lqx, output_path="my_results", clean_up=True)

.. warning::

   You should set ``clean_up=False`` for parallel jobs so that different ranks do not delete the output directories of
   one another!


Step 1: Choosing and configuring a sampler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sampler is responsible for producing configurations :math:`\sigma` distributed
approximately as :math:`p_\theta(\sigma) \propto |\psi_\theta(\sigma)|^2`, where
:math:`\psi_\theta` is your current variational wavefunction represented by the network.

In practice, this is a Markov chain sampler. The solver exposes a high-level API that
lets you pick a sampler type and common hyperparameters without wiring NetKet objects
manually for every run.

A typical usage looks like:

.. code-block:: python

   solver.set_sampler(
       sampler_type="Metropolis Local",
       number_of_chains=32,
       number_of_samples=4096,
       number_of_sweeps=20,
       reset_chains=False,
   )

Sampler hyperparameters are not cosmetic, they determine autocorrelation time, effective
sample size, and the bias/variance tradeoff of your expectation estimates. A practical
interpretation of the main parameters:

* **number_of_chains**: number of parallel Markov chains. More chains reduce correlation and
  improve parallel efficiency (especially under MPI/GPU), but increase per-iteration cost.

  .. warning::

     In parallel jobs, this is the **number of chains per MPI-rank**

* **number_of_samples**: number of measured samples per iteration. This controls the statistical
  error bars in NetKet's :class:`netket.stats.Stats`.
* **number_of_sweeps**: how many local updates define a "sweep". If you leave it to defaults,
  you get a sensible baseline. If your model has nontrivial graph structure or constraints,
  you may want to tune it.

.. hint::

   If you are doing constraint solving (typical in LQG-style workflows), you often care
   about stable estimation of small residuals. In that regime, under-sampling is the
   fastest way to "converge" to nonsense, or not converge at all. Start with conservative sampling parameters
   and only reduce them once you have validated stability.

MPI behaviour
~~~~~~~~~~~~~

In MPI runs, all ranks participate in sampling and estimation. File I/O (exports, plots,
HTML logs) is performed by the global master rank to prevent inconsistent artifacts and
race conditions. This becomes relevant when you checkpoint frequently.

Practical sampler tuning loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A robust workflow is:

1. Start with more chains and fewer samples.
2. Inspect acceptance rates/autocorrelation (NetKet exposes diagnostics, neuraLQX logs
   the runtime curve and ``Stats``).
3. Increase samples until error bars are small compared to the scale of your objective.
4. Only then tune optimiser hyperparameters.

This order is deliberate, optimiser tuning is meaningless if the gradient is dominated by
Monte Carlo noise.


Step 2: Choosing an optimiser
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimisation updates parameters :math:`\theta` of the variational state. neuraLQX uses
optax optimisers, and optionally uses NetKet's stochastic reconfiguration (SR) as a
preconditioner. In VMC/SR language, the update has the form

.. math::

   \theta_{t+1} = \theta_t - \eta \, S^{-1}(\theta_t)\, g(\theta_t),

where :math:`g` is the estimated gradient and :math:`S` is the (regularised) covariance /
quantum Fisher information matrix induced by the variational family and sampling.

A typical setup:

.. code-block:: python

   solver.set_optimizer(
       optimizer_type="Adam",
       learning_rate=1e-3,
       diagonal_shift=1e-2,
   )

How to think about the main parameters:

* **learning_rate**: sets the step scale for the underlying optax optimiser. If you use SR,
  the effective step scale is shaped by the preconditioner.
* **diag_shift**: regularisation added to stabilise the SR solve. Too small can blow up,
  too large turns SR into something closer to vanilla gradient descent.

A practical SR heuristic:

* If loss decreases but becomes noisy and unstable, increase ``diagonal_shift``.
* If loss decreases extremely slowly, reduce ``diagonal_shift`` or increase sampling quality.
* If you see erratic jumps, check sampler quality before blaming the optimiser.

.. warning::

   SR is sensitive to sampling quality. If you aggressively reduce samples, SR can become
   ill-conditioned and you will spend time "tuning ``diagonal_shift``" to patch measurement noise.






Scheduled learning rates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In neuraLQX you do not need to construct Optax schedules manually. The solver exposes
scheduled learning rates *directly* through the optimiser configuration step. Concretely,
:meth:`neuralqx.solver.Solver.set_optimizer` accepts two schedule-specific keywords:

* ``scheduler_type``: a string selecting a schedule family (case-insensitive).
* ``scheduler_parameters``: a dictionary of parameters required by that schedule family.

If you do **not** provide ``scheduler_type``, the solver treats ``learning_rate`` as a constant
scalar and builds the optimiser with a fixed step size. If you **do** provide
``scheduler_type``, the step size becomes a function of the optimisation step counter
(the VMC iteration index), and the solver builds the corresponding Optax schedule
internally and wires it into the optimiser update rule.

This matters in practice because VMC objectives often have two distinct regimes:

1. **Early iterations**: you want large, decisive parameter updates to quickly move into a
   reasonable region of parameter space.
2. **Late iterations**: the objective becomes small and Monte-Carlo noise becomes a
   non-negligible part of the update signal, reducing the learning rate stabilises training and
   prevents "hovering" around a minimum with persistent oscillations.

Calling ``set_optimizer`` multiple times is a supported workflow. Every call replaces the
previous optimiser configuration (including any schedule) so you can reconfigure runs
between stages or after resuming from checkpoints.

Schedule families and required parameters
"""""""""""""""""""""""""""""""""""""""""""""""""

The schedule family is selected by ``scheduler_type``. The supported families are:

Exponential Decay
  A multiplicative decay where the learning rate decreases smoothly at a fixed rate.

  Required keys in ``scheduler_parameters``:
  ``init_value`` (float), ``transition_steps`` (int), ``decay_rate`` (float)

  Optional keys:
  ``staircase`` (bool), ``end_value`` (float)

Cosine Decay
  A smooth annealing schedule based on a cosine profile, typically used when you want a
  gentle decay over a fixed horizon.

  Required keys:
  ``init_value`` (float), ``decay_steps`` (int)

  Optional keys:
  ``alpha`` (float)

Linear Decay
  A linear interpolation from an initial learning rate down to an end value across a fixed
  number of steps.

  Required keys:
  ``init_value`` (float), ``end_value`` (float), ``transition_steps`` (int)

  Optional keys:
  ``transition_begin`` (int)

A key point: the "step" in these schedules is the **optimisation step** (i.e. VMC iteration),
not MCMC sweeps inside a single iteration. This means that ``transition_steps`` / ``decay_steps``
should be chosen on the scale of **how many iterations** you expect to run.

Concrete usage
"""""""""""""""""""""

Fixed learning rate (no schedule):

.. code-block:: python

   solver.set_optimizer(
       "Adam",
       learning_rate=1e-3,
   )

Exponential decay schedule:

.. code-block:: python

   solver.set_optimizer(
       "Adam",
       scheduler_type="Exponential Decay",
       scheduler_parameters={
           "init_value": 1e-3,
           "transition_steps": 200,
           "decay_rate": 0.98,
           # optional:
           # "staircase": False,
           # "end_value": 1e-6,
       },
   )

Cosine decay schedule:

.. code-block:: python

   solver.set_optimizer(
       "Adam",
       scheduler_type="Cosine Decay",
       scheduler_parameters={
           "init_value": 1e-3,
           "decay_steps": 5000,
           # optional:
           # "alpha": 0.0,
       },
   )

Linear decay schedule:

.. code-block:: python

   solver.set_optimizer(
       "Adam",
       scheduler_type="Linear Decay",
       scheduler_parameters={
           "init_value": 1e-3,
           "end_value": 1e-5,
           "transition_steps": 4000,
           # optional:
           # "transition_begin": 0,
       },
   )

Scheduling and SR (stochastic reconfiguration)
"""""""""""""""""""""""""""""""""""""""""""""""""

Learning-rate schedules compose naturally with SR, because SR changes the *geometry* of
the update direction while the schedule changes the *global scale* of each update. In other
words, SR determines "which direction is sensible", while the schedule determines "how far
to move along that direction at iteration ``t``".




Step 3: Attaching an ansatz (and diff-invariance)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You attach a network as a :mod:`flax.linen` module. Minimal pattern:

.. code-block:: python

   solver.set_network(MyFlaxModel())

Once a sampler and optimiser are set and a network is attached, the solver can initialise
its internal VMC objects (variational state and driver). You normally do *not* call the
initialisation manually, it happens when you start the first run.

Diffeomorphism invariance via symmetry projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many workflows want diffeomorphism invariance in the sense of invariance under graph
automorphisms / symmetry actions on configurations. The solver supports enabling this
at network-attachment time by providing a list of symmetries:

.. code-block:: python

   solver.set_network(
       MyFlaxModel(),
       diff_invariant=True,
       symmetries=symmetry_list,
   )

The high-level idea is

* a symmetry :math:`g` acts on configurations :math:`\sigma \mapsto g\cdot\sigma` (typically
  by permuting degrees of freedom consistent with graph automorphisms),
* the projected amplitude is built from amplitudes evaluated on the orbit of :math:`\sigma`,
  so the resulting state is invariant under the group action.

You will typically generate ``symmetry_list`` from graph utilities (for example, graph
automorphisms). In documentation and examples, treat this as an explicit input, the
solver will not guess it for you.

Two ways to use diff-invariant states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Approach A: Train inside the invariant manifold.**

Enable ``diff_invariant=True`` from the start. Then optimisation updates the parameters of
a projected ansatz, and the *trained* state is invariant by construction.

This is the right approach when your *objective* is defined on the invariant subspace,
for example when you interpret solutions as diffeo-equivalence classes.

**Approach B: Train a flexible state, then project for evaluation.**

Train a non-projected ansatz, then build an invariant state for evaluation of invariant observables.

This is useful if the invariant projection is expensive and you only need it at the end.

Both are valid. The solver's "diff-invariant network attachment" is geared toward
Approach A, because it keeps your run pipeline simple and reproducible.

Custom networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your network will be evaluated on batches of configurations produced by the sampler.
In most cases, a configuration is a 1D integer array encoding local quantum numbers.
A minimal Flax module that maps configurations to a complex log-amplitude could look like:

.. code-block:: python

   import flax.linen as nn
   import jax.numpy as jnp

   class MyFlaxModel(nn.Module):
       features: int = 64

       @nn.compact
       def __call__(self, sigma):
           # sigma: (batch, n_dofs) integer-like values
           x = sigma.astype(jnp.float32)

           # simple MLP baseline
           x = nn.Dense(self.features)(x)
           x = nn.tanh(x)
           x = nn.Dense(self.features)(x)
           x = nn.tanh(x)

           # output: log-amplitude (real) + phase (real)
           out = nn.Dense(1)(x)             # (batch, 2)
           return jax.numpy.sum(out, axis=-1)

Then:

.. code-block:: python

   solver.set_network(MyFlaxModel(features=128))

If you want the network to see a more structured view (for example, vertex-wise blocks or
edge-wise blocks), you have two common strategies:

1. reshape inside ``__call__`` based on knowledge of your Hilbert layout,
2. wrap the network with a projector/adapter that constructs the desired input view.

The solver can work with either, as long as the resulting apply function is compatible with
NetKet's variational state interface.

.. note::

   If your gauge group or Hilbert space uses constrained encodings, make sure your network
   is compatible with the sampler's output dtype/shape.


Running simulations
^^^^^^^^^^^^^^^^^^^

Once configured, the main entry point is :meth:`~neuralqx.solver.Solver.run`. It performs
a fixed number of VMC iterations, records time-series data into a runtime log, and exports
diagnostics on completion.

Signature and semantics
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   solver.run(
       n_iters,
       silent_print=False,
       silent_plot=True,
       timer=True,
       live_monitoring=False,
       callbacks: Callable[
            [int, dict, "AbstractVariationalDriver"], bool
        ] = lambda *x: True,
       observables=None,
       **kwargs,
   )

What each argument is for:

**n_iters**
  Number of optimisation iterations to execute.

**callbacks**
  One callable or a list of callables, executed after every iteration. Each callback
  receives ``(iteration_index, log_data_dict, driver)`` and must return a boolean.
  If any callback returns ``False``, the run stops cleanly.

  This is the recommended mechanism for early stopping (wall-time limits, plateau
  detection, scheduled hyperparameter changes).

**observables**
  Optional ``dict[str, operator]``. Each operator is evaluated and logged at every
  iteration in addition to the main optimisation target. Use this when you want to track
  multiple constraints, diagnostic operators, or partial contributions without rebuilding
  the objective.

**live_monitoring**
  When enabled, the solver runs a local in-browser monitoring view that updates during
  training. This is convenient for interactive debugging or short runs on a workstation.

**silent_print / silent_plot**
  Controls whether the final log/plot are shown at the end of the run. This is helpful
  for batch scripts where you only want exported artifacts.

**timer**
  Enables timing/profiling integration through NetKet.

Minimal run:

.. code-block:: python

   solver.run(500)

Run with monitoring, callbacks, and observables:

.. code-block:: python

   solver.run(
       500,
       live_monitoring=True,
       callbacks=[walltime_guard_callback()],
       observables={"H": H, "V": V},
   )

.. hint::

  Want an interactive live view of the simulation? Enable ``live_monitoring=True`` when starting the simulation. This
  will give you a local, in-browser interactive view.


Interrupt behaviour (single-rank vs MPI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A run can be interrupted at any time:

* In **single-rank** runs, ``Ctrl+C`` triggers a graceful abort: the solver terminates,
  exports what it can (logs/plots), and checkpoint-exports the current variational state.

* In **multi-rank MPI** runs, ``Ctrl+C`` is treated as potentially rank-skewed and may
  trigger an MPI abort to prevent deadlocks from incomplete collectives.

For distributed runs, prefer callback-driven early stopping and explicit checkpointing.


What gets exported after a run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the end of a successful run (and after a graceful single-rank abort), the solver exports
a compact set of artifacts into its output directory:

* runtime log time-series (objective and any registered observables),
* optimisation curve plot (image),
* an HTML logger report (metadata/config summary),
* a checkpoint of the current variational state (serialised MCState).

In MPI runs, only the master rank writes files. All ranks participate in serialisation and
synchronise so the checkpoint is consistent.


Continuing and resuming simulations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Long runs are often executed in stages, as you may want to extend training, change monitoring,
or resume from a checkpoint created on a different machine.

In-memory continuation
~~~~~~~~~~~~~~~~~~~~~~~

If you are still in the same Python session:

.. code-block:: python

   solver.run(500)
   solver.continue_simulation(100)

This extends training without reinitialising the solver.

Resume from disk (checkpoint)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can resume from an exported checkpoint:

.. code-block:: python

   solver.continue_simulation(
       state_path="path/to/serialised_state.mpack",
       n_iters=100,
       live_monitoring=True,
       force_load_mpi=True,
   )

Important constraints:

* When resuming from a checkpoint, rebuild the entire system identically: same physical
  model and same solver configuration (sampler, optimiser, network). A checkpoint stores
  the trained network parameters and sampler configuration needed to reproduce sampling.

* By default, continuation preserves the existing runtime log and reuses previously
  registered observables. You can override ``observables`` and ``callbacks`` in the
  continuation call to change monitoring and stopping logic for the next stage.

The ``force_load_mpi`` option exists for best-effort reconstruction when the MPI layout
differs between the save and resume environments. Use it deliberately. It is meant to
help practical workflows, not to make incompatible states magically compatible.


Checkpointing utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The solver exposes explicit checkpointing methods. A "state" here means a fully serialised
variational state (``MCState``) which includes network parameters plus the sampling configuration required
to continue a run.

Exporting
~~~~~~~~~~~~

.. code-block:: python

   path = solver.export_state(marker="FinalState")

If you omit the explicit ``state`` argument, the solver exports its current
``variational_state``. Exported filenames include solver identifiers and an optional marker
string, so you can keep multiple checkpoints per experiment without hand-writing paths.

Importing
~~~~~~~~~~~

.. code-block:: python

   vstate = solver.import_state("path/to/serialised_state.mpack", force_load_mpi=True)

In MPI, only the master rank reads from disk, the payload is broadcast and reconstructed
locally on all ranks.

A common pattern is:

.. code-block:: python

   vstate = solver.import_state(".../FinalState.mpack")
   solver.variational_state = vstate  # if you want to attach it explicitly
   print(vstate.expect(H))



Post-run analysis utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After a run, you generally want two things:

1. compute expectation values on the final state,
2. inspect logged time-series of objective/observables.

Expectation values via solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solver exposes a convenience :meth:`~neuralqx.solver.Solver.expect` which uses the
current variational state.

.. code-block:: python

   stats = solver.expect(H)

Under the hood, this delegates to the ``MCState`` measurement routines. In neuraLQX, those
routines support "sequence expectations" (feeding a list of operators) so that contributions
can be measured on the same sample batch and combined at the estimator level when
appropriate.

That means you can often write:

.. code-block:: python

   stats = solver.variational_state.expect([O1, O2, O3])

instead of forcing operator arithmetic into a single composite operator object. This is
useful when pieces come from different operator families or when you want to preserve
operator identities while still measuring their sum on shared samples.

.. hint::

   You can use ``Solver.expect(operator, n_samples=...)`` to compute the expectation value of an operator with a different
   number of samples than what is currently in your variational state. **Note that this will resample and thus override the existing samples.**


Runtime logs
~~~~~~~~~~~~~~

The solver maintains a runtime log of the objective and observables across iterations. In
scripts, you will typically interact with:

* exported files in the run directory (for plotting/analysis outside Python),
* an in-memory log object (for quick checks),
* convenience plotting via :meth:`~neuralqx.solver.Solver.plot_results`.

Plotting the optimisation curve:

.. code-block:: python

   solver.plot_results()

In automated sweeps you will generally keep ``silent_plot=True`` in ``run`` and rely on
exported plots, but interactive runs often benefit from calling ``plot_results`` directly.

Reading summary statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For many workflows, the most convenient in-memory representation is the NetKet-like
logging dictionary ``solver.log``. It is organised by observable name, and stores
statistics such as mean and error estimates.

Example pattern:

.. code-block:: python

   C_mean  = solver.log["Constraint"]["Mean"]
   C_sigma = solver.log["Constraint"]["Sigma"]

This style is convenient when you run many short trainings and want to aggregate results
without parsing exported files.

.. note::

   The keys present in ``log`` depend on what you log (objective name and any
   registered observables). Always print or inspect keys in exploratory scripts.


End-to-end examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example 1: Basic solve with explicit exports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neuralqx as nqx
   import flax.linen as nn
   import jax.numpy as jnp

   # assume an LQX object 'lqx' exists...

   class MLP(nn.Module):
       @nn.compact
       def __call__(self, sigma):
           x = sigma.astype(jnp.float32)
           x = nn.Dense(128)(x); x = nn.tanh(x)
           x = nn.Dense(128)(x); x = nn.tanh(x)
           out = nn.Dense(1)(x)
           return jax.numpy.sum(out, axis=-1)

   solver = nqx.solver.Solver(lqx, output_path="results", auxiliary_path="demo", seed=123)

   solver.set_sampler(
       sampler_type="Metropolis Local",
       number_of_chains=32,
       number_of_samples=4096,
   )

   solver.set_optimizer(
       optimizer_type="Adam",
       learning_rate=1e-3,
       diag_shift=1e-2,
   )
   solver.set_network(MLP())

   solver.run(
       2000,
       observables={"volume": lqx.volume(2)},
       live_monitoring=False,
       silent_plot=True,
       silent_print=True,
   )

   solver.export_state(marker="final_state")
   print("Final <C>:", solver.expect(lqx.constraint))

Example 2 - Train a diff-invariant state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # symmetry_list is produced by your graph/symmetry utilities
   symmetry_list = nqx.utils.symmetries.GraphSymmetries(lqx.graph.edges).automorphisms

   solver = nqx.solver.Solver(lqx, output_path="results", auxiliary_path="diff_inv", seed=7)

   solver.set_sampler("Metropolis Local", number_of_chains=512, number_of_samples=8192)
   solver.set_optimizer("SGD", learning_rate=5e-4, diagonal_shift=5e-2)

   solver.set_network(
       MyFlaxModel(),
       diff_invariant=True,
       symmetries=symmetry_list,
   )

   solver.run(
       5000,
       observables={"volume": lqx.volume(1), "area": lqx.area([(0, 1, 0), (0, 2, 0)])},
       callbacks=[checkpoint_and_stop()],
   )

   # the trained state is invariant by construction in this workflow.
   stats = solver.expect(lqx.constraint)
   print("Final constraint residual:", stats)

Example 3 - Resume a simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   solver = nqx.solver.Solver(lqx, output_path="results", auxiliary_path="cluster_run", seed=42)

   # must reattach sampler/optimizer/network identically before resuming
   solver.set_sampler("Metropolis Local", number_of_chains=256, number_of_samples=4096,)
   solver.set_optimizer("Adam", learning_rate=1e-3, diagonal_shift=1e-2)
   solver.set_network(MyFlaxModel())

   # continue from checkpoint
   solver.continue_simulation(
       state_path="results/cluster_run/...FinalState.mpack",
       n_iters=2000,
       callbacks=[checkpoint_and_stop],
       force_load_mpi=True,
   )


.. _multistate_solver:

Multi-state Solver
------------------

:class:`neuralqx.solver.MultiSolver` is the public multi state solver for training several
independent variational states in one optimisation run. It follows the same staged workflow
as :class:`neuralqx.solver.Solver`, but replaces the single
:class:`neuralqx.vqs.MCState` / :class:`neuralqx.driver.VMC` pair with:

* :class:`neuralqx.vqs.MultiMCState`, a container of per-state
  :class:`neuralqx.vqs.MCState` objects, and
* :class:`neuralqx.driver.MultiStateVMC`, a VMC driver that optimises each state against the
  same objective and optionally adds a pairwise orthogonality penalty.

The MT-MH implementation is part of the stable public API. Import it from
``neuralqx.solver``, ``neuralqx.vqs``, and ``neuralqx.driver``. The old
``neuralqx.experimental`` MT-MH imports are no longer part of the API.

When to use it
^^^^^^^^^^^^^^

Use :class:`~neuralqx.solver.MultiSolver` when the result you care about is a set of states,
not one state:

* degenerate or near-degenerate solutions,
* a low-energy or low-constraint residual manifold, not just one state,
* multiple ansätze that should minimise the same operator while remaining distinguishable.

The objective has the form

.. math::

   \mathcal{L}(\theta_1,\dots,\theta_n)
   =
   \sum_i \langle \hat{C} \rangle_{\psi_{\theta_i}}
   +
   \lambda_{\mathrm{ortho}}
   \sum_{i<j} F_{ij},

where :math:`F_{ij}` is the squared fidelity-like overlap estimate between states
:math:`i` and :math:`j`. Set ``lambda_ortho=0.0`` to disable the coupling and train the
states independently inside one driver.

Configuration protocol
^^^^^^^^^^^^^^^^^^^^^^

The setup order is intentionally the same as for the single-state solver:

1. :meth:`~neuralqx.solver.Solver.set_sampler`
2. :meth:`~neuralqx.solver.Solver.set_optimizer`
3. :meth:`~neuralqx.solver.MultiSolver.set_network`
4. :meth:`~neuralqx.solver.Solver.run`

The important difference is :meth:`~neuralqx.solver.MultiSolver.set_network`. It accepts
either one Flax module or a sequence of Flax modules:

.. code-block:: python

   import neuralqx as nqx

   solver = nqx.solver.MultiSolver(lqx, output_path="results", seed=0)
   solver.set_sampler("Metropolis Local", number_of_chains=64, number_of_samples=4096)
   solver.set_optimizer("Adam", learning_rate=1e-3, diagonal_shift=1e-2)

   solver.set_network(
       [Ansatz(), Ansatz(), Ansatz()],
       lambda_ortho=1.0,
   )

   solver.run(1000)

Internally, ``MultiSolver`` builds one :class:`neuralqx.vqs.MCState` per network, wraps
them in a :class:`neuralqx.vqs.MultiMCState`, and constructs a
:class:`neuralqx.driver.MultiStateVMC` driver. Each state receives deterministic but distinct
parameter and sampler seeds derived from the solver seed, so identical model classes do not
start from the same random stream.

Diff-invariant multi-state runs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``diff_invariant`` and ``symmetries`` arguments have the same meaning as in
:meth:`neuralqx.solver.Solver.set_network`, but they are applied to every state network:

.. code-block:: python

   solver.set_network(
       [Ansatz(), Ansatz()],
       diff_invariant=True,
       symmetries=symmetry_list,
       lambda_ortho=0.5,
   )

In this mode the orthogonality penalty is computed between the projected states. That is the
right object when the physical state space is defined modulo graph symmetries.

Expectations
^^^^^^^^^^^^

:meth:`neuralqx.solver.MultiSolver.expect` is explicit about which state is measured:

.. code-block:: python

   all_stats = solver.expect(lqx.constraint)              # list[Stats], one per state
   state_0 = solver.expect(lqx.constraint, state_idx=0)   # one Stats object

Passing ``n_samples=...`` temporarily changes the sample count for the selected state or
states and regenerates samples, matching the warning semantics of the single-state solver.
Passing ``n_chains`` is not supported by the multi-state expectation helper.

Logging, checkpoints, and plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multi-state runs preserve the single-state solver's export discipline, with per-state
metadata added where needed:

* network type and parameter counts are logged per state,
* objective curves are logged under per-state keys such as ``Constraint (state 0)``,
* :meth:`neuralqx.solver.MultiSolver.export_state` writes one multi-state checkpoint payload,
* :meth:`neuralqx.solver.MultiSolver.import_state` requires an already initialised
  ``MultiMCState`` template, so rebuild the same sampler/optimiser/network structure before
  loading from disk.

The underlying :class:`neuralqx.vqs.MultiMCState` also exposes overlap diagnostics:

.. code-block:: python

   mstate = solver.variational_state
   F = mstate.fidelity_matrix(resample=True)
   mstate.print_overlap_matrix(kind="orthogonality", resample=False)

The fidelity matrix is a diagnostic. It is useful for checking whether the learned states
are separating, especially when tuning ``lambda_ortho``.

A minimal example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import flax.linen as nn
   import jax.numpy as jnp
   import neuralqx as nqx

   class Ansatz(nn.Module):
       @nn.compact
       def __call__(self, sigma):
           x = sigma.astype(jnp.float32)
           x = nn.Dense(64)(x)
           x = nn.tanh(x)
           x = nn.Dense(64)(x)
           x = nn.tanh(x)
           out = nn.Dense(1)(x)
           return jnp.sum(out, axis=-1)

   solver = nqx.solver.MultiSolver(lqx, output_path="results", auxiliary_path="mtmh", seed=123)
   solver.set_sampler("Metropolis Local", number_of_chains=64, number_of_samples=4096)
   solver.set_optimizer("Adam", learning_rate=1e-3, diagonal_shift=1e-2)
   solver.set_network([Ansatz(), Ansatz()], lambda_ortho=1.0)

   solver.run(1000, silent_plot=True)

   stats = solver.expect(lqx.constraint)
   print("state 0:", stats[0])
   print("state 1:", stats[1])
