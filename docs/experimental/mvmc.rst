.. _multistate_vmc:

===========================================
Multi-state variational Monte Carlo
===========================================

This page documents the **experimental multi-state variational Monte Carlo** implementation shipped with
neuraLQX. NetKet's drivers and variational states are designed around a *single* variational wavefunction
:math:`\psi_\theta`. In several LQG workflows you instead want to **optimise multiple states at once** in
a single training run:

* to capture **degenerate** solutions of a constraint or Hamiltonian,
* to track a small **low-energy manifold** (ground state + near-degenerate partners),
* or to train a family of ansätze that are pushed to be **mutually orthogonal** while each minimises the
  same energy functional.

In neuraLQX this is exposed as a **drop-in solver**:

* :class:`~neuralqx.experimental.solver.MultiSolver` (a subclass of :class:`~neuralqx.solver.Solver`)

and is implemented in terms of:

* :class:`~neuralqx.experimental.vqs.MultiMCState` (a container of ordinary
  :class:`~neuralqx.vqs.MCState` objects),
* :class:`~neuralqx.experimental.driver.MultiStateVMC` (a VMC driver that couples states through an
  orthogonality penalty).

You should use this when you want to approximate a *set* of solutions rather than a single one, for example
degenerate ground states, a low-energy manifold, or multiple diffeo-invariant representatives.


----------------------------------------
What multi-state VMC does
----------------------------------------

Assume you want :math:`n` variational states :math:`\{|\psi_{\theta_i}\rangle\}_{i=1}^n` that minimise the
same operator :math:`\hat{C}` (in LQG this is typically a constraint operator, but the mechanism is generic).

The multi-state objective used by :class:`~neuralqx.experimental.driver.MultiStateVMC` is:

.. math::

   \mathcal{L}(\theta_1,\dots,\theta_n)
   \;=\;
   \sum_{i=1}^{n} \langle \hat{C} \rangle_{\psi_{\theta_i}}
   \;+\;
   \lambda_{\mathrm{ortho}}
   \sum_{1\le i<j\le n} F_{ij},

where :math:`\lambda_{\mathrm{ortho}}` is a user-controlled strength and :math:`F_{ij}` is the squared
fidelity between states :math:`i` and :math:`j`:

.. math::

   F_{ij}
   =
   \frac{|\langle \psi_{\theta_i} | \psi_{\theta_j} \rangle|^2}
        {\langle \psi_{\theta_i}|\psi_{\theta_i}\rangle\;\langle \psi_{\theta_j}|\psi_{\theta_j}\rangle}
   \in [0,1].

Interpretation:

* The first term pushes every state toward a low value of :math:`\langle \hat{C}\rangle`.
* The second term pushes different states away from each other. If the states are normalised,
  :math:`F_{ij}=|\langle \psi_i|\psi_j\rangle|^2` and the penalty promotes orthogonality.

The preconditioner (SR/QGT) is applied **per state independently** (block-diagonal in state index). This is
an intentional design choice. It retains most of SR's stability while keeping the multi-state update roughly
as cheap as ":math:`n` separate SR steps plus fidelity coupling".



---------------------------------------------------------------
``MultiSolver``: a drop-in solver that runs ``MultiStateVMC``
---------------------------------------------------------------

:class:`~neuralqx.experimental.solver.MultiSolver` is designed to behave like the standard
:class:`~neuralqx.solver.Solver` everywhere it can. You still call:

* ``set_sampler(...)``
* ``set_optimizer(...)``
* ``run(...)``, ``plot_results(...)``, etc.

The key differences are concentrated in **two user-facing APIs**:

* :meth:`~neuralqx.experimental.solver.MultiSolver.set_network` (accepts multiple networks and supports
  automatic wrapping for diffeo projection),
* :meth:`~neuralqx.experimental.solver.MultiSolver.expect` (multi-state aware, with explicit ``state_idx``).

and in multi-state aware checkpointing and final reporting.


``set_network``: single model or per-state models
----------------------------------------------------

Unlike the standard solver, ``MultiSolver.set_network`` accepts either:

* a single Flax module (used for one state), or
* a list/tuple of Flax modules, one per state.

Signature (simplified):

.. code-block:: python

   solver.set_network(
       network,            # FlaxModule or Sequence[FlaxModule]
       diff_invariant=False,
       symmetries=None,
       lambda_ortho=None,
       ...
   )

The solver always normalises to a list internally. If you pass one module, you get :math:`n=1`
(a multi-state container still exists, but only holds one state).

Per-state logging
~~~~~~~~~~~~~~~~~~~~

When you set a multi-network, the solver logs:

* a summary field like ``MultiNetwork(N=k): Net0, Net1, ...``,
* the number of networks,
* and the per-state network type under keys such as
  ``State[0] Network type``, ``State[1] Network type``, etc.

This matters for reproducibility: multi-state runs can combine different ansätze, and you want your runtime
log to reflect that explicitly.

Diff-invariant projection is applied per state (optional)
-------------------------------------------------------------

If the solver is configured for diffeomorphism-invariant simulation (``diff_invariant=True`` or the
corresponding solver flag), ``set_network`` wraps **each** provided network with the same group projector:

.. code-block:: python

   networks = [
       wrap_model(net, symmetries=symmetries, graph=self.graph, characters=None)
       for net in networks
   ]

Important practical implication:

* The multi-state orthogonality penalty is then computed between **projected** states.
* If your "physical" state space is modulo diffeomorphisms, this is the correct object to separate.

Orthogonality strength lives at the solver level
----------------------------------------------------

``lambda_ortho`` can be passed to ``set_network``. If omitted, the solver defaults to ``1.0``.
This value is then forwarded into the :class:`~neuralqx.experimental.driver.MultiStateVMC` instance
during ``initialize_vmc`` (unless overridden there).



------------------------------------------------
``initialize_vmc``: what ``MultiSolver`` builds
------------------------------------------------

Once you have set sampler/optimizer/network, calling:

.. code-block:: python

   solver.initialize_vmc()

does **not** create a single :class:`~neuralqx.vqs.MCState`. Instead it creates:

1. A list of ``MCStates``, one per network.
2. A :class:`~neuralqx.experimental.vqs.MultiMCState` wrapping that list.
3. A :class:`~neuralqx.experimental.driver.MultiStateVMC` driver operating on the MultiMCState.

One ``MCState`` per network (seeded deterministically)
---------------------------------------------------------

For each network ``net[i]`` the solver constructs:

.. code-block:: python

   st = MCState(
       self.sampler,
       net,
       n_samples=self._n_samples,
       is_group_averaged=self.diffeomorphism_invariant_simulation,
       seed=int(self.seed + 1337 * i),
       sampler_seed=int(self.seed // 2 + 7331 * i),
   )

Two seed streams are used:

* ``seed`` controls network parameter initialisation / model state initialisation,
* ``sampler_seed`` controls the Monte Carlo chain evolution.

This ensures that states are not accidentally identical due to shared RNG streams.

Holomorphic detection and SR configuration
----------------------------------------------

The solver tests holomorphicity per state using NetKet's ``is_probably_holomorphic(...)`` and then chooses a
single SR setting:

* ``holomorphic=True`` only if **all** states appear holomorphic,
* otherwise ``holomorphic=False``.

This conservative choice avoids enabling holomorphic SR when any state violates the assumptions.

``MultiStateVMC`` driver construction
-----------------------------------------

The driver is created as:

.. code-block:: python

   self._vmc_driver = MultiStateVMC(
       variational_state=self._variational_state,     # MultiMCState
       hamiltonian=self.lqx.constraint,               # operator minimised by VMC
       optimizer=self.optimizer,
       preconditioner=SR(
           diag_shift=self.diagonal_shift,
           holomorphic=holomorphic,
           solver=self.preconditioner_solver,
       ),
       lambda_ortho=lambda_ortho,
   )

Note that ``hamiltonian`` is the model's constraint operator (the same thing the standard solver minimises),
but the driver permits lists of operators too.

Parameter counting and dim(H) ratios are multi-state aware
--------------------------------------------------------------

After building all ``MCState``s, the solver logs:

* total number of parameters across all networks,
* per-state number of parameters,
* and parameter-to-Hilbert-dimension ratios both total and per-state.

For large Hilbert spaces this is a useful diagnostic as it gives a rough sense of how expressive the ansatz is
relative to the discrete configuration space being represented.



----------------------------------------------------------
``MultiSolver.expect``: multi-state expectation API
----------------------------------------------------------

In a single-state solver, ``expect(O)`` returns one ``Stats`` object.

In a multi-state solver, you often want either:

* the expectation value for **all** states (to compare them),
* or for one specific state (for reporting or for a penalty term you compute externally).

``MultiSolver.expect`` supports both via ``state_idx``.

Signature
-----------------------

.. code-block:: python

   stats = solver.expect(operator, state_idx=None, n_samples=None, ...)

Behaviour
-----------

* ``state_idx=None`` (default)
    Returns a ``list[Stats]`` of length ``n_states``.
* ``state_idx=i``
    Returns a single ``Stats`` for that state.

Example:

.. code-block:: python

   # all states
   stats_all = solver.expect(C)                 # list[Stats]
   mean0 = float(stats_all[0].mean)

   # one state
   stats0 = solver.expect(C, state_idx=0)       # Stats
   mean0 = float(stats0.mean)

Optional temporary sample override
------------------------------------

``expect(..., n_samples=...)`` temporarily overrides the selected state(s) sample count, forces resampling,
and then restores the original sample count.

This is intended for "one-off higher accuracy measurements" without permanently changing your run settings.

.. warning::

   Temporary ``n_samples`` overrides discard existing cached samples and regenerate Monte Carlo samples.
   That is the correct behaviour for measurement accuracy, but you should not call it repeatedly inside an
   optimisation loop unless you intend to change the sampling workload.

Unsupported arguments
-----------------------

The implementation explicitly rejects ``n_chains`` overrides and warns that several "alias mode" arguments are
ignored (``use_same_variables``, ``use_same_sampler``, and arbitrary extra kwargs). This is intentional:
multi-state expectation has to be explicit about which state(s) you are evaluating.



--------------------------------------------------------------
``MultiMCState``: how it works and what it offers
--------------------------------------------------------------

:class:`~neuralqx.experimental.vqs.MultiMCState` is a minimal but crucial abstraction.

It is **not** a new probabilistic model. It is:

* a container around a list of ordinary :class:`~neuralqx.vqs.MCState`,
* plus overlap/orthogonality diagnostics implemented in a way that is MPI-safe and robust to NetKet version
  differences.

Basic API surface
---------------------

Construction:

.. code-block:: python

   mstate = MultiMCState([s0, s1, s2])

Core attributes:

``mstate.states``
  List of member :class:`~neuralqx.vqs.MCState`.

``mstate.n_states``
  Number of member states.

``mstate.hilbert``
  Hilbert space (taken from the first state).

``mstate.parameters``
  Returns a list of parameter pytrees, assigning sets parameters per member state.

Sampling control:

``mstate.reset()``
  Calls ``reset()`` on every member MCState.

Expectation:

``mstate.expect(O)``
  Returns ``[s.expect(O) for s in states]``.

This simple interface is enough for the driver and for user code.

Overlap and orthogonality diagnostics
-----------------------------------------

``MultiMCState`` provides three related utilities:

``fidelity_matrix(...)``
  Estimates :math:`F_{ij}` for all pairs, returning an :math:`n\times n` numpy array.
  Can optionally return an estimated error matrix as well.

``overlap_matrix(kind=...)``
  Derived matrices:

  * ``kind="fidelity"`` returns :math:`F_{ij}`
  * ``kind="overlap"`` returns :math:`\sqrt{F_{ij}}` (magnitude only)
  * ``kind="orthogonality"`` returns :math:`1 - F_{ij}`

``print_overlap_matrix(...)``
  Pretty-print on master rank only.

These diagnostics exist because a multi-state run is only meaningful if the states actually separate.
You should use them routinely when tuning ``lambda_ortho`` and samplers.

Fidelity estimator used by diagnostics (joint sampling)
-----------------------------------------------------------

For states :math:`i` and :math:`j`, MultiMCState estimates the squared fidelity:

.. math::

   F_{ij}
   =
   \frac{|\langle \psi_i|\psi_j\rangle|^2}
        {\langle \psi_i|\psi_i\rangle\;\langle \psi_j|\psi_j\rangle}.

The implementation uses the same joint-sampling trick as the training penalty, but without gradients.

Let each state's sampler draw samples from:

.. math::

   p_i(\sigma) \propto |\psi_i(\sigma)|^{m_i},

where ``machine_pow`` :math:`m_i` is often 2.

Given two sample batches :math:`\{\sigma_x\}` from state :math:`i` and :math:`\{\sigma_y\}` from state
:math:`j`, the estimator builds a joint batch of concatenated configurations:

.. math::

   \sigma_{xy} = (\sigma_x,\sigma_y).

It defines a product log-density:

.. math::

   \log p_{ij}(\sigma_x,\sigma_y)
   =
   m_i \Re[\log\psi_i(\sigma_x)] + m_j \Re[\log\psi_j(\sigma_y)] + \text{const.}

and a local estimator:

.. math::

   f_{\mathrm{loc}}(\sigma_x,\sigma_y)
   =
   \exp\Big(
     \log\psi_j(\sigma_x)
     + \log\psi_i(\sigma_y)
     - \log\psi_i(\sigma_x)
     - \log\psi_j(\sigma_y)
   \Big).

For :math:`m_i=m_j=2`, its expectation under the product measure equals the standard squared fidelity.
The implementation takes the real part and clamps into :math:`[0,1]` for presentation.

Master-only printing with MPI-safe computation
--------------------------------------------------

The printing utilities are careful to compute on all processes (so MPI reductions inside NetKet/JAX remain
valid), but only emit human-readable output on the global master process.

This avoids duplicate console spam in MPI runs while preserving correct statistics.

Error handling and backward compatibility
---------------------------------------------

The serializer/deserializer is intentionally defensive:

* if an older checkpoint stored a single ``MCState`` dict (no ``"states"`` key), it is automatically wrapped
  as a single-state ``MultiMCState`` when loading,
* if the number of saved states does not match the current template (constructed by ``initialize_vmc``),
  loading fails with an explicit error, because parameter trees and model states must align state-by-state.



----------------------------------------------------------
``MultiStateVMC``: what the driver actually changes
----------------------------------------------------------

:class:`~neuralqx.experimental.driver.MultiStateVMC` is where coupling happens. It modifies the standard
VMC flow in four key places:

Energy and gradients are computed per state
-----------------------------------------------

For each member MCState :math:`i`:

.. math::

   E_i = \langle \hat{C} \rangle_{\psi_i}, \qquad g_i = \nabla_{\theta_i} E_i.

This uses the existing MCState method ``expect_and_grad`` and therefore inherits all the stability/estimator
behaviour of neuraLQX's single-state implementation.

Orthogonality penalty gradients are computed pairwise
---------------------------------------------------------

For each pair :math:`i<j`, the driver computes:

* :math:`F_{ij}`
* and gradients w.r.t. both parameter sets: :math:`\nabla_{\theta_i}F_{ij}` and :math:`\nabla_{\theta_j}F_{ij}`.

Those are added into the corresponding per-state gradients scaled by ``lambda_ortho``.

Gradient computation is performed by a jitted function that:

* evaluates :math:`F_{ij}` using the joint-sampling estimator above,
* differentiates it with respect to the combined parameter PyTree using VJP (with conjugate rules),
* averages gradients across MPI ranks.

SR is applied block-diagonally
----------------------------------

After gradients are accumulated, the driver calls the preconditioner separately for each state:

.. math::

   \Delta\theta_i = \mathcal{P}_i(g_i),

where :math:`\mathcal{P}_i` is SR/QGT-based if configured (otherwise identity). There are no cross-state
blocks :math:`\mathcal{P}_{ij}`.

``estimate()`` is overridden to flatten per-state observables
----------------------------------------------------------------

The driver overrides ``estimate`` so that callbacks and runtime logs can treat per-state observables as
separate scalar curves.

If an observable produces a leaf ``[Stats_0, ..., Stats_{n-1}]``, the driver emits keys:

* ``"ObservableName (state 0)" -> Stats_0``
* ``"ObservableName (state 1)" -> Stats_1``
* ...

For nested observables (dicts-of-dicts), the path becomes a slash-separated prefix. This makes multi-state
observables compatible with NetKet's loggers/monitors without rewriting the logging stack.



----------------------------------------------------------
Checkpointing: exporting and importing ``MultiMCState``
----------------------------------------------------------

``MultiSolver`` implements multi-state checkpointing that mirrors the single-state solver, but stores one
serialised MCState dict per sub-state.

Export
------

.. code-block:: python

   solver.export_state(marker="after_500")

This produces a file whose payload (conceptually) is:

.. code-block:: python

   {
     "kind": "MultiMCState",
     "n_states": ...,
     "states": [ serialize_MCState(s0), serialize_MCState(s1), ... ],
     "mpi_info": ...
   }

Export is MPI-safe. A barrier is used before/after and only the global master writes to disk.

Import
------

.. code-block:: python

   solver.initialize_vmc()                 # creates the template state structure
   solver.import_state("...mpack")         # loads into that structure

Import requires an existing template ``MultiMCState`` because it needs to know how many states exist and how to
reconstruct each ``MCState``'s structure. If the checkpoint contains a different number of states than the current
template, import fails with a clear message.

Backward compatibility
------------------------

If a checkpoint contains a single ``MCState`` dict (older format), import automatically wraps it as a
single-state ``MultiMCState``.



------------------------------------------------------------
Final constraint reporting and plotting in ``MultiSolver``
------------------------------------------------------------

The standard solver records a single "final constraint estimate". ``MultiSolver`` overrides this to record:

* the full list of per-state final ``Stats`` objects,
* a per-state scalar mean value (coerced to a stable scalar representation),
* and logs per-state entries under unique keys like:

  * ``State[0] Network result``
  * ``State[1] Network result``

For plotting, ``MultiSolver``'s ``plot_results`` is multi-state aware:

* If the runtime log contains per-state keys (e.g. ``Constraint (s0)``, ``Constraint (s1)``, ...),
  it plots **one curve per state**.
* Otherwise, it falls back to a single constraint curve.

This means you can run multi-state training and get a meaningful visual summary without special-case plotting
code in your analysis scripts.



-----------------------------------------
Usage examples (``MultiSolver``-first)
-----------------------------------------

Same model architecture, multiple states
--------------------------------------------

.. code-block:: python

   from neuralqx.experimental.solver import MultiSolver
   import flax.linen as nn
   import jax.numpy as jnp

   class MyAnsatz(nn.Module):
       @nn.compact
       def __call__(self, sigma):
           x = sigma.astype(jnp.float32)
           x = nn.Dense(64)(x); x = nn.tanh(x)
           x = nn.Dense(64)(x); x = nn.tanh(x)
           out = nn.Dense(2)(x)
           return out[..., 0] + 1j * out[..., 1]

   lqx = ...                 # model interface providing lqx.constraint, graph, hilbert, ...
   solver = MultiSolver(lqx, output_path="runs/demo", seed=0)

   solver.set_sampler(...)
   solver.set_optimizer(...)

   # provide 3 networks (same class, separate instances)
   solver.set_network([MyAnsatz(), MyAnsatz(), MyAnsatz()], lambda_ortho=1.0)

   solver.run(500)

   # measure constraint per state
   stats = solver.expect(lqx.constraint)       # list[Stats]
   print(stats[0], stats[1], stats[2])

Diffeomorphism-invariant multi-state run
--------------------------------------------

.. code-block:: python

   symmetries = ...  # list of graph automorphisms / permutations
   solver.set_network(
       [MyAnsatz(), MyAnsatz()],
       diff_invariant=True,
       symmetries=symmetries,
       lambda_ortho=0.5,
   )

   solver.run(500)

The solver wraps each network with the same projector using ``wrap_model(...)``.
The orthogonality penalty then separates the resulting projected states.

Measuring only one state (state_idx)
----------------------------------------

.. code-block:: python

   st0 = solver.expect(lqx.constraint, state_idx=0)
   st1 = solver.expect(lqx.constraint, state_idx=1)
   print("state0", st0.mean, "state1", st1.mean)

Overlap diagnostics
----------------------

.. code-block:: python

   # MultiMCState is stored at solver.variational_state
   mstate = solver.variational_state

   F = mstate.fidelity_matrix(resample=True)
   mstate.print_overlap_matrix(kind="orthogonality", resample=False)



---------------------------------------------------
Practical tuning notes
---------------------------------------------------

Choosing ``lambda_ortho``
---------------------------

If ``lambda_ortho`` is too small, different states can collapse onto the same minimum and you effectively
waste extra networks. If it is too large, the penalty can dominate and slow down minimisation of the target
constraint.

A robust workflow:

* start with a modest value (order 0.1–1),
* monitor the orthogonality matrix :math:`1-F_{ij}`,
* increase if off-diagonal entries remain small (high overlap),
* decrease if energies stop improving or become excessively noisy.

Sampler consistency across states
-----------------------------------

The orthogonality estimator uses the sample batches cached inside each ``MCState``. If one state mixes poorly or
uses very different sampling settings, the fidelity penalty will become noisy and can destabilise training.

In practice, keep sampler settings aligned across states unless you have a principled reason to diverge.

machine_pow and "true" fidelity
----------------------------------

The fidelity estimator matches the standard quantum fidelity when each sampler targets
:math:`p(\sigma)\propto|\psi(\sigma)|^2` (machine_pow = 2). If you deviate from that, the overlap functional
changes accordingly. For orthogonality enforcement in the usual quantum sense, keep machine_pow = 2 for all
states.
