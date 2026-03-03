.. _stmh_vmc:

=========================================================
Single-trunk multi-head variational Monte Carlo (ST-MH)
=========================================================

This page documents the **experimental single-trunk multi-head (ST-MH) variational Monte Carlo**
machinery shipped with neuraLQX for joint optimisation of multiple states using a **shared**
parameter set.

Where :class:`~neuralqx.experimental.solver.MultiSolver` (Multi-Trunk style in this codebase) optimises **N independent networks**
packaged in a :class:`~neuralqx.experimental.vqs.MultiMCState` with a coupling penalty, the Single-Trunk implementation optimises **one shared Flax model** with:

* a shared **trunk** (feature extractor),
* :math:`K` lightweight **heads** (one per target state),
* one shared optimiser state,
* one shared parameter update per iteration.

This is useful when the target manifold is expected to share internal structure, for example

* degenerate or nearly degenerate eigenstates,
* multiple physical representatives in the same constrained sector,
* low-energy manifolds where a shared representation improves sample efficiency and stability.

In neuraLQX this is exposed through:

* :class:`~neuralqx.experimental.solver.STMultiSolver` (drop-in solver interface),
* :class:`~neuralqx.experimental.driver.SingleTrunkMultiHeadVMC` (shared-gradient VMC driver),
* :class:`~neuralqx.experimental.vqs.STMultiMCState` (shared-parameter state container),
* :class:`~neuralqx.experimental.nn.SingleTrunkMultiHeadLogPsi` and head views (Flax wrapper layer).

The ST-MH stack is intentionally designed to keep **ordinary NetKet/neuraLQX MCState objects** in
the loop so sampling and estimator behaviour remain familiar, while changing the parameter geometry
and update rule to match the shared-parameter ansatz.



----------------------------------------
What ST-MH VMC does
----------------------------------------

Assume you want :math:`K` variational states
:math:`\{|\psi_k(\Theta)\rangle\}_{k=1}^{K}` where all states depend on the **same** parameter set
:math:`\Theta` through a shared trunk and head-specific readout.

The ST-MH driver minimises the objective

.. math::

   C(\Theta)
   =
   \sum_{k=1}^{K} w_k E_k(\Theta)
   +
   \lambda_{\mathrm{ortho}}
   \sum_{1 \le i < j \le K} F_{ij}(\Theta),

with

.. math::

   E_k(\Theta) = \langle \hat{C}_k \rangle_{\psi_k(\Theta)},

and pairwise fidelity penalty

.. math::

   F_{ij}(\Theta)
   =
   \frac{|\langle \psi_i(\Theta) | \psi_j(\Theta)\rangle|^2}
        {\langle \psi_i(\Theta)|\psi_i(\Theta)\rangle
         \langle \psi_j(\Theta)|\psi_j(\Theta)\rangle}.

Here

* :math:`w_k` are user-provided ``energy_weights`` that are internally normalised to sum to one
* :math:`\lambda_{\mathrm{ortho}}` is the orthogonality penalty strength
* :math:`\hat{C}_k` is either a shared operator or a per-head operator if a list is supplied

Interpretation:

* The weighted energy term encourages each head to minimise its target operator.
* The fidelity term discourages head collapse onto the same state.
* The optimisation variable is **one shared parameter pytree**.

This is the central difference from MT-MH. In MT-MH each state has its own parameter set
:math:`\theta_i`. In ST-MH every gradient contribution must be accumulated into a single
:math:`\nabla_\Theta C`.



-------------------------------------------------------------
Architecture overview: wrapper, state, driver, solver
-------------------------------------------------------------

The ST-MH implementation is split into four layers.

1. **Flax wrapper layer**
   Helpers in :mod:`neuralqx.experimental.nn.projectors.stmh` which convert an arbitrary feature-producing trunk into a multi-head log-wavefunction model.

2. **Head-view compatibility models**
   Expose one selected head as a standard scalar-output model so each head can be handed to
   :class:`~neuralqx.vqs.MCState`.

3. **Shared-parameter state container**
   Groups the head ``MCState`` objects but exposes a **single** ``parameters`` pytree to the driver.

4. **ST-MH VMC driver**
   Aggregates all energy and orthogonality gradients into one shared update direction.

5. **ST-MH solver**
   Provides the standard neuraLQX user workflow and checkpointing interface.

A key design goal is minimal disruption of the existing single-state and MT-MH code path.
Sampling still happens through ordinary ``MCState`` objects and the new logic is concentrated in
the shared wrapper, shared container, and shared-gradient driver.



-----------------------------------------------------------------------
ST-MH ansatz wrapper: ``SingleTrunkMultiHeadLogPsi`` and head views
-----------------------------------------------------------------------

The wrapper functionalities in :mod:`neuralqx.experimental.nn.projectors.stmh` constructs an ST-MH ansatz from a user-defined **trunk** network.

High-level form
----------------

Let :math:`f_\phi(x)` denote the trunk output (features for configuration :math:`x`). The wrapper
applies one affine readout per head. For the default complex output case,

.. math::

   \log \psi_k(x)
   =
   a_k(f_\phi(x)) + i\,b_k(f_\phi(x)),

where :math:`a_k` and :math:`b_k` are implemented as dense layers. All heads share the same trunk
features and differ only in the final linear maps.

The wrapper can also produce real outputs if ``complex_logpsi=False``.

What the trunk should return
-----------------------------

The preferred convention is a batch-first feature tensor such as

* ``(batch, hidden_dim)``
* or any batch-first shape ``(batch, ...)`` that can be flattened to ``(batch, F)``

The wrapper accepts several output styles and can extract features from

* a direct tensor return
* a dict return (for example ``{"features": ...}``)
* a tuple/list return (first element by default)

This allows you to reuse existing Flax modules without rewriting them.

If the trunk returns a scalar per sample with shape ``(batch,)`` the wrapper lifts it to
``(batch, 1)``. This runs correctly but yields only a one-dimensional shared feature space, which
is usually too restrictive for a useful ST-MH ansatz.

Public classes and helpers
--------------------------------------------

The wrapper module in :mod:`neuralqx.experimental.nn.projectors.stmh` currently provides:

``SingleTrunkMultiHeadLogPsi``
  Main ST-MH wrapper that outputs all heads or a selected head.

``STMHHeadView``
  Compatibility wrapper that exposes one head as a standard scalar-output Flax module. This is the
  object you pass into ``MCState``.

``STMHAllHeadsView``
  Thin wrapper that always returns all heads, shape ``(batch, K)``.

``wrap_trunk_as_stmh(...)``
  Convenience constructor around ``SingleTrunkMultiHeadLogPsi``.

``make_stmh_head_models(...)``
  Returns a list of ``STMHHeadView`` objects, one per head.

``SingleTrunkMultiHeadLogPsi`` constructor parameters
------------------------------------------------------

The wrapper is intentionally generic. The most important parameters are:

``trunk``
  Any Flax module that maps basis configurations to features.

``n_heads``
  Number of heads :math:`K`.

``latent_dim``
  Optional projection width. If the extracted feature width differs from ``latent_dim`` the wrapper
  inserts a learned projection layer ``trunk_proj`` before the heads.

``trunk_output``
  How the trunk return value is interpreted. Supported modes are

  * ``"auto"``
  * ``"features"``
  * ``"dict"``
  * ``"tuple"``

``complex_logpsi``
  If ``True`` the wrapper uses separate real and imaginary head Dense layers and returns complex
  log-amplitudes.

``flatten_features``
  If ``True`` all trailing dimensions after the batch axis are flattened.

``features_key`` and ``tuple_index``
  Selection controls for dict or tuple trunk outputs.

``dtype`` and ``param_dtype``
  Forwarded to the internal Dense layers.

Call behaviour and output shapes
---------------------------------

The wrapper call signature supports two important keyword arguments:

* ``head``
* ``return_features``

When ``head=None`` (default), the output shape is ``(batch, K)`` and contains all heads.

When ``head=i`` the wrapper returns the selected head only with shape ``(batch,)``. This is the
mode used by ``STMHHeadView`` and by the solver compatibility wrappers.

When ``return_features=True`` the wrapper returns ``(out, feats)`` where ``feats`` is the extracted
and optionally projected feature matrix used by the heads. This is useful for debugging and for
confirming that the trunk output shape is what you expect.

Examples
--------

Wrap a trunk directly:

.. code-block:: python

   stmh = SingleTrunkMultiHeadLogPsi(
       trunk=MyTrunk(...),
       n_heads=4,
       complex_logpsi=True,
   )

   y_all = stmh.apply(params, sigma)          # shape (batch, 4)
   y_0   = stmh.apply(params, sigma, head=0)  # shape (batch,)

Create head-view models for ``MCState``:

.. code-block:: python

   head_models = make_stmh_head_models(stmh, n_heads=4)

   # each head_models[i] is scalar-output and can be passed to MCState
   s0 = MCState(sampler, head_models[0], n_samples=2048)
   s1 = MCState(sampler, head_models[1], n_samples=2048)

Notes on diffeomorphism-invariant wrapping
-------------------------------------------

For ST-MH, the recommended place to apply diffeomorphism projection in solver workflows is at the
**head-view level** during ``STMultiSolver.initialize_vmc``. This ensures each head-specific scalar
model is projected exactly as in standard single-state usage while still sharing the same underlying
parameter tree.



--------------------------------------------------------------------
``STMultiMCState``: shared-parameter container semantics
--------------------------------------------------------------------

:class:`~neuralqx.experimental.vqs.STMultiMCState` is the key abstraction that makes ST-MH
compatible with the existing VMC driver interface.

It is **not** a probabilistic model and it is **not** a replacement for ``MCState``. It is:

* a container around a list of ordinary head-specific ``MCState`` objects
* a synchronisation layer that enforces one canonical shared parameter pytree
* a compatibility object that exposes a single ``parameters`` property to the driver

Why a new state container is needed
------------------------------------

The current MT-MH container ``MultiMCState`` is correct for independent networks because it returns a list
of parameter pytrees. That is exactly what MT-MH needs.

ST-MH is different. The optimiser must see **one** parameter pytree because trunk and head weights
live in a joint parameter set and must be updated together. If the driver saw a list of pytrees it
would perform the wrong parameter geometry and the wrong optimiser update semantics.

Construction
-------------

Create one ``MCState`` per head and wrap them:

.. code-block:: python

   shared_state = STMultiMCState(
       [head_state_0, head_state_1, head_state_2],
       canonical_state=0,
       sync_model_state=False,
   )

A convenience function ``make_shared_stmh_state(...)`` is also provided.

Compatibility checks performed at construction
-----------------------------------------------

The constructor validates:

* at least one head state exists
* all head states share the same Hilbert space
* all head states have the same parameter tree structure (same treedef)

If any head has an incompatible parameter tree, construction fails immediately. This is important
because the ST-MH gradient aggregation assumes every head view is a different view of the **same**
shared model.

Canonical parameter source
---------------------------

``canonical_state`` selects the head ``MCState`` whose parameters are treated as the source of truth.

* ``shared_state.parameters`` returns ``states[canonical_state].parameters``
* assigning ``shared_state.parameters = pars`` broadcasts the same pytree to all heads

In normal use the canonical choice does not change the mathematics because all head states are kept
synchronised. It mainly affects which embedded ``MCState`` object is used as the direct parameter
reference and is relevant for debugging and checkpoint restoration.

Optional model-state synchronisation
-------------------------------------

By default only ``params`` are synchronised.

If ``sync_model_state=True`` the container also attempts to broadcast ``model_state`` from the
canonical head to all other heads whenever parameters are broadcast. This is provided for models
that use mutable Flax collections (for example BatchNorm statistics).

Keep this disabled unless you explicitly need it and understand the implications. Mutable state can
become subtle when each head samples a different distribution.

Core API surface
-----------------

The container exposes a driver-friendly subset of ``MCState``-like properties and methods.

Attributes and properties
~~~~~~~~~~~~~~~~~~~~~~~~~

``states``
  The underlying list of head-specific ``MCState`` objects.

``n_states``
  Number of heads.

``parameters``
  Shared parameter pytree property. Setter broadcasts to all heads.

``hilbert``
  Shared Hilbert space object.

``canonical_state``
  Index of the canonical parameter source.

``samples``
  Returns a list of the current sample batches, one per head.

``n_samples`` and related sampling controls
  Getters return per-head lists. Setters broadcast one scalar value to all heads.

Methods
~~~~~~~

``broadcast_from_canonical()``
  Copies canonical parameters to all head states. Optionally also copies model state.

``reset()``
  Broadcasts canonical parameters and then calls ``reset()`` on every head state.

``sample(...)``
  Broadcasts canonical parameters and samples every head state.

``expect(O)``
  Broadcasts canonical parameters and returns ``[st.expect(O) for st in states]``.

Why ``reset()`` broadcasts first
---------------------------------

The ST-MH driver accumulates gradients from multiple head ``MCState`` objects. If any head retained
stale parameters from a previous step, the objective and gradient would be inconsistent.

For this reason the container always synchronises from the canonical state before reset and sample
operations.

Practical note on head parameter drift
---------------------------------------

Each head ``MCState`` owns its own Python object and internal caches. Even though they conceptually
represent the same shared model, they can temporarily diverge in their local parameter storage if you
manually mutate one head. The container intentionally corrects this by broadcasting from the canonical
state before all driver-critical operations.



--------------------------------------------------------------
``SingleTrunkMultiHeadVMC``: shared-gradient ST-MH driver
--------------------------------------------------------------

:class:`~neuralqx.experimental.driver.SingleTrunkMultiHeadVMC` is the component that changes the VMC
update rule from MT-MH semantics to ST-MH semantics.

It subclasses the base neuraLQX ``VMC`` driver but requires a
:class:`~neuralqx.experimental.vqs.STMultiMCState` as ``variational_state``.

Constructor
------------

Simplified signature:

.. code-block:: python

   driver = SingleTrunkMultiHeadVMC(
       variational_state=shared_state,             # STMultiMCState
       hamiltonian=H or [H0, H1, ...],
       optimizer=opt,
       preconditioner=...,                         # identity_preconditioner or SR(...)
       lambda_ortho=1.0,
       energy_weights=None,
       enforce_machine_pow_2=True,
       preconditioner_state_index=0,
   )

Key constructor parameters
---------------------------

``variational_state``
  Must be ``STMultiMCState``. Passing MT-MH ``MultiMCState`` raises ``TypeError``.

``hamiltonian``
  Either one operator shared across all heads or a sequence of operators of length ``n_states``.

``lambda_ortho``
  Weight of the pairwise fidelity penalty.

``energy_weights``
  Optional sequence of weights for per-head energy terms. The driver normalises them internally to
  sum to one. If omitted, equal weights are used.

``enforce_machine_pow_2``
  If ``True`` (default), the driver raises when any head sampler has ``machine_pow != 2``. This
  protects the standard fidelity interpretation of the penalty.

``preconditioner_state_index``
  Selects the head state whose geometry is used when the preconditioner is not identity. This matters
  when using SR/QGT and is discussed in detail below.


What changes relative to ``MultiStateVMC`` (MT-MH)
---------------------------------------------------

The MT-MH driver is correct for **independent parameter sets** and preconditions each state gradient
separately. That is not the right update for ST-MH.

The ST-MH driver instead:

* computes per-head energy gradients with respect to the **same shared parameter pytree**
* computes pairwise orthogonality gradients with respect to the **same shared parameter pytree**
* sums all contributions into one global gradient
* applies the preconditioner **once**
* returns one shared update direction ``dp``

This is the defining ST-MH driver behaviour.


Logging behaviour and observable flattening
--------------------------------------------

The driver extends logging with ST-MH-specific fields:

* ``Constraint (head i)`` or more generally ``<loss_name> (head i)`` for per-head energies
* ``energy/weighted_sum_mean``
* ``orthogonality/pair_fidelity_sum``
* ``fidelity(i,j)`` for each pair

The ``estimate(...)`` override flattens per-head observable lists into logging-friendly keys, similar
to the MT-MH implementation. For an observable returning ``[Stats_0, ..., Stats_{K-1}]``, the driver
emits:

* ``ObservableName (head 0)``
* ``ObservableName (head 1)``
* ...
* optionally ``ObservableName/sum_mean`` if scalar means can be extracted

This keeps callbacks and loggers compatible with minimal changes.



---------------------------------------------------
``machine_pow`` and fidelity interpretation
---------------------------------------------------

The ST-MH orthogonality penalty is implemented using a joint-sampling overlap estimator that has a
clean fidelity interpretation in the usual quantum setting only under a specific condition.

Recommended and default regime
-------------------------------

Use ``machine_pow = 2`` for all head samplers.

When every head samples from a distribution proportional to :math:`|\psi|^2`, the estimator used by
the orthogonality kernel matches the normalized squared fidelity and the penalty is directly
interpretable as orthogonality enforcement in the standard quantum sense.

What the driver enforces
-------------------------

By default ``SingleTrunkMultiHeadVMC`` sets ``enforce_machine_pow_2=True``.

This means the driver checks all head samplers and raises a ``ValueError`` if any head has a
different ``machine_pow``. This is a safety mechanism for correctness of interpretation.

What happens if you disable the check
--------------------------------------

If ``enforce_machine_pow_2=False`` the code still runs. The pairwise penalty becomes a more general
overlap-like surrogate defined by the chosen sampling powers.

This can still act as a useful repulsion regulariser between heads, but

* its scale changes with the sampler exponents
* its interpretation as normalized fidelity is lost
* mixed powers across heads are harder to reason about

The driver clips logged pairwise values into ``[0,1]`` for presentation only. That clipping is not a
proof that the estimator remains a true fidelity under nonstandard ``machine_pow`` values.

Practical recommendation
-------------------------

Unless you intentionally study alternative overlap penalties, keep ``machine_pow = 2`` for every head
and leave ``enforce_machine_pow_2=True``.



---------------------------------------------
Using SR / QGT preconditioning in ST-MH
---------------------------------------------

This is one of the most important conceptual differences between MT-MH and the current ST-MH driver.

Euclidean gradient descent path (exact)
----------------------------------------

If the preconditioner is ``identity_preconditioner`` then the ST-MH driver performs the exact
Euclidean gradient descent direction on the sampled objective defined by the weighted energy and
orthogonality terms.

This is the cleanest baseline for validating a new ST-MH ansatz.

SR / QGT path (current implementation is an approximation)
-----------------------------------------------------------

If you pass an SR/QGT preconditioner, the driver applies it using the geometry of **one selected head
state** only:

.. code-block:: python

   ref_state = self.states[preconditioner_state_index]
   dp = preconditioner(ref_state, global_grad, step)

This is a practical approximation. It often works well enough to gain SR-like stabilisation, but it
is not the exact natural-gradient step for the full ST-MH objective.

Why it is approximate
----------------------

The true ST-MH objective couples all heads through both

* per-head energy terms
* pairwise orthogonality terms

A strict natural-gradient treatment would require a geometry that represents the tangent space of the
full shared multi-head objective, including the way all heads depend on the same trunk features.

The current implementation reuses the existing single-head state geometry for convenience and
compatibility. This captures the parameter structure of the shared model and one head view, but it
does not build a full combined ST-MH QGT.


Future extension direction
---------------------------

A dedicated ST-MH preconditioner could construct a combined geometry from multiple heads and possibly
from weighted objective terms. The current driver is designed so such a preconditioner can be plugged
in later without changing the outer solver interface.



---------------------------------------------------------
``STMultiSolver``: drop-in solver for shared ST-MH runs
---------------------------------------------------------

:class:`~neuralqx.experimental.solver.STMultiSolver` mirrors the standard solver workflow and the MT-MH
``MultiSolver`` workflow where possible, but wires them to the ST-MH state and driver stack.

You still call:

* ``set_sampler(...)``
* ``set_optimizer(...)``
* ``set_network(...)``
* ``initialize_vmc()``
* ``run(...)``, ``expect(...)``, ``plot_results(...)``

The key difference is that ``set_network`` expects a **shared ST-MH base model** and the solver builds
head-specific ``MCState`` views internally.

``set_network`` for ST-MH
--------------------------

Simplified signature:

.. code-block:: python

   solver.set_network(
       network,                    # shared ST-MH base model
       n_heads=None,               # inferred from network.n_heads if possible
       head_models=None,           # optional explicit per-head scalar models
       diff_invariant=False,
       symmetries=None,
       lambda_ortho=None,
       canonical_state=0,
       sync_model_state=False,
   )

Key behaviours
~~~~~~~~~~~~~~~

``network``
  The shared ST-MH base Flax model. In normal use this is ``SingleTrunkMultiHeadLogPsi(...)`` or a
  compatible model that supports ``model(x, head=int)``.

``n_heads``
  Number of heads. If omitted, the solver tries to read ``network.n_heads``.

``head_models``
  Optional explicit scalar-output head models. If not provided, the solver constructs local head-view
  wrappers that call ``base(x, head=i)``. This makes the solver robust even when the exported wrapper
  class path is not final.

``lambda_ortho``
  Stored at the solver level and forwarded during ``initialize_vmc`` unless overridden there.

``canonical_state`` and ``sync_model_state``
  Passed to ``STMultiMCState`` construction.

Logging and reproducibility metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solver logs ST-MH specific metadata such as:

* ``Network type = SingleTrunkMultiHead(N=K): <BaseClassName>``
* number of heads
* optional per-head view model types when explicit ``head_models`` are supplied

This is important because ST-MH runs can be reproduced only if the shared base model and head count
are known.

Diffeomorphism-invariant wrapping in ST-MH
-------------------------------------------

If diffeomorphism-invariant simulation is enabled, the solver applies ``wrap_model(...)`` to each
**head view** during ``initialize_vmc``. This mirrors the single-state workflow and ensures the
projected objects used by ``MCState`` remain scalar-output NQS models.

The orthogonality penalty is then computed between projected head states.


``STMultiSolver.expect``: multi-head expectation API
--------------------------------------------------

Like ``MultiSolver.expect``, the ST-MH solver supports measuring all heads or one selected head.

Simplified signature:

.. code-block:: python

   stats = solver.expect(operator, state_idx=None, n_samples=None, ...)

Behaviour
~~~~~~~~~~

* ``state_idx=None`` returns ``list[Stats]`` of length ``n_heads``
* ``state_idx=i`` returns one ``Stats`` object for head ``i``

Temporary sample override
~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``n_samples`` is provided, the solver temporarily changes the selected head state's sample count,
computes the expectation, and restores the original value.

This is intended for one-off high-accuracy measurements and is not meant for use inside the main
optimisation loop.

Unsupported or ignored arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ST-MH solver currently rejects explicit ``n_chains`` overrides in this alias mode and warns that
``use_same_variables``, ``use_same_sampler``, and extra kwargs are ignored. This mirrors the explicit
behaviour of the MT-MH solver API and avoids ambiguous semantics.

Final constraint reporting
---------------------------

``STMultiSolver`` overrides final reporting to log per-head final constraint estimates under keys such as

* ``Head[0] Network result``

It also stores the list of final ``Stats`` objects.



----------------------------------------------------------
Checkpointing in ST-MH: exporting shared multi-head state
----------------------------------------------------------

The ST-MH solver implements checkpoint export and import for
:class:`~neuralqx.experimental.vqs.STMultiMCState` and does so in a way that is robust and
familiar to users of the existing solver stack.

Why the checkpoint stores per-head ``MCState`` payloads
--------------------------------------------------------

Even though ST-MH uses shared parameters, the solver serialises each head ``MCState`` separately using
the existing ``serialize_MCState`` utility. This keeps the payload format stable and makes it easier to
reuse existing deserialization logic and MPI metadata handling.

Export format (conceptual)
---------------------------

.. code-block:: python

   {
     "kind": "STMultiMCState",
     "n_states": K,
     "canonical_state": ...,
     "sync_model_state": ...,
     "states": [ serialize_MCState(s0), ..., serialize_MCState(sK-1) ],
     "mpi_info": ...
   }

Export is MPI-safe. A barrier is used before and after writing and only the global master rank writes
to disk.

Import workflow
----------------

Import requires an existing ST-MH template state. The typical workflow is

.. code-block:: python

   solver.set_network(...)
   solver.set_sampler(...)
   solver.set_optimizer(...)
   solver.initialize_vmc()        # builds head MCStates + shared container template
   solver.import_state(path)

This requirement exists because deserialization needs the current head models and ``MCState`` objects
to reconstruct parameter and model-state structures correctly.

Backward compatibility behaviour
---------------------------------

The ST-MH deserializer accepts an older payload without a ``"states"`` key and interprets it as a
single ``MCState`` checkpoint, wrapping it into a one-head ``STMultiMCState``.

If the saved number of states differs from the current template head count, import fails with an
explicit error.



-----------------------------------------
Usage examples (``STMultiSolver``-first)
-----------------------------------------

Minimal ST-MH setup from a trunk
---------------------------------

.. code-block:: python

   import flax.linen as nn
   import jax.numpy as jnp

   from neuralqx.experimental.solver import STMultiSolver
   from neuralqx.experimental.nn.projectors.stmh import SingleTrunkMultiHeadLogPsi

   class MyTrunk(nn.Module):
       @nn.compact
       def __call__(self, sigma):
           x = sigma.astype(jnp.float32)
           x = nn.Dense(64)(x)
           x = nn.tanh(x)
           x = nn.Dense(64)(x)
           x = nn.tanh(x)
           return x                           # features, shape (batch, 64)

   base_model = SingleTrunkMultiHeadLogPsi(
       trunk=MyTrunk(),
       n_heads=3,
       complex_logpsi=True,
   )

   solver = STMultiSolver(lqx, output_path="runs/stmh_demo", seed=0)
   solver.set_sampler(...)
   solver.set_optimizer(...)
   solver.set_network(base_model, lambda_ortho=1.0)
   solver.initialize_vmc()
   solver.run(500)

Measure all heads and one head
-------------------------------

.. code-block:: python

   all_stats = solver.expect(lqx.constraint)          # list[Stats]
   s0 = solver.expect(lqx.constraint, state_idx=0)    # Stats
   print(all_stats[0], all_stats[1], all_stats[2])
   print("head 0 mean =", float(s0.Mean))

ST-MH with explicit per-head views
-----------------------------------

If your base model exposes heads in a nonstandard way, you can provide explicit head models to
``set_network`` and still use the ST-MH solver:

.. code-block:: python

   head_models = [Head0View(base_model), Head1View(base_model), Head2View(base_model)]

   solver.set_network(
       base_model,
       head_models=head_models,
       n_heads=3,
       lambda_ortho=0.5,
   )

Head-specific operators with energy weights
--------------------------------------------

The driver supports one operator per head and weighted energy aggregation:

.. code-block:: python

   solver.initialize_vmc(
       energy_weights=[0.6, 0.2, 0.2],
       preconditioner_state_index=0,
       enforce_machine_pow_2=True,
   )

If you construct the driver manually, you can also pass a list of operators directly to
``SingleTrunkMultiHeadVMC(...)``.

Diffeomorphism-invariant ST-MH run
-----------------------------------

.. code-block:: python

   solver.set_network(
       base_model,
       diff_invariant=True,
       symmetries=symmetries,
       lambda_ortho=0.5,
   )
   solver.initialize_vmc()
   solver.run(500)

The solver applies the group projector to each head view before constructing the ``MCState`` objects.
The orthogonality penalty then separates the projected heads.

Checkpoint export and import
-----------------------------

.. code-block:: python

   solver.export_state(marker="after_500")

   # later
   solver2 = STMultiSolver(lqx, output_path="runs/stmh_resume", seed=123)
   solver2.set_sampler(...)
   solver2.set_optimizer(...)
   solver2.set_network(base_model, n_heads=3)
   solver2.initialize_vmc()
   solver2.import_state("...SerialisedSTMHState_....mpack")



---------------------------------------------------
Practical tuning notes for ST-MH
---------------------------------------------------


Feature richness of the trunk
------------------------------

The ST-MH benefit comes from a useful shared representation. A trunk that returns only a scalar
feature severely limits what the heads can express.

In practice, design the trunk to return a moderate feature width and let the heads remain lightweight.

Sampler alignment across heads
-------------------------------

Pairwise fidelity estimates use sample batches from each head state. If one head mixes poorly or uses
very different sampling settings, the orthogonality term becomes noisy and can dominate the update.

It is usually best to keep sampler settings aligned across heads unless there is a deliberate reason
to diverge.

Interpreting parameter counts
------------------------------

The solver logs both shared and nominal replicated parameter counts.

* Shared count approximates the true optimization dimension.
* Nominal replicated count is a comparison number that is closer to what an MT-MH run would spend if
  each head were trained independently.

This is a useful diagnostic when comparing ST-MH and MT-MH runs at similar compute budgets.



---------------------------------------------------
Migration notes: MT-MH to ST-MH
---------------------------------------------------

If you already use :class:`~neuralqx.experimental.solver.MultiSolver`, the main conceptual migration
steps are:

1. Replace a list of independent networks with one shared ST-MH base model.
2. Ensure the base model can expose a selected head with ``head=i`` or provide explicit head views.
3. Use ``STMultiSolver`` instead of ``MultiSolver``.
4. Interpret parameter counts and SR behaviour with the ST-MH shared-parameter geometry in mind.

What stays the same
--------------------

* Sampling still uses ordinary ``MCState`` objects.
* The orthogonality penalty uses the same joint-sampling estimator family.
* The solver workflow remains ``set_sampler -> set_optimizer -> set_network -> initialize_vmc -> run``.

What changes mathematically
----------------------------

* The optimization variable becomes one shared parameter pytree.
* Gradients from all heads and all pair penalties are summed before preconditioning.
* SR is applied once on a reference head geometry in the current implementation.

This is the correct update structure for a shared trunk plus multiple heads ansatz.



------------------------------------------------------------
Troubleshooting checklist
------------------------------------------------------------

Shape errors in the wrapper
----------------------------

Symptoms:

* Flax Dense shape mismatch in the head layers
* unexpected output rank from the base model

Checks:

* confirm the trunk returns batch-first features
* set ``return_features=True`` temporarily and inspect ``feats.shape``
* use ``flatten_features=True`` if the trunk returns ``(batch, ..., ...)``

``TypeError`` when building head views
---------------------------------------

Symptoms:

* solver error stating the base model does not support ``head=``

Fixes:

* use ``SingleTrunkMultiHeadLogPsi`` or a compatible base model
* or pass explicit ``head_models=[...]`` to ``STMultiSolver.set_network(...)``

``ValueError`` about ``machine_pow``
-------------------------------------

Symptoms:

* driver raises because some head has ``machine_pow != 2``

Fixes:

* align all samplers to ``machine_pow=2`` for standard fidelity semantics
* only disable ``enforce_machine_pow_2`` if you intentionally want a surrogate overlap penalty

Unexpected head divergence or inconsistent behaviour
-----------------------------------------------------

Checks:

* verify the shared state is ``STMultiMCState`` and not ``MultiMCState``
* confirm ``reset()`` is being called through the driver each step
* avoid manual mutation of per-head ``MCState.parameters`` outside the container

SR instability
---------------

Checks:

* compare against identity preconditioning first
* try a different ``preconditioner_state_index``
* inspect per-head sample quality and fidelity noise
* reduce ``lambda_ortho`` temporarily to decouple sources of instability

