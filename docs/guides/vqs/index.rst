.. _vqs_mcstate:

============================
Variational Quantum States
============================

This page documents the **Monte-Carlo variational state** abstraction used throughout
neuraLQX. The central object is :class:`neuralqx.vqs.MCState`, a thin but purpose-built subclass of NetKet's
``MCState`` that preserves NetKet's public interface while adding a small number of LQG-focused measurement conveniences.

The intent of this module is straightforward:

* You define a Hilbert space (discrete configurations :math:`\sigma`).
* You choose an ansatz :math:`\psi_\theta(\sigma)` (usually a Flax module).
* You choose a sampler that draws configurations distributed according to the probability
  induced by :math:`\psi_\theta`.
* You use the resulting variational state to evaluate expectations and gradients of operators.

This page focuses on:

1. What :class:`~neuralqx.vqs.MCState` is and what it stores.
2. How :meth:`~neuralqx.vqs.MCState.expect` works (mathematically and operationally).
3. The **multi-operator expectation** extension: passing a *list of operators* to a single call.
4. The corresponding extension to gradients: :meth:`~neuralqx.vqs.MCState.expect_and_grad`
   for **objectives written as explicit sums**.

We intentionally **do not** discuss projectors or projection-based wrappers here.


.. currentmodule:: neuralqx.vqs


-------------------------
The ``MCState`` object
-------------------------

The :class:`~neuralqx.vqs.MCState` object is a concrete realization of NetKet’s
``VariationalState`` interface. In practical terms, it is the “bridge” between:

* **neural-network evaluation** (computing :math:`\log \psi_\theta(\sigma)` on batches), and
* **operator evaluation** (constructing local estimators and aggregating Monte-Carlo statistics).

From a user perspective, ``MCState`` behaves like "a wavefunction with a sampler attached".
It stores the components required for Monte-Carlo estimation and keeps *a cached sample batch*
that measurement routines can reuse.

Typical construction uses:

* a NetKet sampler (Metropolis variants, exchange samplers, custom transitions, ...),
* and a model that maps configurations to log-amplitudes.

In most workflows the model is a ``flax.linen.Module``. However, advanced users may pass callables compatible with
NetKet's ``init_fun/apply_fun`` pattern.

A minimal shape of such a workflow looks like:

.. code-block:: python

   import netket as nk
   import neuralqx as nqx

   H = ...                 # a neuraLQX Hilbert space
   sampler = nk.sampler.MetropolisLocal(H.hilbert)  # feed the NetKet Hilbert space object

   model = ...              # flax.linen.Module (or an apply_fun wrapper)
   vstate = nqx.vqs.MCState(sampler, model, n_samples=4096, n_discard_per_chain=64)

Once constructed, ``vstate`` offers a uniform interface for:

* drawing configurations :math:`\sigma`,
* evaluating :math:`\log \psi_\theta(\sigma)` and log-ratios,
* estimating expectation values and (optionally) forces/gradients.


MCState’s core attributes
----------------------------

The following attributes control runtime behavior and expose the internal state in a way
that is safe and practical for users. The exact attribute set follows NetKet conventions.

hilbert
  The underlying Hilbert space on which configurations :math:`\sigma` live. It defines the
  configuration representation, dtype, and any structure that operators and samplers rely on. This is a NetKet
  Hilbert space.

model
  The ansatz definition used to evaluate log-amplitudes. In most cases this is a Flax module,
  but it can also be a thin wrapper around an ``apply_fun`` (for example when integrating with
  external model code).

variables
  The full Flax-style variable PyTree. It contains trainable parameters under ``"params"`` and
  may contain additional non-trainable collections (for example BatchNorm statistics). Most
  users treat this as an opaque tree passed into NetKet/Optax optimizers.

sampler and sampler_state
  The sampler object and its current Markov chain state. The sampler_state evolves whenever
  sampling is performed. Exposing it is valuable for reproducibility and for advanced workflows
  where you want to checkpoint and resume a run.

n_samples, chain_length, n_discard_per_chain
  Sampling controls. These determine how many configurations are produced (and how they are
  organized into chains), and how many warmup/discard steps are taken per chain.

samples
  A cached array containing the most recently generated batch of configurations. Sampling is
  triggered lazily: measurement routines request samples, and the cache is updated when needed.


Sampling, caching, and determinism
------------------------------------

It is important to understand when sampling happens:

* Accessing ``vstate.samples`` may trigger sampling if the cache is empty or stale.
* Calling :meth:`~neuralqx.vqs.MCState.expect` or :meth:`~neuralqx.vqs.MCState.expect_and_grad`
  triggers sampling if no cached samples are available for the current state.

Parameter initialization and sampling are seeded independently. This is a small but important
detail, it lets you compare two optimization runs with identical initial parameters but different
Monte-Carlo noise, or vice versa.


------------------------------------------------------
Expectation values via local estimators (``expect``)
------------------------------------------------------

At the mathematical level, expectation values are expressed using **local estimators**.
Let :math:`|\psi_\theta\rangle` be a parametrized pure state with amplitudes
:math:`\psi_\theta(\sigma) = \langle \sigma | \psi_\theta\rangle` in a computational basis
:math:`\{|\sigma\rangle\}`. For an operator :math:`\hat{O}`, define the local estimator

.. math::

   O_{\mathrm{loc}}(\sigma)
   \;=\;
   \frac{\langle \sigma|\hat{O}|\psi_\theta\rangle}{\langle \sigma|\psi_\theta\rangle}
   \;=\;
   \sum_{\sigma'}
   O_{\sigma\sigma'}
   \frac{\psi_\theta(\sigma')}{\psi_\theta(\sigma)}.

This identity is the workhorse of Monte-Carlo operator evaluation. The probability distribution
sampled by the Markov chain is

.. math::

   p_\theta(\sigma) \propto |\psi_\theta(\sigma)|^2,

so the expectation becomes

.. math::

   \langle \hat{O}\rangle_\theta
   \;=\;
   \sum_{\sigma} p_\theta(\sigma)\, O_{\mathrm{loc}}(\sigma).

Given a batch of samples :math:`\{\sigma^{(n)}\}_{n=1}^{N_s}` drawn from the chain, we estimate

.. math::

   \langle \hat{O}\rangle_\theta \approx \frac{1}{N_s}\sum_{n=1}^{N_s} O_{\mathrm{loc}}(\sigma^{(n)}).

The return value of :meth:`~neuralqx.vqs.MCState.expect` is NetKet's ``Stats`` object, which
encodes mean and uncertainty estimates (and may include diagnostics depending on your NetKet setup).


What the code computes: connectivity + log-ratios
----------------------------------------------------

The formula for :math:`O_{\mathrm{loc}}(\sigma)` shows that you need two ingredients:

1. **Connectivity**: for each configuration :math:`\sigma`, which configurations :math:`\sigma'`
   are connected by the operator and with what matrix elements :math:`O_{\sigma\sigma'}`?
2. **Wavefunction ratios**: for those connected configurations, compute
   :math:`\psi_\theta(\sigma')/\psi_\theta(\sigma)` efficiently.

In code, most discrete operator families (NetKet operators, and neuraLQX's discrete computational
operators) provide a connectivity primitive with the spirit of:

* ``sigma_prime, mels = O.get_conn_padded(sigma_batch)``

Here:

* ``sigma_batch`` is a batch of sampled configurations :math:`\sigma`.
* ``sigma_prime`` is a padded batch of connected configurations :math:`\sigma'`.
* ``mels`` holds the corresponding matrix elements :math:`O_{\sigma\sigma'}` (also padded).

Then :meth:`~neuralqx.vqs.MCState.expect` computes the local estimator by evaluating the model
on both the original samples and the connected samples, forming log-ratios, and summing:

.. math::

   O_{\mathrm{loc}}(\sigma)
   \;=\;
   \sum_{\sigma'} O_{\sigma\sigma'}
   \exp\!\Big(\log\psi_\theta(\sigma') - \log\psi_\theta(\sigma)\Big).

This is why the MCState interface is defined around a callable

.. math::

   \sigma \mapsto \log \psi_\theta(\sigma),

possibly complex-valued when representing phases explicitly.


The local-kernel pipeline
----------------------------------

Internally, NetKet (and hence neuraLQX) structure expectation values around two operator-dependent
dispatch points:

``get_local_kernel_arguments(vstate, O)``
  Prepares all per-sample operator data needed by a kernel. For discrete operators this is
  typically connectivity (connected configurations and matrix elements), but it can also be
  additional metadata.

``get_local_kernel(vstate, O)``
  Returns a callable that computes :math:`O_{\mathrm{loc}}(\sigma)` given an apply-function for
  :math:`\log\psi_\theta`, the variable tree, the samples, and the prepared kernel arguments.

Most users never touch these functions directly, but understanding them clarifies why arbitrary
operator families can be supported. As long as the operator can supply a kernel (or can be reduced
to something that can), :meth:`~neuralqx.vqs.MCState.expect` stays uniform.


A concrete usage example
---------------------------

Suppose ``lqx.model.number(edge_descriptor)`` returns a number-like operator that reads out a
quantum number on a given edge in your graph-based configuration. You can measure it as:

.. code-block:: python

   N = lqx.model.number((0, 1, 0))
   stats = vstate.expect(N)

The returned object prints like a complex mean with an uncertainty, but it is a full NetKet ``Stats``
container. You can access fields depending on your NetKet version (commonly mean and error-of-mean).


------------------------------------------------------------
Multi-operator expectations: passing a list of operators
------------------------------------------------------------

Why lists matter
------------------

In many LQG workflows the "objective" you want is naturally written as a sum of contributions
that you want to keep **distinct**:

* vertex-wise terms from a functional local operator,
* contributions that live in different operator families (and therefore cannot be merged),
* or components where you want separate identities for logging/analysis.

In plain NetKet you can often build a composite operator :math:`\hat{O}=\hat{A}+\hat{B}` and
then call ``expect(O)``. That works well when operator arithmetic is defined and when you
actually want a composite object.

In neuraLQX, you may want the objective

.. math::

   \langle \hat{A}\rangle_\theta + \langle \hat{B}\rangle_\theta

without forcing construction of a single explicit ``A+B`` operator (which may be impossible for
mixed operator families). For this reason, neuraLQX extends measurement routines to accept **sequences**:

* :func:`neuralqx.vqs.expect(vstate, [O1, O2, ...])`
* :func:`neuralqx.vqs.expect_and_grad(vstate, [O1, O2, ...])`
* :func:`neuralqx.vqs.expect_and_forces(vstate, [O1, O2, ...])`

and the corresponding methods if exposed on the state object.

What ``expect([O1, O2, ...])`` means mathematically
-----------------------------------------------------

The list interface is not "measure all operators and return all results". Instead, it is a
single measurement corresponding to the **sum of expectations**.

Operationally, neuraLQX evaluates all local estimators on the *same cached sample batch* and
aggregates at the estimator level. Define local estimators :math:`O^{(k)}_{\mathrm{loc}}(\sigma)`
for each operator :math:`\hat{O}_k`. Then neuraLQX forms

.. math::

   L_\Sigma(\sigma) := \sum_{k=1}^{K} O^{(k)}_{\mathrm{loc}}(\sigma)

and returns statistics for the Monte-Carlo average of :math:`L_\Sigma`.

This design has an important statistical consequence namely because the sum is built **per sample**,
the resulting ``Stats`` object automatically includes the correct variance contribution from
cross-covariances between terms evaluated on the same sample batch. Put differently, you are
measuring one random variable :math:`L_\Sigma(\sigma)` rather than separately measuring
:math:`O^{(1)}_{\mathrm{loc}}(\sigma)` and :math:`O^{(2)}_{\mathrm{loc}}(\sigma)` with independent
noise.

In addition, it has a practical consequence: sampling is done once, which is often the dominant
cost for large models.


A concrete equivalence example
---------------------------------

The list-based expectation is designed so that if operator arithmetic exists and is appropriate,
you get identical results (including statistics):

.. code-block:: python

   N1 = lqx.model.number((0, 1, 0))
   N2 = lqx.model.number((0, 1, 0))

   # composite operator path (if defined)
   stats_a = vstate.expect(N1 + N2)

   # sequence path (always available in neuraLQX)
   stats_b = vstate.expect([N1, N2])

Conceptually, both calls estimate :math:`\langle \hat{N}_1 + \hat{N}_2\rangle_\theta`. The
difference is that the list path avoids constructing a composite operator and guarantees a
single shared sampling step.


Implementation intuition (what happens in one call)
---------------------------------------------------

A call like ``vstate.expect([O1, O2, O3])`` follows the same pipeline as a single-operator call,
but with one key change: local estimators are accumulated before statistical aggregation.

A good mental model is:

1. Obtain (or generate) the cached sample batch ``sigma = vstate.samples``.
2. For each operator ``Ok``:
   a. prepare its kernel arguments (connectivity, metadata, ...),
   b. evaluate its local estimator array ``Lk[sample_index]``.
3. Form the per-sample sum ``Lsum = L1 + L2 + L3``.
4. Build one ``Stats`` object from ``Lsum``.

Because step (4) sees only the summed per-sample values, the uncertainty reflects the empirical
fluctuations of the full objective, not a naïve sum of independent error bars.


---------------------------------------------------------------
Gradients and multi-operator objectives (``expect_and_grad``)
---------------------------------------------------------------

Why gradients are subtle in VMC
-----------------------------------

Optimizing a variational wavefunction requires gradients of expectation values with respect to
parameters :math:`\theta`. In VMC, differentiating "through the Markov chain" and "through the
operator connectivity" is usually avoided. Instead, one uses identities that express gradients in
terms of covariances with log-derivatives of the wavefunction.

The core quantity is the log-derivative:

.. math::

   \mathcal{O}_j(\sigma) := \partial_{\theta_j}\log \psi_\theta(\sigma).

For Hermitian operators, NetKet's default estimator uses the covariance ("forces") formula.
Schematically, the force component is

.. math::

   F_j = \mathrm{Cov}\big(\mathcal{O}_j(\sigma),\, O_{\mathrm{loc}}(\sigma)\big),

and an internal conversion step ``force_to_grad`` turns forces into parameter gradients in a way
consistent with the parameter PyTree structure (and, depending on optimizer/QGT settings, may
apply additional transformations).

In practical API terms, the usual workflow is:

.. code-block:: python

   stats, grad = vstate.expect_and_grad(H)

where ``grad`` is a PyTree matching the parameter structure.


Extending gradients to operator lists
------------------------------------------

neuraLQX extends the same idea to list-based objectives using linearity of differentiation:

.. math::

   \nabla_\theta \sum_{k=1}^{K}\langle \hat{O}_k\rangle_\theta
   \;=\;
   \sum_{k=1}^{K}\nabla_\theta \langle \hat{O}_k\rangle_\theta.

The important implementation detail is that all terms are evaluated on the **same** sample batch,
so the gradient estimator is consistent with the objective that ``expect([O1, ...])`` computes.

If the operators are Hermitian (or if you explicitly enable covariance mode), neuraLQX routes the
list computation through the forces pipeline, then converts to gradients—mirroring the single
operator path, just with a per-sample accumulated estimator.

For non-Hermitian cases (including NetKet's lazy wrappers such as ``Squared``), NetKet uses a
VJP-based route. neuraLQX extends this to lists by looping over operators, accumulating both the
summed local estimator and the partial gradient contributions, and returning a single ``Stats``
object for the summed estimator.

From a user perspective, the key point is: you can write objectives as explicit lists without
losing access to first-class gradients.


A concrete multi-term objective example
--------------------------------------------

Assume you have an objective that is a sum of three terms, possibly from different operator
families:

.. code-block:: python

   Oa = ...
   Ob = ...
   Oc = ...

   stats, grad = vstate.expect_and_grad([Oa, Ob, Oc])

This computes one scalar objective and one gradient PyTree. The objective is the Monte-Carlo
estimate of :math:`\langle \hat{O}_a\rangle + \langle \hat{O}_b\rangle + \langle \hat{O}_c\rangle`
using a single sample batch.

A common pattern in LQG settings is to keep ``Oa/Ob/Oc`` as separate Python objects for logging
and ablation studies, while training against their sum. List-based gradients are designed for
exactly that style of workflow.


-------------------------------------------------
Practical notes for stable measurements
-------------------------------------------------

Sampling parameters matter more than you think. When expectations look noisy or gradients are
unstable, the first knobs to inspect are:

* ``n_samples``: increases estimator accuracy roughly like :math:`1/\sqrt{N_s}` for fixed
  autocorrelation.
* ``n_discard_per_chain``: helps ensure chains forget their initial state.
* ``chain_length`` and sampler choice: determine autocorrelation and mixing.

It is also worth remembering that the local estimator can have a broader distribution than the
operator spectrum suggests, because it involves ratios :math:`\psi_\theta(\sigma')/\psi_\theta(\sigma)`.
When the model assigns extremely small probabilities to parts of configuration space, rare samples
can yield large log-ratios. In those cases you typically address stability by improving the sampler
(mixing), increasing discard, increasing sample count, or adjusting the ansatz/initialization.


.. toctree::
   :hidden:
   :maxdepth: 2

   penalties/index
   projectors/index
