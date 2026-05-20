.. _change_log:

============================================
Change Log
============================================


Unreleased
-----------------------------------

Breaking changes
~~~~~~~~~~~~~~~~~
- The MT-MH multi-state VMC implementation has been removed from ``neuralqx.experimental`` after promotion
  to the public API. Use :class:`neuralqx.solver.MultiSolver`, :class:`neuralqx.driver.MultiStateVMC`,
  and :class:`neuralqx.vqs.MultiMCState` instead of the old experimental imports.

New features
~~~~~~~~~~~~~
- Promoted the standard MT-MH multi-state Monte Carlo solver to the public API as
  :class:`neuralqx.solver.MultiSolver`, with public :class:`neuralqx.driver.MultiStateVMC` and
  :class:`neuralqx.vqs.MultiMCState` building blocks.

Changes
~~~~~~~~
- The minimum supported NetKet version is now 3.20.0 instead of 3.19.0.

- The minimum supported JAX version is now 0.7.0 instead of 0.5.0.

- The minimum supported Flax version is now 0.10.2 instead of 0.6.5.

- Improved multi-state solver serialization, import/export, logging, and plotting. Multi-state checkpoints now
  carry an explicit schema, validate their state count against the active template, and plotting reports per-state
  constraint curves with the same production path as the standard solver.

Bug fixes
~~~~~~~~~~
- None.

Deprecations
~~~~~~~~~~~~~
- None.


Experimental
~~~~~~~~~~~~~
- None.

------------


neuraLQX v1.1.2 (April 19, 2026)
-----------------------------------

Breaking changes
~~~~~~~~~~~~~~~~~
- neuraLQX now supports Python ``>=3.11,<3.14``. Python ``3.14`` and newer are not supported.

New features
~~~~~~~~~~~~~
- Introduced a new :class:`neuralqx.utils.io.runtime_loggers.DeferredRuntimeLog` which avoids excessive device-to-host
  transfers in the VMC driver when ``logger(...)`` is called.

Changes
~~~~~~~~
- Added a new configuration variable ``NQX_FUSED_KERNELS`` (default ``True``) which when set to ``False``, will avoid
  a fused kernel in the VQS path (expect/gradient/forces) for sequences of operators which can reduce the first step compile time in the VMC.

Bug fixes
~~~~~~~~~~
- Fixed a bug in the debugger which effectively made it always on.

- Fixed a bug in the ``Solver`` which did not propagate ``silent_print=True`` to the VMC driver.

- Fixed an import-side effect where the symbolic operator deprecation warning could be emitted when only the
  experimental gradient path was touched, by lazily importing ``neuralqx.experimental`` submodules.

Deprecations
~~~~~~~~~~~~~
- The experimental symbolic API under :mod:`neuralqx.experimental.operators.symbolic` is now deprecated in favour of
  nkDSL (the independent DSL for both NetKet and neuraLQX), and will be removed in the next release. Users should migrate
  to nkDSL and follow its documentation: https://nkdsl.readthedocs.io/en/latest/?badge=latest


Experimental
~~~~~~~~~~~~~
- Added a ``for_each_distinct_pair()`` to the experimental operator DSL which iterates over sites ``i != j``.

- Added an exact bi-covariance gradient route for non-Hermitian objectives in
  ``expect_and_grad`` (single operators, sequences, and ``Squared`` objectives) behind
  ``NQX_EXPERIMENTAL_GRAD``.

------------




neuraLQX v1.1.1 (April 4, 2026)
-----------------------------------

Breaking changes
~~~~~~~~~~~~~~~~~
- The Hilbert package was restructured so that ``neuralqx.hilbert.utils`` now contains abstractions only, concrete indexing/layout/operations now live in concrete Hilbert-space subpackages.

- ``AbstractHilbertSpace`` and ``AbstractHilbertInterface`` are now strictly abstract and no longer provide concrete U(1)-specific defaults.

- Hilbert interface API cleanup: ``.core`` has been removed. Use ``.hilbert`` (neuraLQX core) and ``.hilbert_netket`` (NetKet Hilbert object).

- Deprecated the legacy ``neuralqx.hilbert.utils.layout.gauge_strided`` path, use ``neuralqx.hilbert.u1.layout.StridedGaugeCopyLayout``.

New features
~~~~~~~~~~~~~
- Added a richer Hilbert enumerator abstraction with explicit metadata and explainability:
  ``scheme_name``, ``ordering_contract``, ``supports_lazy_mode``, ``requires_full_precompute``,
  ``explain_state_to_number``, and ``explain_number_to_state``.

- ``states_to_numbers`` / ``numbers_to_states`` now dispatch through plum via each core's attached enumerator class.

- Added expanded profiling controls (run id, JAX trace/annotation toggles, sampling cadence, trace buffer sizing, and optional deep Python-call tracing) for clearer end-to-end runtime diagnostics.

- Added new profiling helpers to instrument external library calls (for example, sampling paths) without editing upstream source code.

- An implementation for the Lorentzian constraint for the single vertex QRLG model is now available.

- Introduced a new strict versioning subsystem under ``neuralqx.utils.module.version`` with ``Version`` and ``NeuralqxVersion`` classes, plus module-version parsing helpers for robust dependency and compatibility checks.

Changes
~~~~~~~~
- The configuration system (``neuralqx.configs``) has been redesigned with a typed architecture. Options now carry explicit types, a three-level mutability model (``IMMUTABLE`` / ``STARTUP`` / ``RUNTIME``), and structured mutation hooks. A ``cfg.patch()`` context manager and ``cfg.thread_local_override()`` are available for scoped overrides. Existing ``NQX_*`` environment variables and ``cfg.get()`` / ``cfg.set()`` calls are unaffected.

- ``NQX_LOG_LEVEL`` is now runtime-mutable (previously startup-only).

- ``NQX_EXPERIMENTAL`` and ``NQX_TESTING`` now accept boolean values (``true`` / ``false``) in addition to ``0`` / ``1``.

- U(1) random/flip operations now live under ``neuralqx.hilbert.u1.operations``, the old top-level ``neuralqx.hilbert.operations`` package has been removed.

- U(1) concrete layout now lives under ``neuralqx.hilbert.u1.layout``.

- U(1)-specific index helpers were moved out of ``neuralqx.hilbert.utils.index``, the utils index package is now abstraction-focused.

- Slightly improved VMC runtime with fused operator-evaluation kernels in ``expect_and_grad``/``expect_and_forces`` for multi-constraint workloads, while preserving separate constraint estimators.

- ``neuralqx.__version__`` is now exposed as ``NeuralqxVersion`` (string-compatible, semantic comparisons), and ``neuralqx.version_info`` provides the strict ``Version`` object for typed checks.

- ``neuralqx`` no longer requires being imported before JAX or NetKet, a ``neuralqx_boot.pth`` file installed into site-packages sets all required environment variables at interpreter startup automatically.

- Added ``neuralqx.utils.dtypes``, a ``DTypePolicy`` class that derives the active real, complex, and index dtypes from configuration (respecting ``NQX_DTYPE_*`` and ``NQX_ENABLE_X64``), plus policy-aware JAX array constructors (``zeros_real``, ``ones_complex``, ``array_index``, etc.) so call sites never hard-code dtype arguments.


Bug fixes
~~~~~~~~~~
- Fixed circular-import issues in Hilbert index dispatch/core initialization paths.

- Corrected dispatch resolution so state-number conversion reliably uses concrete enumerator implementations.


Deprecations
~~~~~~~~~~~~~
- None.


Experimental
~~~~~~~~~~~~~
- Introduced an experimental declarative symbolic operator DSL under
  ``neuralqx.experimental.operators.symbolic.dsl``. Users can now define
  operator action by composing iterator scopes, predicates, and emissions
  instead of hand-writing ``get_conn_padded`` kernels. The compiler lowers the
  resulting symbolic IR to executable matrix-free operators, and ships with
  documented extension points for custom passes, iterator semantics, and JAX
  lowerers.

- Introduced a new struct system under ``neuralqx.utils.struct`` for building immutable, JAX-native data containers.
  Subclass ``Struct`` (or use ``@register_class`` / ``@dataclass`` for existing classes) to get automatic JAX pytree
  registration, a typed field API (``field(static=..., derived=..., converter=..., validator=...)``), and built-in
  serialization via ``export()`` / ``load()`` and ``to_state_dict()`` / ``from_state_dict()``.
  Fields are classified as *node* (JAX leaves), *static* (pytree aux / JIT cache keys), or *opaque* (runtime-only,
  excluded from tracing and serialization). Derived fields are computed from other fields and recomputed
  automatically on ``replace()``.  Non-``Struct`` classes can participate via ``register_pytree_type()`` or
  ``register_attrs_type()``.

------------

neuraLQX v1.1.0 (March 10, 2026)
------------------------------------

Breaking changes
~~~~~~~~~~~~~~~~~
- neuraLQX now requires a NetKet version of 3.19+, and no longer supports earlier versions. This means that, thanks to
  NetKet's latest versions, neuraLQX simulations now run substantially faster than the previous version.

- **neuraLQX no longer support MPI based parallelisation.** The only mode of parallelisation is now through JAX's sharding,
  and only for Linux systems, as currently supported by the latest NetKet versions. This means that older neuraLQX scripts
  are not going to be compatible with this version.

- The ``NQX_MPI`` and ``NQX_MPI_CUDA`` configuration variables are now removed. neuraLQX will use all available
  GPUs automatically and if none are available, it will fallback to CPU execution.


New features
~~~~~~~~~~~~~
- A new module :mod:`~neuralqx.utils.distributed` is now responsible for all parallelisation related utils.


Changes
~~~~~~~~
- None.


Deprecations
~~~~~~~~~~~~~
- Module :mod:`~neuralqx.utils.mpi` has been removed as neuraLQX no longer supports MPI based parallelisation.


Experimental
~~~~~~~~~~~~~
- None.


------------


neuraLQX v1.0.1 (March 6, 2026)
------------------------------------

Breaking changes
~~~~~~~~~~~~~~~~~
- The gauge-copy layout implementation has been refactored around a new abstract layout API.
  The concrete strided implementation is now :class:`neuralqx.hilbert.utils.layout.StridedGaugeCopyLayout`
  (with ``GaugeLayout`` retained as a backward-compatible alias).


New features
~~~~~~~~~~~~~
- The :mod:`solver.utils.mpi` module now has support for JAX based MPI collectives and point-to-point communications.

- Added :class:`neuralqx.hilbert.utils.layout.AbstractBasisLayout`, a generic abstract base class for
  flattening/unflattening structured basis coordinates into 1D site indices. This provides a
  future-proof API for upcoming non-U(1) layouts, including SU(2)-specific coordinate schemes.

- Added :class:`neuralqx.hilbert.utils.layout.GaugeCoord`, an explicit structured coordinate type
  for gauge-copy strided layouts.

- Added :class:`neuralqx.hilbert.utils.layout.StridedGaugeCopyLayout` as the concrete implementation
  of the current gauge-copy block-strided flattening convention.

- Added layout-agnostic methods ``site_of(coord)`` and ``coord_of(site)`` to support generic code,
  alongside convenience methods ``encode(gauge_copy, edge_index)`` and ``decode(site)`` for gauge layout implementations.


Changes
~~~~~~~~
- class:`neuralqx.hilbert.utils.layout.GaugeLayout` has now been renamed to ``StridedGaugeCopyLayout``. Backward compatibility exists, but the
  a deprecation warning will be issued and the renaming will be final in an upcoming release. Please change your code accordingly.


Bug fixes
~~~~~~~~~~
- Fixed a bug in the :class:`neuralqx.solver.Solver` which caused the current simulation's output directory to be deleted before the run is completed when ``clean_up=True``.


Deprecations
~~~~~~~~~~~~~
- The legacy helper ``GaugeLayout.site(edge_index, gauge_copy=0)`` (and the corresponding method on
  :class:`StridedGaugeCopyLayout`) is deprecated and will be removed in a future release.
  Use ``encode(gauge_copy, edge_index)`` instead. The new argument order is consistent with
  ``decode(site) -> (gauge_copy, edge_index)``.

- The :mod:`neuralqx.utils.mpi` is now marked to be deprecated. In the next release, neuraLQX will stop MPI support in
  favour of JAX's sharding and to support the latest NetKet releases. This means that the only mode of parallelisation
  will be on GPUs and using JAX, and not MPI.


Experimental
~~~~~~~~~~~~~
- Experimental multi-state VMC now supports two distinct strategies:

  - **MT-MH / independent-state** training (one network per target state, existing behavior)
  - **ST-MH / shared-trunk multi-head** training (new behavior)

  The ST-MH path shares trunk parameters across all heads and applies one optimizer update to the shared model.

- Added a single-trunk multi-head (ST-MH) multi-state VMC stack for learning multiple
  (near-)degenerate states with a shared parameterization instead of ``N`` fully independent networks.

- Added a Flax ST-MH wrapper module for constructing shared-trunk / multi-head ansätze from an arbitrary
  user-defined trunk network (feature extractor), including head selection utilities so each head can be
  exposed as a standard scalar-output model for ``MCState`` compatibility.

- Added :class:`neuralqx.state.stmh_state.STMultiMCState`, a shared-parameter multi-state container
  that wraps multiple per-head ``MCState`` objects while exposing a single canonical parameter pytree and
  synchronizing parameters across heads.

- Added :class:`neuralqx.solver.stmh_multi_vmc.SingleTrunkMultiHeadVMC`, a dedicated ST-MH VMC driver that
  performs a single shared update by aggregating weighted per-head energy gradients and pairwise
  orthogonality/fidelity penalty gradients.

- Added :class:`neuralqx.solver.stmh_solver.STMultiSolver`, a solver-level interface for ST-MH training that mirrors
  the previous multi-state solver workflow while using the new shared-parameter state/driver backend.
