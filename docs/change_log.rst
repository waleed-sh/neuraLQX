.. _change_log:

============================================
Change Log
============================================


Unreleased
------------

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

  - **MT-MH / independent-state** training (one network per target state; existing behavior)
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
