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
