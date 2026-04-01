.. _cfgs:

============================
Configuration Options
============================

This page lists all **environment variables** available in neuraLQX.

Each variable allows you to modify the runtime behaviour of the package, enable optional features,
or adjust numerical and parallelisation settings.

.. note::

   Environment variables must be set **before importing** neuraLQX in your python script or shell
   session, unless the variable is marked *Runtime Mutable* (see table below).

   Example (bash):

   .. code-block:: bash

      export NQX_EXPERIMENTAL=true
      export NQX_LOG_LEVEL=DEBUG
      python your_script.py

   Example (python script):

   .. code-block:: python

        import os
        os.environ['NQX_EXPERIMENTAL'] = 'true'

        import neuralqx as nqx

.. raw:: html

   <hr style="margin: 20px 0;">

Environment Variables
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 25 15 30 10

   * - **Name**
     - **Value Type**
     - **Possible Values**
     - **Default**
     - **Description**
     - **Runtime Mutable**
   * - ``NQX_DEBUG``
     - ``bool``
     - ``False`` | ``True``
     - ``False``
     - Enable or disable debugging mode throughout various parts of neuraLQX
     - Yes
   * - ``NQX_PROFILE``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Enable profiling in neuraLQX.
     - No
   * - ``NQX_PROFILE_NVTX``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Enable nvidia-smi profiling in neuraLQX.
     - No
   * - ``NQX_PROFILE_METRICS``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Enable profiling telemetry (CPU/memory + GPU via NVML when available).
     - No
   * - ``NQX_PROFILE_TRACE``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Enable Perfetto UI traces in neuraLQX.
     - No
   * - ``NQX_PROFILE_SYNC``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Enable device-accurate timing (forces synchronisation and can slow execution).
     - No
   * - ``NQX_PROFILE_DIR``
     - ``str``
     - Any valid path
     - ``<project>/.neuralqx_profiling/neuralqx_<YYYYMMDD>``
     - Override the directory where profiling artifacts are written.
     - No
   * - ``NQX_PROFILE_RUN_ID``
     - ``str``
     - Any string
     - ``""`` (auto)
     - Optional fixed profiling run id; outputs are written under ``run_<RUN_ID>``.
     - No
   * - ``NQX_PROFILE_JAX_ANNOTATE``
     - ``int``
     - ``0`` | ``1``
     - ``1``
     - Enable JAX trace annotations for profiling sections.
     - No
   * - ``NQX_PROFILE_JAX_TRACE``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Enable JAX profiler trace capture in the profiling output folder.
     - No
   * - ``NQX_PROFILE_SAMPLE_PERIOD_S``
     - ``float``
     - ``> 0``
     - ``0.1``
     - Telemetry sampling period in seconds when ``NQX_PROFILE_METRICS=1``.
     - No
   * - ``NQX_PROFILE_MAX_EVENTS``
     - ``int``
     - Positive integer
     - ``2000000``
     - Maximum in-memory trace events before export.
     - No
   * - ``NQX_PROFILE_MPI_AGG``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Aggregate profiling summaries across ranks at exit (can block on slow ranks).
     - No
   * - ``NQX_PROFILE_PY_CALLS``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Enable deep Python call tracing for profiled wrappers/calls (high overhead).
     - No
   * - ``NQX_PROFILE_PY_CALLS_INCLUDE``
     - ``str``
     - Comma-separated module prefixes
     - ``netket,neuralqx``
     - Include filter for deep Python call tracing.
     - No
   * - ``NQX_PROFILE_PY_CALLS_EXCLUDE``
     - ``str``
     - Comma-separated module prefixes
     - ``neuralqx.profile``
     - Exclude filter for deep Python call tracing.
     - No
   * - ``NQX_PROFILE_PY_CALLS_MAX_DEPTH``
     - ``int``
     - Integer (``<=0`` means unlimited)
     - ``6``
     - Max nested Python call depth captured by deep tracing.
     - No
   * - ``NQX_VERBOSE``
     - ``bool``
     - ``False`` | ``True``
     - ``True``
     - Enable or disable Rich console printing.
     - Yes
   * - ``NQX_LOG_LEVEL``
     - ``str`` (enum)
     - ``DEBUG`` | ``INFO`` | ``WARNING`` | ``ERROR`` | ``CRITICAL``
     - ``INFO``
     - Set the logging level for the neuraLQX logger hierarchy.
     - Yes
   * - ``NQX_EXPERIMENTAL``
     - ``bool``
     - ``False`` | ``True``
     - ``False``
     - Enable experimental functions throughout the neuralqx and netket packages.
     - Yes
   * - ``NQX_TESTING``
     - ``bool``
     - ``False`` | ``True``
     - ``False``
     - Relax some neuraLQX features for testing purposes.
     - Yes
   * - ``NQX_CACHE``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Enable caching when possible.
     - No
   * - ``NQX_ENABLE_X64``
     - ``bool``
     - ``False`` | ``True``
     - ``True``
     - Enable x64 precision throughout neuraLQX, NetKet and JAX. Falls back to
       ``JAX_ENABLE_X64`` if ``NQX_ENABLE_X64`` is not set.
     - No

.. important::

   neuraLQX also has profiling specific configurations options. For that, please see the :doc:`profiling page <guides/profiling>`.

.. raw:: html

   <hr style="margin: 30px 0;">


Runtime Editable Configurations
-----------------------------------

Options marked *Yes* in the **Runtime Mutable** column can be changed after import.
Use the :data:`cfg <neuralqx.configs.cfg>` singleton - do **not** write directly to
``os.environ`` at runtime, as that will not trigger hooks or validation.

**Persistent update** - change the value for the remainder of the process:

.. code-block:: python

    from neuralqx import cfg

    cfg.set('VERBOSE', False)   # or: cfg.VERBOSE = False

**Temporary patch** - restore the original value automatically on exit:

.. code-block:: python

    from neuralqx import cfg

    with cfg.patch('LOG_LEVEL', 'DEBUG'):
        # DEBUG logging active only inside this block
        ...
    # original LOG_LEVEL is restored here

**Thread-local override** - affects only the calling thread:

.. code-block:: python

    from neuralqx import cfg

    with cfg.thread_local_override('DEBUG', True):
        # DEBUG mode active only in this thread
        ...

.. seealso::

   The full configuration API - including hooks, fingerprinting, and introspection
   helpers - is documented in :mod:`neuralqx.configs`.


Unused Configurations
----------------------------

neuraLQX will warn you if it finds any environment variables which start with the ``NQX_`` prefix
when you import it. This is to help quickly recognise any typos that might have been made (e.g. in
a huge batch job on the HPC, you want to disable Rich printing but you accidentally
specified ``NQX_VERBOSW = False`` instead of ``NQX_VERBOSE = False`` in your script).

Note that the warning **is not fatal**, your code will continue, but it will be printed to you.
