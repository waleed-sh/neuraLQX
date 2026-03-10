.. _cfgs:

============================
Configuration Options
============================

This page lists all **environment variables** available in neuraLQX.

Each variable allows you to modify the runtime behaviour of the package, enable optional features,
or adjust numerical and parallelisation settings.

.. note::

   Environment variables must be set **before importing** neuraLQX in your python script or shell
   session, unless stated otherwise.

   Example (bash):

   .. code-block:: bash

      export NQX_EXPERIMENTAL=1
      export NQX_LOG_LEVEL=DEBUG
      python your_script.py

   Example (python script):

   .. code-block:: python

        import os
        os.environ['NQX_EXPERIMENTAL'] = '1'

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
   * - ``NQX_VERBOSE``
     - ``bool``
     - ``False`` | ``True``
     - ``True``
     - Enable or disable Rich console printing.
     - Yes
   * - ``NQX_LOG_LEVEL``
     - ``str``
     - ``DEBUG`` | ``INFO`` | ``WARNING`` | ``ERROR`` | ``CRITICAL``
     - ``INFO``
     - Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
     - No
   * - ``NQX_EXPERIMENTAL``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Enable experimental functions throughout the neuralqx and netket packages
     - Yes
   * - ``NQX_TESTING``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Relax some neuraLQX features for testing purposes.
     - Yes
   * - ``NQX_CACHE``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Enable caching when possible
     - No
   * - ``NQX_ENABLE_X64``
     - ``int``
     - ``0`` | ``1``
     - ``1``
     - Enable x64 precision throughout neuraLQX, NetKet and Jax
     - No

.. important::

   neuraLQX also has profiling specific configurations options. For that, please see the :doc:`profiling page <guides/profiling>`.

.. raw:: html

   <hr style="margin: 30px 0;">


Runtime Editable Configurations
-----------------------------------

Some configuration variables (e.g. ``NQX_VERBOSE``) can be changed during runtime. However, this
should be done through **neuraLQX's internal configuration manager** as follows

.. code-block:: python

    import os

    # no Rich console printing will be displayed
    os.environ['NQX_VERBOSE'] = 'False'

    import neuralqx as nqx

    # some computations

    from neuralqx import cfg

    cfg.set('VERBOSE', 'True')

    # now, Rich console printing will be enabled


Unused Configurations
----------------------------

neuraLQX will warn you if it finds any environment variables which start with the ``NQX_`` prefix
when you import it. This is to help quickly recognise any typos that might have been made (e.g. in
a huge batch job on the HPC, you want to disable Rich printing but you accidentally
specified ``NQX_VERBOSW = False`` instead of ``NQX_VERBOSE = False`` in your script).

Note that the warning **is not fatal**, your code will continue, but it will be printed to you.
