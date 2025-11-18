Configuration Options
=====================

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
     - No
   * - ``NQX_CACHE``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Enable caching when possible
     - No
   * - ``NQX_MPI``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Enable MPI (CPU) for neuraLQX
     - No
   * - ``NQX_MPI_CUDA``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Enable CUDA-aware MPI for neuraLQX
     - No
   * - ``NQX_JAX_DISTRIBUTED``
     - ``int``
     - ``0`` | ``1``
     - ``0``
     - Enable Jax distributed computations for neuraLQX
     - No

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