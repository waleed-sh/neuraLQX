.. _installing-neuralqx:

====================
Installing neuraLQX
====================


.. warning::

   **This package is still under development and is set to be released soon.** The install instructions below are currently
   non-functional!

This page explains how to install **neuraLQX** for typical *local* development and usage (laptop/workstation).
It focuses on the standard serial install and the most common optional extras, and intentionally avoids
cluster/HPC-specific workflows.

neuraLQX is distributed on PyPI and can be installed with ``pip``.

Prerequisites
-------------

Python version
~~~~~~~~~~~~~~

neuraLQX requires **Python ≥ 3.11**.

Virtual environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use an isolated environment so your scientific stack (JAX/NetKet/etc.) stays consistent:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate        # macOS/Linux
   # .venv\Scripts\activate         # Windows (PowerShell)

Then upgrade packaging tools:

.. code-block:: bash

   python -m pip install --upgrade pip setuptools wheel

Install the latest stable release
---------------------------------------------------

The simplest install is:

.. code-block:: bash

   pip install --upgrade neuralqx

This installs the latest stable release from PyPI.

Install from source (editable / nightly)
------------------------------------------

If you want the newest features (or you plan to contribute), install from the GitHub repository in editable
mode. The manuscript describes cloning and then installing with ``pip install -e .``.

.. code-block:: bash

   git clone https://www.github.com/waleed-sh/neuralqx
   cd neuralqx
   pip install -e .

Editable installs track your local working tree (so changes are picked up immediately).

.. warning::

    Nightly/source installs may expose unfinished features and can be less stable than a tagged PyPI release.

Optional install extras
-------------------------

Development / contributor dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you intend to run the test suite and contribute changes, install the ``dev`` extra:

.. code-block:: bash

   pip install --upgrade "neuralqx[dev]"

This pulls in additional testing dependencies that the project uses to validate changes.

MPI extra (local use only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

neuraLQX provides an ``mpi`` extra that installs MPI-related Python dependencies.
Using it requires a working MPI toolchain already present on your machine (for example, the ability to run
``mpicc``).

.. code-block:: bash

   # Quick check that an MPI compiler wrapper exists:
   mpicc --showme:link

.. code-block:: bash

   # Install neuraLQX with MPI-related dependencies:
   pip install --upgrade "neuralqx[mpi]"

.. note::

   This page does **not** cover cluster-specific MPI/GPU deployment details. For MPI install guides on cluster, please
   see the :ref:`parallel_nqx` page.


Docs dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you intend to contribute to the documentation of neuraLQX, you can install the dependencies required using

.. code-block:: bash

   pip install --upgrade "neuralqx[docs]"

This pulls in additional documentation dependencies that the project uses to build the Sphinx based documentation.


Profiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

neuraLQX ships with the core profiler enabled by configuration (``NQX_PROFILE=1``),
but GPU telemetry and certain advanced integrations require optional dependencies.

To install the profiling extras, use the ``profile`` extra:

.. code-block:: bash

   pip install --upgrade "neuralqx[profile]"

This installs NVML bindings (``nvidia-ml-py`` aka ``pynvml``) used for GPU metrics
(utilization, memory, power, temperature, clocks) and additional profiling helpers.

.. note::

   CPU/RAM metrics rely on ``psutil``. GPU metrics require the NVIDIA driver + NVML
   to be available on the system (typical on CUDA nodes).

Dependency stack and version alignment
--------------------------------------

neuraLQX sits on top of **NetKet** and **JAX**, and the first release is tested with a specific set of core
and auxiliary dependency versions.

neuraLQX requires the specific packages with the specific versions shown below

- NetKet ``>=3.17.0, <=3.18.1``
- mpi4jax ``0.7.1``
- mpi4py ``4.1.0``
- dash ``>=2.18.2``
- plotly ``>=5.24.1,<=6.0.0``
- optax ``>=0.2.4``

In practice:

- If you install from PyPI, the resolver will pull compatible versions automatically.
- If you are mixing environments (e.g. upgrading JAX manually), keep in mind that neuraLQX's supported NetKet
  version is constrained by parallelisation/backend compatibility discussed in :ref:`parallel_nqx`.

Device model (what "serial install" means)
---------------------------------------------

The default installation path installs neuraLQX in **serial mode**, meaning you do not explicitly enable a
distributed execution backend as part of installation.

On a normal workstation run, neuraLQX behaves like NetKet. Computation happens on the **JAX default device**
(typically ``jax.local_devices()[0]``).


Sanity check your installation
--------------------------------

After installing, verify that imports work and that JAX sees a device:

.. code-block:: bash

   python -c "import neuralqx as nqx; import jax; print('neuraLQX import OK'); print('JAX devices:', jax.local_devices())"

If you also want to confirm the key library versions:

.. code-block:: bash

   python -c "import neuralqx as nqx; print('neuraLQX:', nqx.__version__);"

Conventions used throughout the docs
------------------------------------

Across neuraLQX documentation, the following abbreviations are used:

.. code-block:: python

   import neuralqx as nqx
   import netket as nk
   import jax.numpy as jnp
   import numpy as np
   import scipy as sp

We also refer to:

- ``neuralqx.experimental`` as ``nqxx`` (neuraLQX's experimental API)
- ``netket.experimental`` as ``nkx`` (NetKet's experimental API)

.. important::

   When using neuraLQX, you **must** import it **before** importing NetKet or JAX. neuraLQX internally may set some
   environment variables for both NetKet and JAX. If you import either one of them *before* neuraLQX, these variables
   will be ineffective, resulting in incorrect behaviour.


Updating and removing neuraLQX
--------------------------------

Upgrade to the newest compatible release:

.. code-block:: bash

   pip install --upgrade neuralqx

If you installed from source and want to pull updates:

.. code-block:: bash

   cd neuralqx
   git pull
   pip install -e .

To uninstall:

.. code-block:: bash

   pip uninstall neuralqx
