.. _dist_cpu:

=======================================
CPU-based Parallelisation in neuraLQX
=======================================


.. raw:: html

    <hr style="margin: 30px 0;">

**Last updated: October 21, 2025**

In the following, we will go over installing neuraLQX with vanilla MPI (CPU based parallelisation).
The following documentation should guide you on doing this locally on your laptop, or on an HPC
cluster. For CUDA-aware MPI (GPUs), please see `this documentation <../gpu/index.html>`_.


Installation: laptop
=====================

.. note::

    The following installation instructions are for Linux and Mac based machines.

The MPI approach requires you to have MPI (who would have thought...). You can make sure ``mpi`` is
installed and can be used. To know if MPI is installed, try running the following command:

.. code-block:: bash

    mpicc --showme:link

If the command does not work, you will need to install MPI on your device. You can do this using
your favourite package manager.

On Mac, you can install MPI via HomeBrew using the following command

.. code-block:: bash

    brew install openmpi

while on Linux based systems, you can use the following

.. code-block:: bash

    # fedora
    sudo dnf install mpich

    # ubuntu/debian
    sudo apt-get install mpich

Once MPI is installed on your machine, you can install neuraLQX with MPI easily. You will first need
to create a virtual environment. You can skip this step if:

- you already have a virtual environment
- you are on some IDE like PyCharm, since the virtual environment already exists in your project

To create one, just run

.. code-block:: bash

    python -m venv <path to your venv>

where you should replace ``<path to your venv>`` with the path to your virtual environment and give it
some name. Once created, you can activate it using

.. code-block:: bash

    source <path to your venv>/bin/activate

Note that once again, if you are using an IDE, this is already done. You can just open the command line
in your IDE project.

You can, and should, upgrade `pip` as well as `setuptools` and `wheel` before starting using

.. code-block:: bash

    pip install --upgrade pip setuptools wheel

Now, you can install neuraLQX with MPI dependencies. In this pre-alpha phase, you need to fetch
the latest code from the shared GitHub repository. Once completed, just run the following command

.. code-block:: bash

    pip install "neuralqx[mpi]"

The ``"[mpi]"`` tells neuraLQX to install with MPI dependencies.

.. raw:: html

    <hr style="margin: 30px 0;">

Installation: HPC
==================

.. note::

    The following are install instructions compatible with the Woody cluster at the NHR@FAU HPC
    facility in Erlangen, Germany.

Here, we provide a guide to demonstrate the installation process of neuraLQX on a cluster with
CPU-based MPI enabled. Note that this can differ from cluster to another, and you should **not**
blindly copy commands, but rather check with your cluster's documentation prior. **This is supposed
to serve only as a guide of what needs to be done in principle.**

Installation on the HPC is a little bit more involved. Here, we will install MPI such that neuraLQX is
parallelisable over CPUs. We will do this via an interactive job on the HPC facility for ease of use.
After logging into the facility (e.g. via ``ssh``), you can (probably) allocate an
interactive job via

.. code-block:: bash

    salloc --constraint=icx --time=01:00:00 --nodes=1 --ntasks=8 --cpus-per-task=1

In the command above, we are allocating 1 node (``--nodes=1``), on which we will run 8 MPI tasks
(``--ntasks=8``, we dont need them for installation purposes though) and we will use 1 CPU per task
(``--cpus-per-task=1``). This means that we are in total allocating 8 CPUs for our computation.
The interactive job is valid for 1 hour (``--time=01:00:00``) and will utilise a specific compute
node on the cluster (``--constraint=icx``).

Once allocated, we need to load the following modules:

    (i) python
    (ii) gcc
    (iii) OpenMPI.

You can list all available modules on your node using

.. code-block:: bash

    module avail

**The results are cluster specific.** On this cluster, we can load python using

.. code-block:: bash

    module load python/3.12-conda

and load gcc using

.. code-block:: bash

    module load gcc/12.1.0

and finally OpenMPI using

.. code-block:: bash

    module load openmpi/5.0.6-gcc12.1.0

Now, these modules should be loaded. The next step is that we want to create a virtual environment for
python. On this cluster, the compute nodes are not connected to the internet directly. Therefore,
we have to run the following two commands to connect them to the internet

.. code-block:: bash

    export http_proxy=http://proxy.nhr.fau.de:80
    export https_proxy=http://proxy.nhr.fau.de:80

Once completed, we can create a virtual environment in which we will install all neuraLQX dependencies.
You can do this via

.. code-block:: bash

    python3 -m venv $WORK/venvs/neuralqx_dev

which will create a virtual environment located at ``$WORK/venvs/`` and called ``neuralqx_dev``. Now,
we can activate the that virtual environment using

.. code-block:: bash

    source $WORK/venvs/neuralqx_dev/bin/activate

We now need to install ``mpi4py`` or ``mpi4jax``, as neuraLQX (by default) does not install them.
This is because it can be cluster dependent. You will need to do the following:

- upgrade pip, setuptools and wheel using

 .. code-block:: bash

    pip install --upgrade pip setuptools wheel

- Remove any cached ``mpi4py`` and ``mpi4jax`` from build cache

 .. code-block:: bash

        pip cache remove mpi4py
        pip cache remove mpi4jax

- Install ``mpi4py``

 .. code-block:: bash

        pip install mpi4py cython

 .. warning::
        **On this specific cluster**, the above command will install a generic ``mpi4py`` which will
        **NOT** work on the cluster. To correctly install it, you will need to do the following (**instead
        of the above command**)

        .. code-block:: bash

            MPICC=$(which mpicc) pip install --no-cache-dir mpi4py cython


- Install ``mpi4jax``

 .. code-block:: bash

        pip install --upgrade --no-build-isolation "mpi4jax==0.7.1"

- Install the requirements for neuraLQX, which will install (by default) Jax for the CPU. This can
  be done by running the following command

 .. code-block:: bash

        pip install neuralqx

 .. warning:: Do **not** install with the ``"[mpi]"`` as in the case of the laptop!

Now, you should have neuraLQX installed as well as the correct MPI dependencies.

.. raw:: html

    <hr style="margin: 30px 0;">

Now you are ready to use neuraLQX with CPU-based MPI enabled. To see how to get started, read the
documentation provided `here <../using_parallelisation.html>`_.
