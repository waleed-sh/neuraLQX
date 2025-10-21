=======================================
GPU-based Parallelisation in neuraLQX
=======================================


.. raw:: html

    <hr style="margin: 30px 0;">

**Last updated: October 21, 2025**

In the following, we will go over installing neuraLQX with CUDA-aware MPI (GPU based parallelisation)
on HPC clusters. For instructions on installing CPU-based MPI, please see `this documentation <../cpu/index.html>`_.


CUDA-aware MPI for HPC systems
==================================

.. note::

    The following are install instructions compatible with the TinyGPU cluster at the NHR@FAU HPC
    facility in Erlangen, Germany.

Installing neuraLQX with CUDA-aware MPI follows similar steps to vanilla CPU-based MPI. However, we
need to install certain dependencies. **As each cluster's available modules are different, neuraLQX
does not do this automatically.** Further, we need to utilise more modules than done in the CPU case.

To start, we will allocate an interactive job using the following command

.. code-block:: bash

    salloc.tinygpu --time=01:00:00 --gres=gpu:v100:1 -p v100 --ntasks=1 --cpus-per-task=8

On this specific HPC cluster, this will allocate 1 Nvidia V100 GPU (``--gres=gpu:v100:1``) and choose
the corresponding V100 partition (``-p v100``). By default (on this cluster), each GPU comes with 8 CPU cores (16 threads)
(which we explicitly also specify using ``--cpus-per-task=8``) and we specify 1 MPI task (``--ntasks=1``).
The interactive job is valid for 1 hour (``--time=01:00:00``).

Once allocated, you will be greeted with the following message (or something similar depending on your
cluster/GPU vendor).

.. code-block:: bash

    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 570.158.01             Driver Version: 570.158.01     CUDA Version: 12.8     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  Tesla V100-PCIE-32GB           On  |   00000000:18:00.0 Off |                    0 |
    | N/A   35C    P0             27W /  250W |       0MiB /  32768MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+


What is important here to take note of is the CUDA version at the top right. For this case, it is ``12.8``.
We will keep that in mind for now.

To start the installation, we need to load some modules. You can generally list all available modules by using

.. code-block:: bash

    module avail

**The results are cluster specific**. On this cluster, we can load python using

.. code-block:: bash

    module load python/3.12-conda

Next, we need to load some modules depending on the CUDA version shown above. For this cluster, they
are not available by default, but through some ``spack`` loaded modules. To load these ``spack`` modules
we first execute the following command

.. code-block:: bash

    module load 000-all-spack-pkgs/0.23.1

Then, we now load ``gcc`` using

.. code-block:: bash

    module load gcc/13.3.0-gcc13.3.0-a4xdbwt

Next, we load the CUDA module using

.. code-block:: bash

    module load cuda/12.8.0-gcc13.3.0-vnhbqjm

We then load the CUDA Deep Neural Network library (``cudnn``) using

.. code-block:: bash

    module load cudnn/9.2.0.82-12-gcc13.3.0-cuda-wmejh6k

and lastly, CUDA-aware openMPI using

.. code-block:: bash

    # This is the hash corresponding to the ucx module
    spack load /gchn6sb

    # This is the hash corresponding to the CUDA-aware openMPI module
    spack load /tiyr6xl

.. important::

    This cluster did not have any CUDA-aware MPI modules already installed. Which means that
    this CUDA-aware openMPI version was built using a **local** spack (the process of which
    can be tedius and we do not show here) along with ucx to enable GPU-aware communication
    (direct device-to-device copies via CUDA IPC/NVLink on-node, or RDMA off-node if available).
    As a "belt-and-suspenders" approach, we also amend the ``LD_LIBRARY_PATH`` to point to those
    loaded modules and not other cluster-default modules. In that case, the following command needs
    to be run to add the local CUDA-aware openMPI to the path

    .. code-block:: bash

        # UCX and OpenMPI install prefixes
        PREFIX_UCX=$(spack location -i /gchn6sb)
        PREFIX_OMPI=$(spack location -i /tiyr6xl)

        # Prepend their libs to LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=$PREFIX_UCX/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=$PREFIX_OMPI/lib:$OMPI_PREFIX/lib/openmpi:$LD_LIBRARY_PATH

    Note that this can be different based on your cluster. **Please check your cluster's documentation
    for details.**

Now, we have all the modules needed to proceed.

.. note::

    While the results are cluster specific, what you need to load (aside from python) are the
    modules ``gcc``, ``cuda``, ``cudnn`` and ``openmpi`` **all with the same version**. In the above, that was ``13.3.0``.
    Typically, this is dictated by the CUDA version. In the greeting message shown above after job allocation,
    we can maximally load CUDA 12, and on this cluster, that was compiled with ``gcc`` version 13.3.0,
    hence everything else needs to be with that specific ``gcc`` version.

Now, we can proceed to create a virtual environment. On this cluster, the compute nodes are not connected
to the internet directly. Therefore, we have to run the following two commands to connect them to the
internet

.. code-block:: bash

    export http_proxy=http://proxy.nhr.fau.de:80
    export https_proxy=http://proxy.nhr.fau.de:80

Once completed, we can create a virtual environment in which we will install all neuraLQX dependencies.
You can do this via

.. code-block:: bash

    python3 -m venv $WORK/venvs/neuralqx_dev

which will create a virtual environment located at ``$WORK/venvs/`` and called `neuralqx_dev``. Now, we can
activate it using

.. code-block:: bash

    source $WORK/venvs/neuralqx_dev/bin/activate

We can then start installing neuraLQX. Unlike in the vanilla MPI installation in the section above,
we will first start in the other direction. Namely, installing neuraLQX first. You will need to

- upgrade pip, along with other things, using
 .. code-block:: bash

    pip install --upgrade pip setuptools wheel

- Remove any cached ``mpi4py`` and ``mpi4jax`` from build cache
 .. code-block:: bash

    pip cache remove mpi4py
    pip cache remove mpi4jax

- Install neuraLQX **without** the ``"[mpi]"`` flag using
 .. code-block:: bash

    pip install --upgrade neuralqx

- Install **CUDA aware Jax, with this specific version shown below otherwise you get a conflict with ``mpi4jax`` later**

 .. code-block:: bash

    pip install --upgrade "jax[cuda12_local]"==0.5.2

 .. note::

    Here, the ``[cuda12_local]`` version should match the available/loaded CUDA module above. For example,
    if you have CUDA 11 loaded, this would be ``[cuda11_local]``.

- Install ``mpi4py``
 .. code-block:: bash

    pip install mpi4py

 .. important::

    **On this specific cluster**, the above command will install a generic ``mpi4py`` which will
    **NOT** work on the cluster. To correctly install it, you will need to do the following (instead
    of the above command)

    .. code-block:: bash

        MPICC=$(which mpicc) pip install --no-cache-dir mpi4py cython

- Lastly, install ``mpi4jax``
 .. code-block::

    pip install --upgrade --no-build-isolation "mpi4jax==0.7.1"

Now, you should have neuraLQX installed with CUDA-aware MPI, which means you can use MPI based GPU
computations for your work.

To see how to get started, read the documentation provided `here <../using_parallelisation.html>`_.