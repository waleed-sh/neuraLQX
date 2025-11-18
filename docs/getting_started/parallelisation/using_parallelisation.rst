=================================
Using neuraLQX in Parallel
=================================

Using neuraLQX with MPI
========================

Now that you have properly set up neuraLQX with MPI enabled, you can start using neuraLQX parallelisation.
The first step of doing so is enabling parallelisation for neuraLQX using a configuration variable.

For **CPU-based** parallelisation, this can be done in your python script as follows

.. code-block:: python

    # enable CPU-based MPI parallelisation before importing neuraLQX
    import os

    os.environ['NQX_MPI'] = '1'

    # now you can import neuraLQX
    import neuralqx as nqx

or in shell as

.. code-block:: bash

    export NQX_MPI=1
    python your_script.py

where ``your_script.py`` is some python script which uses neuraLQX.

For CUDA-aware MPI enabled, the environment variable to be set is the one below

.. code-block:: python

    # enable CUDA-aware MPI parallelisation before importing neuraLQX
    import os

    os.environ['NQX_MPI_CUDA'] = '1'

    # now you can import neuraLQX
    import neuralqx as nqx

or in shell as

.. code-block:: bash

    export NQX_MPI_CUDA=1
    python your_script.py

where ``your_script.py`` is some python script which uses neuraLQX.

Note that in all cases, this has to be done **before** any import of neuraLQX is done. Once the
configuration variable is set, you do not need to do anything else. neuraLQX, and subsequently NetKet,
now have MPI enabled **for CPU-based parallelisation**. This means that you do **not** have to
specify the NetKet configuration variable ``NETKET_MPI`` to ``1``, neuraLQX does that internally for you.

Running with MPI on your laptop
--------------------------------

Running parallel neuraLQX code is simple. For example, the ``neuralqx.solver.Solver`` class, responsible
for handling the entire VMC process, already takes care of everything MPI for you. **Essentially, you
can simply keep your normal (non-parallel) Python script.** However, instead of doing the following
to run your script

.. code-block:: bash

    python your_parallel_script.py

you would run with MPI as follows

.. code-block:: bash

    mpirun -n 8 python your_parallel_script.py

The above command will allocate 8 MPI ranks on 1 node (your laptop) and run your code with MPI where
each task is by default bound to 1 CPU.

The only thing you need to be careful about when running MPI enabled neuraLQX code is any I/O
processes. Specifically, you should only do any I/O from the master (rank 0) worker. neuraLQX
provides you with some functionality to make that easier. Through ``neuralqx.utils.mpi``, you have
available the following

- ``is_master()``: a function that returns ``True`` if you are on rank 0 worker
- ``rank``: the rank of the current worker
- ``barrier()``: a function that makes sure that all workers synchronise at it when called before proceeding to any further code

among other functionalities in that module. Therefore, if you wish to compute and output/print some
expectation values after a simulation is done, you will follow the logic provided here:

.. code-block:: python

    # ... your solver.run() is assumed to be executed and done by this point

    # [1] start a barrier to synchronise all workers
    # the following import can also be placed at the top of the code
    from neuralqx.utils.mpi import barrier
    barrier()

    # [2] compute expectation values on ALL workers
    some_expectation = solver.variational_state.expect(op)

    # [3] print only from the rank 0 worker
    # the following import can also be placed at the top of the code
    from neuralqx.utils.mpi import is_master
    if is_master():
        print(f"Expectation value = {some_expectation}")

    # [4] print the same as above, but automatically only from rank0
    # the following import can also be placed at the top of the code
    from neuralqx.utils.mpi import print0
    print0(f"Expectation value = {some_expectation}")

    # [5] syncrhonise once more
    barrier()

When doing any I/O, it is important to use the ``barrier()`` in order to avoid any race conditions
between different workers (different workers attempting to modify or access the same object/file).
This is the reason for ``[1]`` in the code above.

When computing quantities such as expectation values
(think anything that has chains or neural networks), it is important to allow all workers to access
that part of the code. This is because internally that code is parallelised and in that parallel
routine, there often are barriers placed to ensure synchronisation. If all workers do not access ``[2]``
in the code above for example, then this will cause a deadlock and the execution will hang. That is
why ``[2]`` is not safe-guarded and is run on all workers.

When printing, however, as shown in ``[3]`` above, we only want to print from the master (rank 0) worker.
That is why we guard that code with a conditional with the ``is_master()`` function. Such I/O operations,
including writing simple things to disk, should be done in a similar manner. Note that as shown in
``[4]``, simple I/O such as printing can be done automatically from the rank 0 worker using the
``print0`` function from ``neuralqx.utils.mpi``.

Once any I/O is done, all workers reconvene at the barrier again as shown in ``[5]`` above.
This ensures that all workers proceed to the next part of the code in sync.


Running with MPI on an HPC
------------------------------

neuraLQX can run in parallel either on CPU or GPU using MPI. In what follows, we show a standard use-case
for both.

CPU-based
^^^^^^^^^^^^^^^

.. note::

    The following is an example for the Woody cluster at the NHR@FAU HPC facility in Erlangen, Germany

To run your code on an HPC, you typically submit a batch job for MPI enabled scripts. It may be that
your facility does not allow MPI to run on interactive jobs. An example batch script would be as follows:

.. code-block:: bash

    #!/bin/bash -l
    #
    #SBATCH --nodes=1                               # Use 1 physical node
    #SBATCH --ntasks-per-node=32                    # 32 tasks per node
    #SBATCH --cpus-per-task=1                       # Allocate 1 CPUs per task
    #SBATCH --time=01:30:00                         # Max runtime (hh:mm:ss)
    #SBATCH --export=NONE                           # Don’t export current env vars
    #SBATCH --mail-type=ALL                         # Send email on ALL events
    #SBATCH --hint=nomultithread                    # Disable SMT / hyper-threading
    #SBATCH --mail-user=USER@DOMAIN.com             # User email
    #SBATCH --constraint=icx                        # Run on the IceLake cores (facility dependent)

    unset SLURM_EXPORT_ENV

    # for Slurm version >22.05: cpus-per-task has to be set again for srun (facility dependent)
    export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

    # for more efficient computing using OpenMP
    export OMP_PLACES=cores
    export OMP_PROC_BIND=true

    # load the needed modules
    module purge                                    # Delete any loaded modules if any
    module load python/3.12-conda                   # Load Python
    module load gcc/12.1.0                          # Necessary for MPI
    module load openmpi/5.0.6-gcc12.1.0             # Necessary for MPI
    source $WORK/venvs/neuralqx_dev/bin/activate    # Activate your environment which contains all
                                                    # requirements installed

    # run your code
    # do not specify any options for mpirun which differ from the SBATCH above
    srun python $WORK/neuralqx_dev/some_CPU_parallel_code.py

In the above example, reading from the top, we have done the following:

- allocated 1 physical node using ``--nodes=1``
- allocated 32 MPI ranks on each node using ``--ntasks-per-node=32``
- allocated 1 CPU per MPI rank with ``--cpus-per-task=1``, every rank runs single-threaded code and you
  scale with more ranks, not threads
- allocated a compute time of an hour and a half using ``--time=01:30:00``
- disabled hyper-threading to avoid over-subscription by Jax using ``--hint=nomultithread``
- other "decorative" commands include:

    - ``--constraint=icx`` specifies on this specific cluster, which compute nodes to use

    - ``--mail-type=ALL`` will notify the email specified in ``--mail-user`` with everything (job started, done, failed, etc.)

In this setup, we have chosen 1 physical node and allocated 32 MPI ranks per node, each allocating
1 CPU. This gives us a total of 32 tasks. Slurm (your HPC "batch managing system") will start one
Python process per task, each becoming an MPI rank.

Once the "HPC environment" is set up, we then loaded all needed modules. These included ``python``, ``gcc`` and
``openmpi``. **Note that these must match the modules used during installation.** The last thing is to
activate your virtual environment, which we do via ``source $WORK/venvs/neuralqx_dev/bin/activate`` and
then run your parallel neuraLQX python script. Unlike on your laptop, you do not use ``mpirun`` here.
HPC facilities have their own specifications on running parallel code. On this facility (and most
likely on yours too **but check first**) you use `srun`.

.. note::

    On some HPC facilities, it might be that ``srun`` is a little funny, and does not do the proper
    "binding" it is supposed to do, and you would be recommended to use ``mpirun`` instead. **This is
    the case for the specific cluster being demonstrated here.**

.. warning::
    If you use ``mpirun``, you should not specify any further options **especially**
    that may differ from the booked tasks, etc. in the batch script.

Once you write your batch script like above, you can save it as some ``some_name.sh`` and then submit
a batch job as

.. code-block:: bash

    sbatch some_name.sh

Again, **how you submit jobs can be cluster dependent.** Check with your facility first. **It is always a
bad idea to copy/paste code ESPECIALLY on HPC facilities.**

.. important::

    When loading python, gcc and OpenMP in your batch script, you **must** use the same builds
    used during the installation process.


GPU-based
^^^^^^^^^^^^^

.. note::

    The following is an example for the TinyGPU cluster at the NHR@FAU HPC facility in Erlangen, Germany

This is done exactly in the same way as the CPU-based parallelisation, except we need to specify more
configuration options and also load the CUDA related packages we used during the installation process.
Here, we will be more explicit. The job script would look like this

.. code-block:: bash

    #!/bin/bash -l
    #
    #SBATCH --gres=gpu:v100:2                               # Use 2 Nvidia V100 GPUs
    #SBATCH --ntasks=2                                      # 2 tasks, one per GPU
    #SBATCH --cpus-per-task=2                               # Allocate 8 CPUs per task
    #SBATCH --time=01:30:00                                 # Max runtime (hh:mm:ss)
    #SBATCH --export=NONE                                   # Don’t export current env vars
    #SBATCH --mail-type=ALL                                 # Send email on ALL events
    #SBATCH --hint=nomultithread                            # Disable SMT / hyper-threading
    #SBATCH --mail-user=USER@DOMAIN.com                     # User email
    #SBATCH --partition=v100                                # Run on the v100 partition

    unset SLURM_EXPORT_ENV

    # For Slurm version >22.05: cpus-per-task has to be set again for srun (facility dependent)
    export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

    # For more efficient computing using OpenMP
    export OMP_PLACES=cores
    export OMP_PROC_BIND=true

    # "Export" the local spack from which the CUDA-aware MPI and ucx is loaded
    . $WORK/spack/share/spack/setup-env.sh

    #
    #
    # Now load the needed modules
    # These should be the same ones used during the install

    # Delete any loaded modules carried over, if any
    module purge

    # Load the spack modules from which gcc and co. will be loaded
    module load 000-all-spack-pkgs/0.23.1

    # Load Python
    module load python/3.12-conda

    # Load gcc
    module load gcc/13.3.0-gcc13.3.0-a4xdbwt

    # Load CUDA compiled with the same gcc version
    module load cuda/12.8.0-gcc13.3.0-vnhbqjm

    # Load cuDNN compiled with the same gcc version
    module load cudnn/9.2.0.82-12-gcc13.3.0-cuda-wmejh6k

    # Load the locally installed ucx module
    spack load /gchn6sb

    # Load the locally installed CUDA-aware openMPI module
    spack load /tiyr6xl

    # Create prefixes for the ucx and CUDA-aware openMPI
    PREFIX_UCX=$(spack location -i /gchn6sb)
    PREFIX_OMPI=$(spack location -i /tiyr6xl)

    # Prepend them to LD_LIBRARY_PATH, for safety
    export LD_LIBRARY_PATH=$PREFIX_UCX/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$PREFIX_OMPI/lib:$OMPI_PREFIX/lib/openmpi:$LD_LIBRARY_PATH

    # Activate your environment which contains all requirements installed
    source $WORK/venvs/neuralqx_dev/bin/activate

    # run your code
    # do not specify any options for mpirun which differ from the SBATCH above
    # here we are more explicit with mpirun for portability
    mpirun -np "${SLURM_NTASKS}" \
      --map-by ppr:"${SLURM_NTASKS_PER_NODE}":node \
      --bind-to core \
      --mca pml ucx \
      --mca osc ucx \
      --mca btl ^tcp \
      python $WORK/neuralqx_dev/some_GPU_parallel_code.py

In the above example, reading from the top, we have done the following:

- allocated 2 Nvidia V100 GPUs ``--gres=gpu:v100:2``, each comes with 8 CPU cores associated
- allocated 2 MPI ranks using ``--ntasks=2``. **This should match the number of GPUs.**
- allocated 2 CPU cores per MPI-rank/GPU with ``--cpus-per-task=2``, every rank runs single-threaded code and you scale with more ranks, not threads
- allocated a compute time of an hour and a half using ``--time=01:30:00``
- disabled hyper-threading to avoid over-subscription by Jax using ``--hint=nomultithread``
- other "decorative" commands include:

  - ``--partition=v100`` specifies on this specific cluster, which partition to use
  - ``--mail-type=ALL`` will notify the email specified in ``--mail-user`` with everything (job started, done, failed, etc.)

- For the ``mpirun``, we are more explicit for portability but also for ensuring that each MPI ranks sees only 1 GPU (it may be that your HPC already does this for you). Here, we use the following ``mpirun`` commands:

  - ``-np "${SLURM_NTASKS}"``: run as many MPI ranks as requested in the Slurm job. Each rank typically gets bound to one GPU when using CUDA-aware MPI.
  - ``--map-by ppr:"${SLURM_NTASKS_PER_NODE}":node``: (``ppr`` = Process Per Resource) here, ``ppr:1:node`` means "1 MPI rank per GPU per node." With boths number of tasks (`--ntasks`) and the number of GPUs allocated being 2 in this example, each of the 2 GPUs will get exactly 1 rank.
  - ``--bind-to core``: ensures each MPI rank is pinned to a physical CPU core (or the cores requested via ``--cpus-per-task``), reducing cache contention and improving performance consistency.
  - ``--mca pml ucx``: chooses the UCX point-to-point messaging layer (PML) instead of the default. UCX enables GPU-aware communication (direct device-to-device copies via CUDA IPC/NVLink on-node, or RDMA off-node if available).
  - ``--mca osc ucx``: uses UCX for one-sided communication (OSC), e.g. MPI RMA operations (``MPI_Put``, ``MPI_Get``, etc.), ensuring GPU memory can be directly used there too.
  - ``--mca btl ^tcp``: disables the legacy TCP byte transfer layer (BTL). This avoids falling back to slower socket-based communication, forcing UCX to handle all transfers (shared memory, CUDA IPC, or network fabrics).

You can run this batch script as done in the CPU-based case.

.. note::

    The more complicated ``mpirun`` command is due to the fact that on some clusters (like on the
    one used in the documentation here), the standard ``srun`` command will **not** bind processes to GPUs
    and will result in incorrect execution. Therefore, the more explicit command compared to the CPU case.

.. important::

    The number of MPI ranks **should match the number of allocated GPUs**, this is because MPI
    cannot see more than 1 GPU, and internally, both neuraLQX and NetKet make sure of that.

.. note::

    In this example, we have shown how to use neuraLQX to use multi-GPUs *on one node*. Running
    on multi-node/multi-GPU is slightly more involved but possible.

.. note::

    In this example, we have CUDA-aware MPI installed with UCX support. Depending on your
    facility, you may have to install this as well if not already installed, or you might have to use
    another PML. Please contact your facilities' support for detailed information on what should be done.


Verifying MPI with neuraLQX
------------------------------

You can verify if you have installed everything correctly and that neuraLQX can now run with MPI by
doing the following.

.. code-block:: python

    # set the MPI configuration variable to 1 before importing neuraLQX
    import os
    os.environ['NQX_MPI'] = '1'

    # import neuraLQX
    import neuralqx as nqx

    # check for proper MPI installation
    nqx.utils.mpi.check_mpi()

The output should be something that looks as follows, if everything went fine

.. code-block:: bash

    ===============================================================
                             MPI Summary
    ===============================================================
      Number of nodes                        : 1
      Total MPI ranks (tasks)                : 2
      Total CPUs available per node (rank 0) : 8
      CPUs per MPI task                      : 1
      Total CPU cores used across ranks      : 2
      GPU allocation mode                    : JAX_VISIBLE_DEVICES
      Total GPUs allocated (env/Slurm)       : unknown
    ===============================================================

    Details:
    ---------------------------------------------------------------
    CUDA-aware MPI               : False (ENV_VAR)
    cpus_per_node (rank 0)       : 8
    cpus_per_task                : 1
    gpu_allocation_mode          : JAX_VISIBLE_DEVICES
    gpu_sanity_check             : OK
    jax_available                : True
    mpi4jax_available            : True
    mpi4py | MPI library_version : Open MPI v5.0.8, package: Open MPI brew@Ventura-arm64.local Distribution, ident: 5.0.8, repo rev: v5.0.8, May 30, 2025
    mpi4py | MPI version         : (3, 1)
    mpi4py_available             : True
    n_nodes                      : 1
    n_ranks                      : 2
    python_implementation        : CPython
    python_version               : 3.10.1
    total_cpu_cores_used         : 2
    total_gpus_allocated (env)   : unknown
    ---------------------------------------------------------------

In the above output, we have allocated 1 node (a laptop) on which we want to have 2 MPI tasks
on it, each running on (bound to) 1 CPUs. Note that by default, neuraLQX will set ``export OMP_NUM_THREADS=1``
to avoid over-subscription by Jax. You can run the same verification on your laptop.

.. note::

    If you chose CUDA-aware MPI, then you should be able to see all GPU related data come to life.


Common pitfalls
================

Big systems on too many ranks
--------------------------------

Every MPI rank is an independent process with its own address space. That means that when you create
your system using neuraLQX, every rank will have its own copy of the entire system. In principle,
``mpi4py`` and ``mpi4jax`` ``broadcast`` functionalities can move python objects or Jax arrays, but it
cannot give ranks a view into rank‑0's memory. After the broadcast each rank holds its own copy.

When comupting expectation values, all ``LocalOperator`` types go through a ``setup`` process. In this
process, several look-up tables are computed which hold, for example, all non-zero matrix elements of
all sub-operators/matrices stored in the ``LocalOperator`` as well as their local indices. If your
``LocalOperator`` that you are trying to minimise is *huge* (aka consists of thousands of matrices),
these look-up tables are going to require a lot of memory.

Typically on HPC facilities, you allocate a certain amount of CPUs in a node (simple example) where
every core will come with an amount of RAM (cluster/facility dependent). One common pitfall, in case
of a large ``LocalOperator``, is to allocate 32 MPI ranks on one node, and give each rank 1 CPU. If
every core comes with, say 8GB of RAM, that means each MPI rank will have 8GB of RAM to access. If
these look-up tables are huge in memory, this will cause an ``OOM`` and your job will be killed.

Workaround
^^^^^^^^^^

Use fewer number of MPI ranks and give each rank more CPUs, hence more RAM. This means that your
computation wont be as efficient as it could be, but it will at least run... In practice, the extra
cores are going to be wasted quota since they will not be actually utilised in the computation. This
is because Jax will run on only one CPU.

Too many Markov chains
------------------------

Suppose you have a system where you have chosen 450 samples for your Markov chain Monte-Carlo process
and have split up these samples over 10 chains. You would typically do this

.. code-block:: python

    solver.set_sampler(
        sampler_type = "Metropolis Local",
        number_of_samples = 450,
        number_of_chains = 10,
    )

In serial mode, this should be fine. You will have 450 samples distributed over 10 chains, which
means 45 samples per chain. In MPI enabled neuraLQX, the ``number_of_chains`` parameter refers to the
number of chains **per MPI rank**. Therefore, if you supply ``number_of_chains = 10``, and you have,
let's say, 32 MPI ranks, you will end up with ``n_ranks * number_of_chains`` many *total* chains for your
simulation. In this case, you will have 320 chains, with each MPI rank having 10 chains. Clearly, this
is *way* more chains than conducted in serial mode. Therefore, you have to be more careful with the
``number_of_chains`` supplied when running with MPI.


Exporting and importing states
---------------------------------

neuraLQX enables you to export and import previously exported states using
``export_serialized_state()`` and ``import_serialized_state()`` from the ``neuralqx.solver.Solver``
class, respectively. When using MPI, the chains are distributed across all MPI ranks. Currently, due
to some internal issues, **when exporting a state from a simulation conducted with MPI enabled, you can only load
it with the same MPI configuration**. Further, the MPI configuration **must match** the one the state
was exported from (e.g. same number of nodes and MPI ranks per node).

For example, if you have done a simulation with 4 MPI ranks,
and have exported a state during that simulation, you can only import that state in a simulation
conducted on the same number of MPI ranks. This implies that any states exported in simulations on an
HPC where a number of MPI ranks were used **cannot** be imported on your laptop.

.. note::

    This is a temporary issue, we are actively working on resolving it!

Notes on using GPUs
=====================

In principle, it is feasible to use MPI with GPUs as shown above. However, as you saw this is more
invasive, and not necessarily what MPI excels in. Unless for very specific situations, where
you are using an _extremely_ large network or an _extremely_ large amount of samples, you will often
be fine parallelising over the CPU.

.. note::

    We are currently working on finishing up GPU support in neuraLQX using Jax's sharding
    mechanism. Simulations should be consistently around 10% faster on GPU and *substantially* easier
    to set up compared to using CUDA-aware MPI.


Testing Jax distributed locally
------------------------------------

**In preparation for Jax distributed GPU support**, we borrow from NetKet the ``djaxrun`` code.
This enables you to run distributed Jax code locally/on a single node. It works by pretending that it is running
under SLURM, setting up the environment variables and launching the command in multiple processes.

.. important::

    **This is strictly for testing your Jax distributed code.** This should **not** be used for
    conducting simulations as it is **significantly** slower than utilising GPUs.

To use ``djaxrun``, once you install neuraLQX via `pip`, you can do the following

.. code-block:: bash

    djaxrun --simple -n 2 python your_script.py

which will simulate running your script on 2 nodes.


Citing
==========

If you use either CPU or GPU enabled neuraLQX for your research, then you are going to use ``mpi4jax``.
This is developer by researchers. **Please cite it as**

.. code-block:: bash

    @article{mpi4jax:2021,
        doi = {10.21105/joss.03419},
        url = {https://doi.org/10.21105/joss.03419},
        year = {2021},
        publisher = {The Open Journal},
        volume = {6},
        number = {65},
        pages = {3419},
        author = {Dion Häfner and Filippo Vicentini},
        title = {mpi4jax: Zero-copy MPI communication of JAX arrays},
        journal = {Journal of Open Source Software}
    }
