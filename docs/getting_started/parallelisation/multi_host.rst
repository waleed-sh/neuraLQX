.. _parallel_nqx_multi_host:

=============================================
JAX Distributed Parallelisation (Multi-Host)
=============================================

This page describes multi-host execution in neuraLQX: multiple nodes, each with
one or more GPUs, coordinated through JAX distributed runtime.

neuraLQX follows the same modern JAX-sharding-first parallel model used by
NetKet.

If you are using older neuraLQX versions prior to 1.1.0 that still use MPI-era code,
refer to :doc:`mpi_deprecated/index`.


Scope
-------

Multi-host means the global job spans more than one machine.

Typical mapping on clusters:

- one process per GPU
- multiple processes per host
- one global process group spanning all hosts

Model and solver code stays mostly unchanged, launcher topology and
distributed-safe I/O are the operational focus.


Automatic Distributed Initialisation In neuraLQX
---------------------------------------------------

neuraLQX performs JAX distributed initialization internally during import when
distributed environment signals indicate a multi-process launch.

Detected signals include:

- ``JAX_COORDINATOR_ADDRESS`` present
- ``JAX_PROCESS_COUNT > 1``
- ``SLURM_NTASKS > 1``
- ``OMPI_COMM_WORLD_SIZE > 1``
- ``PMI_SIZE > 1``

Therefore, in standard scheduler launches, you usually should not call
``jax.distributed.initialize()`` yourself inside your simulation script.


Launch Pattern
----------------

Use your scheduler to launch the full process set. Example (SLURM):

.. code-block:: bash

    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=4
    #SBATCH --cpus-per-task=2

    srun python run_simulation.py


.. important::

   Here, you must set the number of tasks per node to be the number of gpus per task.


neuraLQX will automatically instruct JAX to use available devices, with the default choice being GPUs. If
no GPUs are detected, JAX will fallback to the next available, which is typically CPUs. This is why we
recommend calling using

.. code-block:: python

    from neuralqx.utils import distributed as dist

    dist.check_distributed()

At the start of your script to make sure that JAX is correctly utilising the resources.


Minimal Script
----------------

.. code-block:: python

    import neuralqx as nqx
    from neuralqx.utils import distributed as dist

    # Optional summary/diagnostics from rank 0.
    dist.check_distributed()

    # Build your simulation as in single-host mode.
    # graph = ...
    # hilbert = ...
    # lqx = ...
    # solver = nqx.solver.Solver(lqx, output_path="out")
    # solver.set_sampler(...)
    # solver.set_optimizer(...)
    # solver.set_network(...)
    # solver.run(n_iters=500)

    if dist.is_global_master():
        print("Multi-host run completed.")


Diagnostics
-------------

Print these once near startup to validate launcher wiring:

.. code-block:: python

    import jax

    print(
        f"[{jax.process_index()}/{jax.process_count()}] "
        f"devices={jax.devices()} local={jax.local_devices()}",
        flush=True,
    )

Expected:

- process count matches scheduler allocation
- each process sees expected local device(s)
- process/device layout matches job request


Custom I/O Rules
------------------

Built-in neuraLQX solver exports/imports are distributed-aware. For custom
artifacts, keep explicit synchronization and process-0 writes:

.. code-block:: python

    import json
    from neuralqx.utils import distributed as dist
    from neuralqx.utils.distributed import safe_replicate_for_io

    obs = solver.variational_state.expect(op)
    obs_host = safe_replicate_for_io(obs, block_until_ready=True)

    # Write only from process-0
    if dist.is_global_master():
        with open("out/obs.json", "w") as f:
            json.dump({"obs": str(obs_host)}, f)



Further Reading
-----------------

- JAX multi-process guide:
  https://docs.jax.dev/en/latest/multi_process.html

- NetKet distributed-parallelisation docs:
  https://netket.readthedocs.io/en/latest/parallelization.html

- Legacy neuraLQX MPI documentation:
  :doc:`mpi_deprecated/index`
