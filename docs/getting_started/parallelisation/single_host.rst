.. _parallel_nqx_single_host:

===============================================
JAX Distributed Parallelisation (Single Host)
===============================================

This page describes single-host execution in neuraLQX: one node (laptop,
workstation, or server) with one or more local GPUs.

neuraLQX relies on JAX sharding/runtime behavior (same overall model used in
NetKet's latest stack).

If you are using older neuraLQX versions prior to 1.1.0 that still use MPI-era code,
refer to :doc:`mpi_deprecated/index`.


Scope
-------

Single-host means all compute is on one physical machine.

Two common cases:

- one GPU on one host
- multiple GPUs on one host

In both cases, JAX handles device placement/sharding on local devices.


Runtime Semantics
-------------------

On single host, you can run with a normal script launch:

.. code-block:: bash

    python run_simulation.py

For multi-GPU local runs, JAX can still shard across local devices even when
``jax.process_count() == 1``.

Unlike the previous MPI based parallelism, no configuration variables are required. Use JAX/runtime environment
variables only when you want explicit control. For example ``CUDA_VISIBLE_DEVICES=0,4`` will instruct JAX to use only
the GPUs number 0 and 4 out of all available GPUs.


Automatic Initialisation In neuraLQX
--------------------------------------

neuraLQX initializes JAX distributed internally when distributed launcher
signals are detected. On standard single-host Python launches, there is
typically nothing to initialize manually.

Do not add manual ``jax.distributed.initialize()`` unless you have a very
specific launcher/runtime requirement.

Import-order rule still applies: import neuraLQX before importing JAX/NetKet. neuraLQX will automatically instruct
JAX to use available devices, with the default choice being GPUs. If no GPUs are detected, JAX will fallback to the
next available, which is typically CPUs. This is why we recommend calling using

.. code-block:: python

    from neuralqx.utils import distributed as dist

    dist.check_distributed()

At the start of your script to make sure that JAX is correctly utilising the resources.


Minimal Script
----------------

.. code-block:: python

    import neuralqx as nqx
    from neuralqx.utils import distributed as dist

    # Optional: rank/device diagnostics
    dist.check_distributed()

    # Build and run your simulation as usual.
    # graph = ...
    # hilbert = ...
    # lqx = ...
    # solver = nqx.solver.Solver(lqx, output_path="out")
    # solver.set_sampler(...)
    # solver.set_optimizer(...)
    # solver.set_network(...)
    # solver.run(n_iters=200)


Diagnostics
-------------

Use this startup check:

.. code-block:: python

    import jax

    print(
        f"process={jax.process_index()}/{jax.process_count()} "
        f"local_devices={jax.local_devices()}",
        flush=True,
    )

Interpretation on single host:

- ``jax.local_devices()`` shows what this host can use.
- ``jax.process_count() == 1`` is normal for many single-host runs.
- more than one local GPU still enables local sharding workflows.


Custom I/O Rules
------------------

In the previous MPI case, different processes are launched for different MPI ranks, which meant that
before doing any I/O, you had to do that only from the 0-th rank. In single-host JAX, you only need to
make sure that the sharded data is collected from all devices before doing I/O. in neuraLQX, you can do this
by using the ``safe_replicate_for_io`` from the distributed utils. This will ensure that the data is
first on the device using ``jax.device_get(...)`` and also is available on all processes (irrelevant for
single-host runs).


Further Reading
-----------------

- JAX multi-process and sharding concepts:
  https://docs.jax.dev/en/latest/multi_process.html

- NetKet distributed-parallelisation guides:
  https://netket.readthedocs.io/en/latest/parallelization.html

- Legacy neuraLQX MPI documentation:
  :doc:`mpi_deprecated/index`
