=================================
Parallelisation in neuraLQX
=================================

.. raw:: html

    <hr style="margin: 30px 0;">

**Last updated: October 21, 2025**

.. note::
    The following is true for NetKet versions up to 3.17.X, which is currently the highest version
    of NetKet that neuraLQX supports. **The latest NetKet releases do not support
    MPI based parallelisation.** Future releases of neuraLQX will also drop the MPI support in favour
    of Jax's sharding.

neuraLQX relies on NetKet for all computations. By default, NetKet only uses the ``jax`` default
device ``jax.local_devices()[0]`` to perform its computations. To parallelise, this implies that you
must resort to one of two parallelization strategies:

- through MPI (using ``mpi4py`` and ``mpi4jax``)
- through Jax sharding

NetKet is written such that code that both approaches are compatible with one another. You only need
to be careful when doing I/O operations such that you only do them on the master (rank 0) process.

However, in neuraLQX, a substantial amount of NetKet functionality is rewritten to be tailored for
our purposes. In doing so, **the Jax sharding approach is by default no longer possible.** This is,
however, **temporary** and is currently under development! **We expect to have this completed very
soon**. Therefore for the time being, neuraLQX is parallelisable using only MPI.

When using MPI, the parallelisation is done by offloading and
distributing the Markov chains and samples across the available compute power. This means, for example,
that you can run simulations with a larger number of samples and larger networks faster. We will nevertheless not limit
ourselves to only that, and may take parallelisation a step further to distribute "all that can be
distributed" across the available compute power. This is still a work in progress.


What can be done with MPI
---------------------------

The default way that neuraLQX runs, which is the default way NetKet runs, is 1 CPU/1 GPU. Using MPI,
neuraLQX now out-of-the-box is parallelisable with

- 1 node, MultiCPU
- distributed, MultiCPU
- 1 node, MultiGPU
- distributed, MultiGPU

MPI is fine-tuned for CPU based parallelisation, therefore we recommend it be used in that manner.
However, it is possible to use MPI for GPU computations although **it may be a tedious process**.
Once sharding is implemented, that will be the default for GPU based parallelisation and MPI will be
the default for CPU based parallelisation.

.. note::
    The GPU based parallelisation as currently available by neuraLQX is consistently around 10%
    slower than the Jax distributed GPU parallelisation. We are currently working on getting Jax's
    sharding based parallelisation up and working.

To get started with parallelisation in neuraLQX, please see the following documentations.

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: CPU-based MPI
      :link: cpu/index.html
      :text-align: center

   .. grid-item-card:: CUDA-aware MPI (GPU)
      :link: gpu/index.html
      :text-align: center

   .. grid-item-card:: Using neuraLQX in parallel
      :link: using_parallelisation.html
      :text-align: center

.. toctree::
   :hidden:
   :maxdepth: 1

   cpu/index
   gpu/index
   ./using_parallelisation
