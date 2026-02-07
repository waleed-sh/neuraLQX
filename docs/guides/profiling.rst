.. _profiling_in_neuralqx:

Profiling in neuraLQX
=======================

neuraLQX ships with a lightweight, HPC-friendly profiling toolkit designed for
NetKet/JAX workflows. It is built to answer questions like:

- *What is my code spending time on?*
- *How does time per VMC step change from CPU -> 1 GPU -> 2 GPUs?*
- *Is scaling limited by sampling/expectation/gradient, optimizer, logging, or communication?*
- *How do I get a timeline view comparable to ``xprof`` / Perfetto / Nsight?*
- *How do I profile my own custom code without modifying neuraLQX internals?*

The profiler is safe to enable in serial runs, MPI runs (CPU or CUDA-aware MPI),
and JAX multi-host settings. It produces standard artifacts (text summary, JSON,
Perfetto/Chrome trace), and optionally NVTX ranges for Nsight tools and low-rate
telemetry.

.. warning::

   This feature is still actively under development. However, it is not in the experimental API as the majority of its
   aspects are fine (but there are a few issues that we are aware of which do not affect the functioning parts)...


Installation (profiling extras)
-------------------------------

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

Quick Start
-------------------------

Enable profiling and run your script:

.. code-block:: bash

   export NQX_PROFILE=1
   python run_vmc.py

Profiling outputs are written to:

- ``neuralqx.cfg.get_static("Profiling Directory")`` by default, which is typically::

    /.neuralqx_profiling/neuralqx_YYYYMMDD/

or can be overridden by ``NQX_PROFILE_DIR``.

The output directory contains per-rank files, for example:

.. code-block:: text

   run_<RUN_ID>/
     summary_rank000000.txt
     summary_rank000000.json
     trace_rank000000.json          (if enabled)
     metrics_rank000000.csv         (if enabled)
     README.txt

Configuration
-----------------

Profiling is controlled by neuraLQX configuration and/or environment variables.

Primary switch
~~~~~~~~~~~~~~~~~

``NQX_PROFILE`` (or ``neuralqx.cfg.get("PROFILE")``)
  Enable profiling globally.

- **Default:** ``0``
- **Set to:** ``1`` to enable.

.. note::

   ``PROFILE`` is runtime-immutable in neuraLQX. In scripts, you have to set
   ``NQX_PROFILE`` before importing/using neuraLQX. In Jupyter, you must also explicitly flush the profiler
   (see :ref:`profiling_in_jupyter`).

Output directory
~~~~~~~~~~~~~~~~~~~

``NQX_PROFILE_DIR``
  Override the directory where profiling artifacts are written.

- **Default:** ``neuralqx.cfg.get_static("Profiling Directory")``

Run directory naming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``NQX_PROFILE_RUN_ID``
  Force all ranks/processes to write into the same ``run_<RUN_ID>`` directory.

- **Default:** auto-generated from job id + timestamp.
- **Recommended:** set this explicitly for multi-rank runs so results are easy to compare.

Trace / timeline output
~~~~~~~~~~~~~~~~~~~~~~~~~~

``NQX_PROFILE_TRACE``
  Emit Chrome/Perfetto trace JSON (timeline view).

- **Default:** ``0``
- **When enabled:** writes ``trace_rankXXXXXX.json``

NVTX ranges (Nsight Systems/Compute)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   This feature is still undergoing testing and development and is currently a bit buggy.

``NQX_PROFILE_NVTX``
  Emit NVTX ranges around profiling regions.

- **Default:** ``0``
- **Use with:** ``nsys profile`` or Nsight Compute.

JAX trace annotations
~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   This feature is still undergoing testing and development and is currently a bit buggy.

``NQX_PROFILE_JAX_ANNOTATE``
  Insert JAX trace annotations (works with JAX profiler and can appear in traces).

- **Default:** ``1``

JAX profiler trace directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   This feature is still undergoing testing and development and is currently a bit buggy.

``NQX_PROFILE_JAX_TRACE``
  Start/stop ``jax.profiler`` trace and write JAX profiler output.

- **Default:** ``0``
- **Output:** a directory under the profiling run folder, per rank.

Low-rate telemetry
~~~~~~~~~~~~~~~~~~~~~

``NQX_PROFILE_METRICS``
  Enable a low-overhead background sampler that records CPU/memory and (if NVML
  available) GPU utilization/memory/power/temperature/clocks.

- **Default:** ``0``
- **Output:** ``metrics_rankXXXXXX.csv``

``NQX_PROFILE_SAMPLE_PERIOD_S``
  Telemetry sampling period in seconds.

- **Default:** ``1.0`` (1 Hz)

Synchronization (JAX async caveat)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``NQX_PROFILE_SYNC``
  Force device synchronization inside profiled function wrappers (via
  ``block_until_ready``) to make section durations closer to real device time.

- **Default:** ``0``

.. warning::

   JAX execution is asynchronous by default. Without synchronization, timings often
   represent *dispatch/host overhead*, not *true kernel/device time*. Enabling
   ``NQX_PROFILE_SYNC=1`` can significantly slow your run and should be used for
   targeted profiling only.

Trace buffer size
~~~~~~~~~~~~~~~~~~~~

``NQX_PROFILE_MAX_EVENTS``
  Maximum number of trace events retained in memory before export.

- **Default:** large enough for long runs, but bounded to avoid memory blowup.
- **Tune this** if you run extremely long jobs with trace enabled.

What neuraLQX Profiles by Default
------------------------------------

neuraLQX instruments the variational driver at well-defined boundaries. For the
VMC-style driver, the following regions are typically present:

- ``vmc:VMC``
  One region per optimization step (identified by ``step_count``). This is the
  core "iteration" unit.

  Subsections inside each step include:

  - ``vmc:forward_backward``
    Gradient + loss evaluation (usually dominates runtime)
  - ``vmc:update_parameters``
    Optimizer update and parameter application

- ``vmc:estimate``
  Observable estimation (when ``obs`` are provided and logged)
- ``vmc:callbacks``
  User callback overhead
- ``vmc:loggers``
  Logging overhead (including file I/O on the root rank)
- ``vmc:apply_gradient``
  JIT-ed optimizer update (timings depend on JAX async behavior unless sync enabled)

Within ``forward_backward``, typical nested regions include:

- ``vmc:state.reset``
- ``vmc:expect_and_grad``
- ``vmc:preconditioner``
- ``vmc:tree_cast``

These correspond to the instrumentation inserted in
``neuralqx.driver.abstract_variational_driver`` and VMC implementations.





Syncing
--------------------

JAX synchronization is the single most important "gotcha" when interpreting timings from
a Python-level profiler. JAX executes most work **asynchronously**: when you call a JIT-compiled function (or any
JAX transform that triggers device work), Python usually returns **before** the device has
finished running kernels. As a result:

- Section timings can measure mostly **host dispatch overhead** (enqueue time),
  not actual device/kernel time.
- The "real cost" may show up later, at the first point where Python **forces a
  synchronization** (e.g. converting a JAX array to a Python ``float``, printing a value,
  writing to disk, or explicitly calling ``block_until_ready``).
- In MPI/multi-device runs, collectives (reductions/allreduces) are also often part of
  the async stream, so *communication time can be misattributed* without sync.

``NQX_PROFILE_SYNC`` exists to optionally insert a **device synchronization barrier**
so that the measured duration of a region is closer to "true device time".

What syncing does (and what it does not do)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When syncing is enabled, neuraLQX uses ``block_until_ready`` on a value that depends on
the region's JAX work.

- If the value is a JAX array (or a pytree of arrays), ``block_until_ready`` waits until the
  device has finished all work needed to produce it.
- If the value is pure Python (``int``, ``float``, ``dict`` of Python objects), syncing is
  effectively a no-op.

.. important::
    
    Syncing is only meaningful for regions that produce/trigger JAX device work.
    It will not "speed up" or change pure Python timing, it only makes the reported durations
    more representative of device execution time.

No-op by default
~~~~~~~~~~~~~~~~~~~~~~~~

By default:

- ``NQX_PROFILE_SYNC=0`` (off)
- neuraLQX does **not** insert synchronization barriers.
- This means profiling has minimal perturbation, but timings may reflect host dispatch time
  rather than device time for JAX-heavy regions.

Syncing only happens if the user explicitly requests it:

.. code-block:: bash

   export NQX_PROFILE=1
   export NQX_PROFILE_SYNC=1
   python run_vmc.py

Where neuraLQX syncs by default (when enabled)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

neuraLQX does **not** sync every section. Instead, when ``NQX_PROFILE_SYNC=1`` is enabled,
it synchronizes at critical boundaries where users most often want accurate attribution.

This is the recommended "default sync set" because it captures the dominant compute and
communication costs without inserting barriers everywhere. When viewing the produced logs, synced sections (even if
syncing was *not* explicitly requested) are labelled by ``(sync)`` while un-synced sections are labelled by ``(dispatch)``.

Public VMC driver
^^^^^^^^^^^^^^^^^

When syncing is enabled, neuraLQX synchronizes at the end of:

- ``vmc:expect_and_grad``
  The dominant physics/learning workload: sampling + expectation + gradient.
  This is typically where GPU kernels and MPI reductions happen.

- ``vmc:preconditioner``
  SR/QGT/preconditioner work (often a solver and/or extra reductions).

- ``vmc:apply_gradient``
  Optimizer update and parameter application (often JIT-compiled).

- ``vmc:estimate`` (only if observables are estimated)
  Observable evaluation may be JAX-heavy and include reductions.

It does **not** sync by default in:
- ``vmc:callbacks`` (pure Python)
- ``vmc:loggers`` (I/O + Python)
- ``vmc:tree_cast`` and similar bookkeeping sections

Experimental MultiStateVMC driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the public VMC sync points, when syncing is enabled neuraLQX synchronizes at:

- ``vmc:fidelity_expect_and_grad_states_i_j``
  The pairwise overlap/fidelity estimator and its gradients, which can include device work
  and MPI collectives.

- Per-state energy/grad regions (if you instrument them separately), e.g.
  ``vmc:expect_and_grad_state_i``
  This gives an accurate per-state breakdown instead of only a total.

.. hint::

   Syncing is most useful at boundaries that return a small "token" (e.g. a scalar loss),
   because blocking on a huge gradient pytree can add overhead in Python traversal.
   In practice, syncing on one dependent scalar from the region is enough to ensure the
   whole region's device work has completed.

How to use syncing to profile custom code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you profile your own code with ``section``/``step`` or ``@profile``, you have two options:

1) Global syncing via ``NQX_PROFILE_SYNC=1`` (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Turn on syncing once, and any neuraLQX regions (and any profiled user functions that opt
into sync) will measure device time more accurately.

.. code-block:: bash

   export NQX_PROFILE=1
   export NQX_PROFILE_SYNC=1
   python my_script.py

This is best when you want a broad picture of where time is going (compute vs comms)
and you can tolerate the extra barriers.

2) Targeted syncing for specific regions (best for minimal perturbation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you want accurate timing only for a few hot spots, synchronize only those.

With the decorator
""""""""""""""""""""

If you wrap a function that returns JAX arrays, set ``sync=True``:

.. code-block:: python

   from neuralqx.profile import profile

   @profile(cat="user", sync=True)
   def my_jitted_kernel(x):
       return f(x)  # returns JAX arrays

With context managers
"""""""""""""""""""""""

If you are using ``section(...)`` around JAX code, synchronize on a value produced by
that region (typically a scalar or one array leaf):

.. code-block:: python

   import jax
   from neuralqx.profile import section
   from neuralqx.profile.profiler import get_profiler

   with section("user:compute", cat="user"):
       y = f(x)  # JAX work enqueued
       if get_profiler().config.sync:
           jax.block_until_ready(y)  # only blocks when sync enabled

Alternatively, you can auto-sync based on the environment variable using

.. code-block:: python

   import jax
   from neuralqx.profile import section
   from neuralqx.profile.profiler import get_profiler

   with section("user:compute", cat="user") as sec:
       y = sec.sync(f(x))  # only blocks when sync enabled


This pattern is ideal when you want to keep sync off globally, but still get correct
numbers for a particular region.

Practical guidance
~~~~~~~~~~~~~~~~~~~~~

- Use ``NQX_PROFILE_SYNC=0`` (default) for production throughput runs.
- Use ``NQX_PROFILE_SYNC=1`` when diagnosing:
  - "why is this section slow on 2 ranks?"
  - "is the slowdown compute or communication?"
  - "why does a small inner section look too fast?"
- For deep kernel-level profiling, pair neuraLQX labels with:
  - Perfetto trace output (``NQX_PROFILE_TRACE=1``)
  - NVTX ranges (``NQX_PROFILE_NVTX=1``) under Nsight Systems/Compute
  - JAX profiler traces (``NQX_PROFILE_JAX_TRACE=1``)

.. warning::

   Syncing introduces barriers and can reduce overlap between host, device, and communication.
   Always treat synchronized timings as "measurement mode", not "peak performance mode".

Understanding the Outputs
----------------------------

There are three main kinds of outputs:

1. Human-readable summary (``summary_rankXXXXXX.txt``)
2. Structured summary (``summary_rankXXXXXX.json``)
3. Timeline trace (``trace_rankXXXXXX.json``)

Text summary: ``summary_rank*.txt``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The text file is a tree report:

- **Total runtime** for the process

- For each section:

  - percentage of total time

  - section name (``cat:name``)

  - inclusive time

  - call count

  - (optionally) derived roofline metrics if FLOPs/bytes counters are provided

Example shape:

.. code-block:: text

   Total: 54.246 s
   ├── (98.4%) | vmc:VMC : 53.381 s | calls=150
   │   ├── (98.0%) | vmc:forward_backward : 52.303 s | calls=150
   │   │   ├── (91.7%) | vmc:expect_and_grad : 48.011 s | calls=150
   │   │   ├── ( 6.2%) | vmc:preconditioner : 3.262 s | calls=150
   │   │   └── ( 0.1%) | vmc:tree_cast : 0.049 s | calls=150
   │   └── ( 1.8%) | vmc:update_parameters : 0.951 s | calls=150
   └── ( 1.5%) | vmc:loggers : 0.816 s | calls=...

Key ideas:

- **Inclusive time** = time spent in that region *including* children.

- **Exclusive time** (available in JSON) = inclusive minus child time.

- If one node dominates (e.g. ``expect_and_grad``), scaling will largely depend
  on how that region scales.

JSON summary: ``summary_rank*.json``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The JSON summary contains the same tree plus metadata:

- Rank info (rank, size, hostname, etc.)

- Config flags used during the run

- Timing stats per node:

  - calls, inclusive/exclusive ns, min/max ns

  - (optional) counters: flops, bytes

This is the recommended format for automated comparison and regression tests.

Trace timeline: ``trace_rank*.json``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The trace file is compatible with Perfetto UI and ``chrome://tracing``.
It contains one event per profiled region, including nested regions.

Every ``vmc:VMC`` step event includes:

- duration

- ``args.step`` (the VMC step index)

This makes it easy to:

- exclude compilation/warm-up steps

- compute per-step distributions (median/p10/p90)

- compare serial vs GPU vs multi-GPU fairly

Viewing trace output
-----------------------

Perfetto (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Enable tracing:

.. code-block:: bash

   export NQX_PROFILE=1
   export NQX_PROFILE_TRACE=1
   python run_vmc.py

2. Open ``trace_rankXXXXXX.json`` in Perfetto UI.

Chrome tracing
~~~~~~~~~~~~~~~~~

1. Open ``chrome://tracing`` in Chromium/Chrome.
2. Load the ``trace_rankXXXXXX.json`` file.

Nsight Systems / Nsight Compute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable NVTX ranges and run under Nsight Systems:

.. code-block:: bash

   export NQX_PROFILE=1
   export NQX_PROFILE_NVTX=1
   nsys profile -o report python run_vmc.py

Your profiling regions will appear as NVTX ranges, allowing you to correlate
high-level sections (``expect_and_grad``, ``apply_gradient``) to GPU kernels,
communication, and hardware counters.

.. note::

   For roofline-style analysis (memory-bound vs compute-bound) you generally
   rely on Nsight Compute / CUPTI counters. neuraLQX provides the labels
   (NVTX + section names) to make that analysis straightforward.

How to Compare Serial vs 1 GPU vs 2 GPUs (Scaling Methodology)
-----------------------------------------------------------------

This section is critical: correct scaling analysis requires controlling for
warm-up (compilation) and for *total work per step*.

1) Decide strong vs weak scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Strong scaling:** total work per optimization step is held constant.
  Example: keep total MCMC samples per step fixed, divide samples across GPUs.
- **Weak scaling:** work per GPU is held constant (total work increases with GPUs).
  Example: keep samples per GPU fixed so total samples doubles when doubling GPUs.

If you do not control this, you can easily misinterpret results. Always record
the effective total sampling work (samples, chains, etc.) when comparing runs.

2) Exclude compilation / warm-up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JAX compilation often makes step 0 (or first few steps) much slower. For fair
comparison, compute steady-state step time using the trace and exclude at least
step 0, often the first 5–10 steps.

3) Use critical-path time for multi-rank runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In MPI / multi-process runs, the effective step time is determined by the
slowest rank when synchronization occurs. For 2 GPUs using 2 ranks:

- compute per-step times per rank
- take per-step maximum across ranks
- then compute median (excluding warm-up)

4) Report speedup and efficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let:

- ``T1`` = steady-state step time for baseline (serial or 1 GPU)
- ``Tp`` = steady-state step time for ``p`` GPUs/ranks

Then:

- **Speedup:** ``S(p) = T1 / Tp``
- **Efficiency:** ``E(p) = S(p) / p``

Compute this for:

- total step (``vmc:VMC``)
- dominant region (often ``vmc:expect_and_grad``)

This tells you whether scaling is limited by the physics kernel or by overhead.

5) Interpret which section limits scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare per-section times across runs:

- If ``expect_and_grad`` scales well but total doesn't -> overhead moved elsewhere.
- If ``expect_and_grad`` stops scaling -> likely communication, underutilization,
  or increased work (weak scaling).

Example workflow:

- Serial: ``VMC`` dominated by ``expect_and_grad`` (sampling/grad)
- 1 GPU: ``expect_and_grad`` drops, but logging/callbacks unchanged
- 2 GPUs: ``expect_and_grad`` drops less than expected -> communication/allreduce
  overhead or load imbalance.

Practical tip: make scaling plots from the JSON summaries (inclusive time per
region per run) and from the trace (median step time).

Self-Profiling Custom User Code
----------------------------------

Users can profile their own code without modifying neuraLQX internals.

There are two primary interfaces:

1) Context managers: ``section`` and ``step``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuralqx.profile import section, step

   for i in range(n_iters):
       with step(i, name="my_step", cat="user"):
           with section("prepare_batch", cat="user"):
               batch = make_batch()

           with section("compute", cat="user"):
               out = model(batch)

           with section("postprocess", cat="user"):
               results = analyze(out)

This will appear alongside neuraLQX internal regions in both the summary and trace.

2) Decorator: ``@profile``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuralqx.profile import profile

   @profile(cat="user")
   def expensive_python_fn(x):
       ...

   expensive_python_fn(data)

Synchronization for JAX-returning functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the function returns JAX arrays and you want accurate device time, you can:

- set ``NQX_PROFILE_SYNC=1`` globally, or
- use ``@profile(sync=True)`` for a specific function.

.. warning::

   Synchronization can slow down runs and perturb scaling. Use for targeted profiling.

.. _profiling_in_jupyter:

Profiling in Jupyter
------------------------

In notebooks, the Python process does not exit at the end of a cell, so the
automatic ``atexit`` flush may not run when you expect. Always flush explicitly:

.. code-block:: python

   import os
   os.environ["NQX_PROFILE]="1"
   import neuralqx as nqx

   # ... run your simulation ...

   from neuralqx.profile import flush
   flush()



Common Pitfalls and Best Practices
-------------------------------------

1) JAX async timing confusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Default timings are often *host dispatch time*.
- For accurate device timing: ``NQX_PROFILE_SYNC=1`` or use Nsight/JAX profiler.

2) Comparing runs with different total work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Always confirm whether you performed strong vs weak scaling. Record:

- number of chains / batch size
- samples per chain
- total samples per step
- observable set size (``obs``)
- whether additional diagnostics/logging are enabled

3) File I/O and logging
~~~~~~~~~~~~~~~~~~~~~~~~~~

Logging can dominate on CPU and become significant on GPU if logging is too frequent.
Use ``step_size``, ``write_every`` and ``save_params_every`` appropriately.

4) MPI aggregation
~~~~~~~~~~~~~~~~~~~~~

neuraLQX writes per-rank output to avoid contention. You can aggregate later:

- compare JSON summaries offline
- compute per-step critical path from traces (max across ranks)

Recommended Profiling Recipes
--------------------------------

Baseline summary only (minimal overhead)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export NQX_PROFILE=1
   python run_vmc.py

Perfetto trace for step-time distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export NQX_PROFILE=1
   export NQX_PROFILE_TRACE=1
   python run_vmc.py

Nsight Systems correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export NQX_PROFILE=1
   export NQX_PROFILE_NVTX=1
   nsys profile -o report python run_vmc.py

Device-synchronized section timings (targeted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export NQX_PROFILE=1
   export NQX_PROFILE_SYNC=1
   python run_vmc.py

Telemetry (CPU/mem + NVML GPU metrics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export NQX_PROFILE=1
   export NQX_PROFILE_METRICS=1
   export NQX_PROFILE_SAMPLE_PERIOD_S=1.0
   python run_vmc.py

Interpretation Checklist (CPU -> 1 GPU -> 2 GPUs)
-----------------------------------------------------

When you compare runs:

1. Use the **trace** to measure steady-state median step time for ``vmc:VMC``.
2. Exclude warm-up steps (JAX compilation).
3. For multi-rank runs, compute step time as **max across ranks** (critical path).
4. Compute speedup and efficiency.
5. Compare the dominant subtree (often ``vmc:expect_and_grad``).
6. If speedup stalls:

   - check whether work per step changed

   - check if communication or logging is now significant

   - use Nsight/JAX profiler to diagnose kernel utilization and bandwidth

Reference: Profiled VMC Driver Regions
-------------------------------------------

In the VMC driver implementation, the following profiling regions are inserted:

- Per-step:

  - ``vmc:VMC`` (one per optimization step)

- Inside step:

  - ``vmc:forward_backward``

    - ``vmc:state.reset``

    - ``vmc:expect_and_grad``

    - ``vmc:preconditioner``

    - ``vmc:tree_cast``

  - ``vmc:update_parameters``

    - ``vmc:apply_gradient``

- In ``run()``:

  - ``vmc:estimate``

  - ``vmc:log_additional_data``

  - ``vmc:callbacks``

  - ``vmc:loggers``

These labels are stable and are designed to be used as anchors for Perfetto, Nsight,
and post-processing scripts.

FAQ
-----

**Why do I only see files after the program ends?**
  Profiling artifacts are flushed at program exit to avoid perturbing the simulation.
  In Jupyter, call ``flush()`` manually.

**Why is step 0 much slower than the rest?**
  JAX compilation. Use trace output to exclude warm-up steps.

**Can I get roofline (memory-bound vs compute-bound) automatically?**
  neuraLQX provides region labels (and optional NVTX). For true roofline you typically
  run Nsight Compute and collect hardware counters for kernels inside ``expect_and_grad``.

**How do I compare multi-GPU?**
  Use per-rank traces and take the max across ranks per step (critical path), then compute
  median steady-state step time and speedup/efficiency.
