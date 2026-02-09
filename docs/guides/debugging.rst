.. _debugging_in_neuralqx:

=========================
Debugging in neuraLQX
=========================

neuraLQX ships with a built-in debugging and instrumentation layer designed for
scientific / HPC workloads (MPI, long runs, lots of nested calls). It is *opt-in*
and controlled entirely through environment variables.

What you get when it's enabled
=================================

When debugging is enabled, neuraLQX will:

- Create a **per-session log file** in the directory configured by::

    cfg.get_static("Leach Directory")

  The file name encodes timestamp, PID, MPI rank (if available), and a session id.

.. note::

    Typically, this default directory is called ``.neuralqx_logs``, it is a hidden directory and will site under your
    current directory (i.e. your ``os.getcwd()`` directory).

- Aggregate log messages **continuously** in that file (no need to "flush at the end").
- Keep a small **in-memory ring buffer** of recent log lines, used to dump context
  automatically on crashes.
- (Best-effort) Install hooks for:

  - **Unhandled exceptions** (with traceback + recent events)
  - **Python warnings** (captured into the log with call context)
  - **Faulthandler** for low-level crashes (e.g., segfaults, deadlocks), including
    signal-triggered stack dumps when supported.

Enabling / disabling debugging
=================================

Debugging is controlled by two environment variables:

- ``NQX_DEBUG``: ``True`` or ``False`` (turns the whole system on/off)
- ``NQX_LOG_LEVEL``: one of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``

Internally, neuraLQX reads these via your config object:

- ``cfg.get("DEBUG")`` maps to ``NQX_DEBUG``
- ``cfg.get("LOG_LEVEL")`` maps to ``NQX_LOG_LEVEL``

Typical command-line usage
----------------------------

.. code-block:: bash

   # minimal noise, still records key milestones + errors (recommended first step)
   export NQX_DEBUG=True
   export NQX_LOG_LEVEL=INFO

.. code-block:: bash

   # maximum detail (more expensive, most useful for performance + correctness issues)
   export NQX_DEBUG=True
   export NQX_LOG_LEVEL=DEBUG

.. code-block:: bash

   # essentially off
   export NQX_DEBUG=False

Initialisation
----------------

neuraLQX should initialise the debug system once at import time (package init).
If you're integrating manually, it looks like:

.. code-block:: python

   from neuralqx.debug import initialise
   initialise()  # creates log file + installs hooks if NQX_DEBUG=True

If debugging is disabled (``NQX_DEBUG=False``), initialisation is effectively a no-op
and no log file is created.

Log levels: what you get at each level
=========================================

The log level controls *how much* neuraLQX emits while debugging is enabled.

.. important::
    
    ``NQX_DEBUG`` is the master switch. If it's off, you get nothing
    regardless of ``NQX_LOG_LEVEL``.


INFO (recommended default)
----------------------------

Use this when you want a clean audit trail:

- Session header (environment, PID, rank, platform info)
- High-signal milestones (anything logged explicitly at INFO)
- For functions decorated with ``@trace``:

  - **Function entry**: which function was called, when, and from which caller site
  - A stable **call id** that lets you correlate nested calls

You get enough to answer: "What happened? In what order? Who called what?"

DEBUG (most detail)
----------------------

Use this when you need deep observability for correctness or performance:

Everything from INFO, plus:

- For ``@trace``-decorated functions:

  - **Function exit** with success marker and precise duration
  - Optional **return summaries** (bounded, safe summaries—no giant tensors dumped)
  - Optional slow-call detection (can emit WARNING when a call crosses a threshold)

- For ``@timeit`` and ``@io_trace``:

  - Timing lines (duration in ms)
  - Summarized input/output shapes and types (very useful for array-heavy kernels)

You get enough to answer: "Where is time going? Which calls are slow? What shapes flow through?"

WARNING
----------

Use this when you want to reduce noise but still capture issues:

- Python warnings (captured into the log)
- Slow-call warnings (when enabled by a decorator threshold)
- Anything explicitly logged as WARNING

You get enough to answer: "Are there suspicious conditions or performance cliffs?"

ERROR
--------

Use this when you want only failures:

- Exceptions logged by ``@errors_only`` and ``@trace`` (with traceback when available)
- Unhandled exceptions are logged as crash events
- Anything explicitly logged as ERROR

You get enough to answer: "What failed, where, and why?"

CRITICAL (crash-grade diagnostics)
-------------------------------------

Use this when you want maximum context around fatal issues:

- Everything that would be emitted at CRITICAL
- On crash-style events (unhandled exceptions), neuraLQX will also:

  - Dump a chunk of the **recent in-memory log buffer** (the last N events)
  - (Best-effort) include additional context (arguments summaries where configured)

This is intended for: "It dies on the cluster, and I need context without reproducing locally."

What exactly is logged by decorators
=======================================

neuraLQX provides lightweight decorators you can apply across the package.
Each decorator accepts a ``tag=...`` string, which prefixes each line like:

``[MY TAG] :: ...``

trace
-------

.. code-block:: python

   from neuralqx.debug import trace

   @trace(tag="SAMPLER")
   def sample(...):
       ...

Designed for "essential functions" where you need call provenance:

- INFO: entry (function + caller + call id)
- DEBUG: exit (success + timing + optional return summary)
- WARNING: slow-call warnings (if configured)
- ERROR/CRITICAL: exception with traceback + extra diagnostics

timeit
--------

.. code-block:: python

   from neuralqx.debug import timeit

   @timeit(tag="LANCZOS", warn_ms=250)
   def lanczos_step(...):
       ...

Designed for performance instrumentation with minimal overhead:

- DEBUG (or chosen level): duration line
- WARNING: if duration exceeds ``warn_ms``

errors_only
-------------

.. code-block:: python

   from neuralqx.debug import errors_only

   @errors_only(tag="IO")
   def load_checkpoint(...):
       ...

Designed for places where you only care about failures:

- ERROR: exception line (and traceback when verbosity allows)

io_trace
-----------

.. code-block:: python

   from neuralqx.debug import io_trace

   @io_trace(tag="KERNEL")
   def _get_conn_flattened_kernel(x, ...):
       ...

Designed for array-heavy internals:

- DEBUG: summarized inputs and outputs (shape/dtype, bounded repr)

Logging from your own code (beyond neuraLQX defaults)
========================================================

You can use the same debugging utilities in your scripts, notebooks, and custom code.

1) Use neuraLQX decorators in your own functions
---------------------------------------------------

If you wrote glue code around neuraLQX, decorate it too:

.. code-block:: python

   from neuralqx.debug import trace, timeit

   @trace(tag="MYPIPELINE")
   def run_pipeline(cfg, state):
       return do_the_thing(cfg, state)

   @timeit(tag="MYPIPELINE", warn_ms=500)
   def do_the_thing(cfg, state):
       ...

This is the easiest way to get end-to-end traces that include both your code and neuraLQX internals.

2) Emit explicit milestones with event() and log_once()
----------------------------------------------------------

.. code-block:: python

   from neuralqx.debug import event, log_once
   import logging

   event("starting run", tag="RUN", level=logging.INFO, seed=0, steps=1000)

   log_once(logging.WARNING, key="slow_fallback", tag="RUN",
            msg="Falling back to non-JIT path (this message is printed once).")

Use these for high-signal "breadcrumbs" that help you correlate phases of a run.

3) Capture deadlocks/hangs with dump_stacks() or signals
-----------------------------------------------------------

If a program appears stuck, you can dump stacks to the log:

.. code-block:: python

   from neuralqx.debug import dump_stacks
   dump_stacks(tag="HANG")

On many systems, neuraLQX also installs signal handlers (best-effort). If enabled,
you can often trigger a stack dump without attaching a debugger:

.. code-block:: bash

   # send SIGUSR1 to dump Python thread stacks (when supported)
   kill -USR1 <pid>

This is especially useful on clusters where interactive debugging is hard.

4) Use standard Python debugging tools alongside neuraLQX
------------------------------------------------------------

neuraLQX logs *observability*. For interactive debugging, you still want:

- ``pdb`` / ``breakpoint()`` for step-through debugging
- ``traceback`` / ``inspect`` to introspect failing states
- Profilers:
  - ``cProfile`` for function-level profiling
  - sampling profilers (e.g., ``py-spy``) for low overhead on long runs

Example using ``cProfile``:

.. code-block:: python

   import cProfile, pstats
   from neuralqx.main import run

   prof = cProfile.Profile()
   prof.enable()
   run()
   prof.disable()

   pstats.Stats(prof).sort_stats("tottime").print_stats(30)

5) For MPI / multi-process runs
----------------------------------

Debugging distributed programs is different. Practical advice:

- Reproduce with **1 rank** first (simplifies output and determinism).
- Each rank typically gets its **own log file**. When investigating:
  - Start with rank 0
  - Compare with the rank that reports the error
- If an error happens "only on cluster", crank up to ``DEBUG`` and add a few
  ``event(...)`` milestones around the suspected region.

Troubleshooting
==================

Nothing is being logged
-------------------------

Check:

- ``NQX_DEBUG=True`` (master switch)
- ``cfg.get_static("Leach Directory")`` points to a writable directory
- You are calling ``initialise()`` (usually done in package init)

Log file exists but it's empty
---------------------------------

- Confirm you actually hit decorated functions or emitted ``event(...)`` lines.
- Increase verbosity to ``DEBUG``.
- Ensure the run is not exiting before initialisation.

Logs are too noisy
---------------------

- Lower verbosity to ``INFO`` or ``WARNING``.
- Prefer ``@timeit`` over ``@trace`` for hot inner loops.
- Put argument/return logging behind CRITICAL (the defaults already do).

Logs contain sensitive data
------------------------------

neuraLQX intentionally avoids dumping full tensors or huge structures by default,
but logs can still include:

- file paths
- caller locations
- summarized argument/return representations

If you need stricter hygiene, avoid decorating functions that handle secrets and
keep your own ``event(...)`` messages free of sensitive content.

Suggested workflow
=====================

1. Enable debugging with::

      NQX_DEBUG=True
      NQX_LOG_LEVEL=INFO

2. Reproduce the issue and inspect the log file in the Leach Directory.

3. If you need more detail, upgrade to ``DEBUG`` and re-run.

4. If the problem is a hang or intermittent crash, use ``dump_stacks()`` and/or
   signal-triggered stack dumps.

5. If you still can't see enough, decorate **your own wrapper code** and add
   a few targeted ``event(...)`` milestones to bracket suspected regions.
