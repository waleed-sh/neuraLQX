# Copyright (c) 2026 The neuraLQX Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Concrete solver implementation for neuraLQX Variational Monte Carlo workflows.

This module provides a production-ready :class:`Solver` that implements the abstract solver API
defined by :class:`~neuralqx.solvers._abstract_solver.AbstractSolver`. It orchestrates the full
optimisation lifecycle around an LQX model:

- configuration of sampler, optimiser (and SR preconditioner), and neural network,
- construction of a Monte-Carlo variational state (:class:`~neuralqx.vqs.MCState`),
- construction and execution of a VMC driver (:class:`~neuralqx.driver.VMC`),
- robust finalisation (checkpoint export, logging, plotting, and live monitoring shutdown),
- safe continuation from a previous checkpoint (including MPI-aware state broadcast).

Run structure
-------------
The main optimisation entry point (:meth:`Solver.run`) is intentionally split into three phases:

1) Preparation (:meth:`Solver._prepare_run`)
   - Ensures driver/state initialisation for fresh runs
   - Normalises callbacks and attaches optional live monitoring
   - Records observables to be evaluated during optimisation

2) Execution (:meth:`Solver._execute_run`)
   - Delegates to the underlying VMC driver loop
   - Tracks performed iteration count when available

3) Finalisation (:meth:`Solver._finalise_run`)
   - Mirrors driver state back into solver state
   - Computes robust final observable/constraint estimates
   - Logs trailing averages and exports checkpoints
   - Plots and serialises results (rank-0) and synchronises ranks

MPI and interrupts
------------------
This implementation treats Ctrl+C differently depending on the MPI context:

- Single-rank mode: Ctrl+C triggers a graceful abort that still finalises, exports an "aborted"
  marker checkpoint, and attempts to plot/log safely.
- Multi-rank mode: Ctrl+C triggers a hard MPI abort to avoid deadlocks from rank-skewed interrupts
  during collectives. A lightweight abort marker file is written by rank-0 when possible.

Serialisation
-------------
Export/import uses an MCState-level serialisation format and ensures that all MPI ranks synchronise
around serialisation steps. This prevents situations where JAX device computations (including
MPI-backed collectives) would otherwise block serialisation on some ranks.
"""

import logging
import os.path
from dataclasses import dataclass
from datetime import datetime
import warnings

from collections.abc import Callable
from collections.abc import Sequence

from typing import Any
from typing import Union
from typing import TYPE_CHECKING

import humanize

import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from netket.optimizer.preconditioner import AbstractLinearPreconditioner

from neuralqx.callbacks import LiveMonitoringCallback
from neuralqx.driver import VMC
from neuralqx.nn.projectors.group_projector import wrap_model
from neuralqx.optimizer import Optimizer
from neuralqx.optimizer.solvers import Solvers
from neuralqx.utils.serialization import load_from_file
from neuralqx.utils.serialization import save_to_file
from neuralqx.utils.parsing import log_module_attributes
from neuralqx.utils.parsing import required_kwargs
from neuralqx.vqs import MCState
from neuralqx.utils import mpi as _mpi
from neuralqx.vqs.mc.mc_state.state import serialize_MCState
from neuralqx.vqs.mc.mc_state.state import deserialize_MCState
from neuralqx.samplers import Sampler as SamplerNQX

from ._abstract_solver import AbstractSolver

from neuralqx import cfg
from neuralqx.debug import errors_only, event
from neuralqx.profile import flush
from neuralqx.utils.errors import LiveMonitoringUnavailableWarning

import jax.numpy as jnp

from netket.logging import RuntimeLog
from netket.optimizer import SR, identity_preconditioner
from netket.utils import is_probably_holomorphic
from netket.utils.mpi import mpi_bcast

if TYPE_CHECKING:
    # these imports are only for type checking/docs
    # concrete implementations may use Flax/NetKet types or neuraLQX wrappers
    from flax.linen import Module as FlaxModule
    from netket.operator import AbstractOperator
    from netket.experimental.observable import AbstractObservable
    from netket.stats.mc_stats import Stats
    from netket.driver import AbstractVariationalDriver


def reject_outliers(data, m: float = 2.0):
    """
    Reject statistical outliers from a 1D data array using a robust median absolute deviation filter.

    This helper is primarily used for plotting (e.g. determining inset y-limits) where a handful of
    extreme values can make the view uninformative. The method computes:

    - the median of the data,
    - the median absolute deviation (MAD),
    - a scaled deviation score for each entry,

    and returns only those values whose scaled deviation is below a threshold ``m``.

    Notes
    -----
    - This filter is robust to heavy-tailed distributions and is less sensitive than mean/std-based
      rules.
    - If MAD is zero (all values identical), the function falls back to a denominator of 1.0 to avoid
      division by zero and will keep all points.

    :param data: 1D array-like of numeric values to be filtered.
    :param m: Outlier rejection threshold in units of MAD. Larger values keep more points.
    :return: A NumPy array containing only the inlier values.
    :raises ValueError: If ``data`` cannot be converted to a numeric NumPy array.
    """

    data = np.asarray(data)

    # remove NaN/inf first
    data = data[np.isfinite(data)]
    if data.size == 0:
        return data

    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.0)

    kept = data[s < m]

    # if the filter nukes everything (can happen with nasty data), fall back gracefully
    return kept if kept.size else data


@dataclass
class _RunContext:
    """
    Internal run context used to pass configuration and mutable run-state between run phases.

    This dataclass is a lightweight "transport object" that makes the run pipeline explicit and keeps
    phase-specific logic decoupled:

    - Preparation populates immutable configuration (requested iters, flags, callbacks, observables)
      and initialises optional components (live monitoring callback).
    - Execution updates runtime fields such as performed iteration count.
    - Finalisation consults runtime state to decide what to export/plot/log and how to label outputs.

    Fields
    ------
    Core configuration (provided by :meth:`Solver._prepare_run`):

    - ``n_iters``: Requested number of optimisation iterations to execute.

    - ``silent_print``: If True, suppress user-facing printing in finalisation.

    - ``silent_plot``: If True, suppress interactive plot display (figures may still be saved).

    - ``timer``: If True, enable timing/profiling support in the driver.

    - ``live_monitoring``: If True, attach a live-monitoring callback.

    - ``callbacks``: list of callback callables invoked by the driver each iteration.

    - ``observables``: Optional mapping of observable names to operators/observables to evaluate.

    - ``is_continuation``: True for continuation runs (do not reinitialise core state).

    Runtime-populated fields (filled during execution/finalisation):

    - ``live_callback``: Live monitoring callback instance (if enabled).

    - ``aborted``: True if the run ended early (KeyboardInterrupt or error).

    - ``abort_reason``: Human-readable reason for early termination.

    - ``performed_iters``: Best-effort count of iterations actually completed.

    - ``marker``: Checkpoint/export label (e.g. "FinalState" or "AbortedState").

    :param n_iters: Requested number of optimisation iterations to run.
    :param silent_print: Whether to suppress final printed summaries.
    :param silent_plot: Whether to suppress interactive plot display.
    :param timer: Whether to enable timing/profiling in the driver.
    :param live_monitoring: Whether to enable live monitoring callbacks.
    :param callbacks: list of per-iteration callbacks.
    :param observables: Optional observable registry evaluated during the run.
    :param is_continuation: Whether this run is a continuation of a previous run.
    :return: None.
    :raises TypeError: If field values are of incompatible types (e.g. callbacks not callable).
    """

    n_iters: int
    silent_print: bool
    silent_plot: bool
    timer: bool
    live_monitoring: bool
    callbacks: list[Callable]
    observables: dict | None
    is_continuation: bool

    # runtime-populated
    live_callback: Any | None = None
    aborted: bool = False
    abort_reason: str | None = None
    performed_iters: int = 0
    marker: str = "FinalState"


class Solver(AbstractSolver):
    """
    Concrete neuraLQX solver implementing a VMC-based optimisation workflow.

    This class is the main user-facing solver implementation that executes variational optimisation for
    an LQX model. It inherits core wiring, output/log bookkeeping, and safety flags from
    :class:`~neuralqx.solvers._abstract_solver.AbstractSolver` and implements the operational API:

    - :meth:`set_sampler` builds and attaches a NetKet-compatible sampler (including specialised gauge
      samplers when supported by the Hilbert space).
    - :meth:`set_optimizer` builds an Optax optimiser and configures stochastic reconfiguration (SR),
      including the SR linear solver backend.
    - :meth:`set_network` installs a neural network Ansatz and optionally wraps it with symmetry/group
      projection when running diffeomorphism-invariant simulations.
    - :meth:`initialize_vmc` constructs the :class:`~neuralqx.vqs.MCState` and the :class:`~neuralqx.driver.VMC`
      driver, including SR preconditioning.
    - :meth:`run` executes an optimisation loop with robust interruption handling, finalisation, state
      export, and plotting.
    - :meth:`continue_simulation` resumes from an in-memory or on-disk checkpoint while preserving
      runtime logs and ensuring driver/state consistency.

    Key design goals
    ----------------
    Robustness and reproducibility:

    - Every solver run has a unique hash and a deterministic seed (unless user overrides).

    - State export/import is MPI-aware and uses broadcast to ensure all ranks reconstruct the same
      state.

    - Finalisation is defensive: I/O and plotting failures do not crash the solver after a successful
      run; they are treated as best-effort.

    Clarity of run semantics:

    - The run pipeline is explicitly split into preparation, execution, and finalisation phases.

    - Continuations do not implicitly reset logs or state unless explicitly requested.

    MPI-safe interruption:

    - Single-rank runs attempt graceful abort and still export a final checkpoint.

    - Multi-rank runs hard-abort on Ctrl+C to avoid deadlocks from rank-skewed collectives.

    :param lqx: The model interface implementing :class:`~neuralqx.lqx.AbstractLqxInterface`.
    :param output_path: Optional base directory for solver outputs.
    :param auxiliary_path: Optional grouping component appended under ``output_path``.
    :param clean_up: If True, remove empty stale directories before starting.
    :param seed: Optional explicit seed controlling solver reproducibility.
    :return: None.
    :raises ValueError: If ``output_path`` is provided but invalid.
    :raises OSError: If output directories cannot be created.
    """

    def run(
        self,
        n_iters: int,
        *,
        silent_print: bool = False,
        silent_plot: bool = True,
        timer: bool = False,
        live_monitoring: bool = False,
        callbacks: Callable[
            [int, dict, "AbstractVariationalDriver"], bool
        ] = lambda *x: True,
        observables: dict[str, "AbstractOperator"] | None = None,
        **kwargs,
    ) -> None:
        """
        Run an optimisation/minimisation simulation.

        This method is the primary optimisation entry point. The implementation is structured into three
        phases to keep the execution path explicit and robust:

        1) Preparation (:meth:`_prepare_run`)
           - For fresh runs, ensures VMC driver/state are initialised.
           - Normalises callbacks into a list.
           - Optionally attaches a live monitoring callback.
           - Stores the observable registry and logs basic run metadata.

        2) Execution (:meth:`_execute_run`)
           - Calls the underlying VMC driver loop with consistent parameters.
           - Tracks performed iteration count when available.

        3) Finalisation (:meth:`_finalise_run`)
           - Mirrors driver state back into solver state.
           - Computes final expectation/constraint estimates.
           - Logs trailing statistics, exports a checkpoint, and plots results.

        Interrupt handling
        ------------------

        - KeyboardInterrupt (Ctrl+C):

          * Single-rank: marks the run as aborted, writes an abort marker, and still finalises safely.

          * Multi-rank: writes a lightweight abort marker from rank-0 when possible and then hard-aborts
            the MPI job to prevent deadlocks.

        - Other exceptions:

          * Marks run as aborted and writes a lightweight abort marker on rank-0 when possible.

          * Re-raises the exception to keep the error visible.

        :param n_iters: Number of optimisation iterations to execute.
        :param silent_print: If True, suppress user-facing printing during finalisation.
        :param silent_plot: If True, suppress interactive plot display (figures may still be saved).
        :param timer: If True, enable timing/profiling integration in the driver loop.
        :param live_monitoring: If True, attach a live monitoring callback for real-time feedback.
        :param callbacks: Callback or list of callbacks invoked after each iteration. Each callback must
            accept ``(iteration_index, log_dict, driver)`` and return a boolean. Returning False requests
            early termination.
        :param observables: Optional mapping of observable names to operators/observables to evaluate during
            optimisation.
        :param kwargs: Implementation-specific run options. Internal continuation mode may set ``_ct=True``.
        :return: None.
        :raises RuntimeError: If required solver components are not initialised and cannot be initialised.
        :raises Exception: Propagates any unexpected exception raised by the driver loop or callbacks.
        """

        event(
            "STARTING_SIMULATION",
            tag="SOLVER:RUN",
            level=logging.INFO,
            n_iters=n_iters,
            live_monitor=live_monitoring,
        )

        # delegate run preparation to the helper
        # this returns a RunContext which can be used to actually start the simulations
        ctx = self._prepare_run(
            n_iters=n_iters,
            silent_print=silent_print,
            silent_plot=silent_plot,
            timer=timer,
            live_monitoring=live_monitoring,
            callbacks=callbacks,
            observables=observables,
            is_continuation=bool(kwargs.get("_ct", False)),
        )

        try:
            self._execute_run(ctx)
        except KeyboardInterrupt as exc:
            # decide graceful vs hard abort strategy
            self._handle_keyboard_interrupt(ctx, exc)

            # if we reach here in single-rank mode, we continue to finalise
            self._finalise_run(ctx)
            return
        except Exception as exc:
            # handle non keyboard interrupt exits
            ctx.aborted = True
            ctx.abort_reason = f"{type(exc).__name__}: {exc}"

            self.printer.print(
                f"Simulation failed before completing iterations: {type(exc).__name__}"
            )

            # we also do NOT export/plot/log here
            # just a minimal trace:
            if _mpi.is_global_master():
                try:
                    self._write_abort_marker(ctx)
                except Exception:
                    pass

            # important: keep the error visible
            raise

        else:
            self._finalise_run(ctx)

            # flush profiling to disk
            flush()

    def _prepare_run(
        self,
        *,
        n_iters: int,
        silent_print: bool,
        silent_plot: bool,
        timer: bool,
        live_monitoring: bool,
        callbacks: Callable | list[Callable],
        observables: dict | None,
        is_continuation: bool,
    ) -> _RunContext:
        """
        Prepare internal state for an optimisation run and return a populated run context.

        This helper collects all run configuration into a single :class:`_RunContext` and performs the
        lightweight setup needed before starting the driver loop.

        Responsibilities
        ----------------

        - Ensures the VMC driver and variational state are initialised for fresh runs.

        - Normalises the callback argument into a list.

        - Attaches a :class:`~neuralqx.callbacks.LiveMonitoringCallback` when ``live_monitoring=True``.

        - Stores the observable registry for later reuse (e.g. continuation runs).

        - Sets the solver iteration counter and logs basic metadata via the solver logger.

        :param n_iters: Number of optimisation iterations to execute.
        :param silent_print: Whether to suppress final printed summaries.
        :param silent_plot: Whether to suppress interactive plot display.
        :param timer: Whether to enable driver timing/profiling.
        :param live_monitoring: Whether to attach a live monitoring callback.
        :param callbacks: Callback or list of callbacks, normalised to a list.
        :param observables: Optional mapping of observable names to operators/observables.
        :param is_continuation: True if this is a continuation run (do not reset core state).
        :return: A populated :class:`_RunContext` describing the run.
        :raises RuntimeError: If driver/state initialisation fails for a fresh run.
        :raises TypeError: If callbacks are not callable.
        """

        event(
            "PREPARING_RUN",
            tag="SOLVER:RUN",
            level=logging.INFO,
            n_iters=n_iters,
            live_monitor=live_monitoring,
            is_continuation=is_continuation,
        )

        # ensure VMC is initialised for fresh runs
        if not is_continuation:
            # this method validates flags internally
            self.initialize_vmc()

            # fresh log for a fresh run
            # self._nk_log = RuntimeLog()

        # normalise callbacks to list
        cb_list = self._normalise_callbacks(callbacks)

        # prepare live monitoring callback
        live_cb = None
        if live_monitoring:
            if cfg.get("MPI_CUDA") or cfg.get("MPI") or cfg.get("JAX_DISTRIBUTED"):
                LiveMonitoringUnavailableWarning()
            else:
                live_cb = LiveMonitoringCallback()
                cb_list = [live_cb] + cb_list

        # store run configuration
        self._registered_observables = observables

        # set current run counters
        self.n_iters = int(n_iters)

        # log metadata if logger exists
        if hasattr(self, "_logger") and self._logger is not None:
            self._logger.log(
                ["Number of iterations", "Diagonal shift"],
                [self.n_iters, getattr(self, "_diagonal_shift", None)],
            )

        # user-facing start message
        if self.printer is not None:
            obs_msg = (
                f"\nObservables: {list(observables.keys())}" if observables else ""
            )
            self.printer.print(f"Starting simulation{obs_msg}")

        # return the populated RunContext
        return _RunContext(
            n_iters=self.n_iters,
            silent_print=silent_print,
            silent_plot=silent_plot,
            timer=timer,
            live_monitoring=live_monitoring,
            callbacks=cb_list,
            observables=observables,
            is_continuation=is_continuation,
            live_callback=live_cb,
        )

    @staticmethod
    def _normalise_callbacks(
        callbacks: Callable | list[Callable],
    ) -> list[Callable]:
        """
        Normalise the callback specification into a list of callables.

        This helper accepts:
        - ``None`` -> returns an empty list,
        - a list of callables -> returned unchanged,
        - a single callable -> wrapped into a singleton list.

        :param callbacks: A callable, list of callables, or None.
        :return: A list of callback callables (possibly empty).
        :raises TypeError: If ``callbacks`` is not None, not a callable, and not a list of callables.
        """

        event(
            "NORMALSING_CALLBACKS",
            tag="SOLVER:RUN",
            level=logging.INFO,
        )

        if callbacks is None:
            return []
        if isinstance(callbacks, list):
            return callbacks
        return [callbacks]

    def _execute_run(self, ctx: _RunContext) -> None:
        """
        Execute the optimisation loop using the underlying VMC driver.

        This method is responsible only for running the driver with consistent parameters and recording a
        best-effort performed-iteration count for finalisation.

        If the driver is not initialised (e.g. continuation run with missing state), this method attempts to
        initialise it by calling :meth:`initialize_vmc`.

        :param ctx: Run context produced by :meth:`_prepare_run`.
        :return: None.
        :raises RuntimeError: If driver initialisation fails or if the driver run fails unexpectedly.
        :raises Exception: Propagates exceptions raised by the driver loop or callbacks.
        """

        event(
            "EXECUTING_RUN",
            tag="SOLVER:RUN",
            level=logging.INFO,
        )

        # guard: ensure we have a driver
        if not getattr(self, "_driver_flag", False):
            self.initialize_vmc()

        # run the simulation
        self.vmc_driver.run(
            n_iter=ctx.n_iters,
            out=self.log,
            callback=ctx.callbacks,
            obs=ctx.observables,
            timeit=ctx.timer,
        )

        # best-effort performed iteration count
        ctx.performed_iters = int(getattr(self.vmc_driver, "step_count", ctx.n_iters))

    @errors_only(tag="SOLVER:GRACEFUL_ABORT")
    def _handle_keyboard_interrupt(
        self, ctx: _RunContext, exc: KeyboardInterrupt
    ) -> None:
        """
        Handle Ctrl+C (KeyboardInterrupt) during the optimisation run.

        This helper implements a split strategy depending on MPI context:

        - Single-rank: mark the run as aborted and allow finalisation to proceed (export, plotting, etc.).
        - Multi-rank: attempt a lightweight abort marker write from rank-0 and then hard-abort the MPI job
          to avoid deadlocks from rank-skewed interrupts during collectives.

        The method updates the run context with:
        - ``aborted=True``
        - ``abort_reason="KeyboardInterrupt"``
        - ``marker="AbortedState"``
        - ``performed_iters`` as a best-effort driver step count

        :param ctx: Active run context to update.
        :param exc: The KeyboardInterrupt exception instance.
        :return: None.
        :raises SystemExit: In multi-rank mode, this method may terminate the process after issuing an MPI
            abort.
        """

        ctx.aborted = True
        ctx.abort_reason = "KeyboardInterrupt"
        ctx.marker = "AbortedState"

        # best-effort step count before any hard abort
        ctx.performed_iters = int(getattr(self.vmc_driver, "step_count", 0))

        # Multi-rank safety: hard abort
        if getattr(_mpi, "available", False) and getattr(_mpi, "n_nodes", 1) > 1:

            # try to write a lightweight marker from rank-0 without barriers
            if _mpi.is_global_master():
                self._write_abort_marker(ctx)

            # hard abort all ranks
            self._abort_mpi_job(exit_code=130, reason="User KeyboardInterrupt")

        # single-rank path: allow graceful finalisation
        if self.printer is not None:
            self.printer.print(
                f"Simulation interrupted by user at approx. step {ctx.performed_iters}."
            )

    def _write_abort_marker(self, ctx: _RunContext) -> None:
        """
        Write a lightweight abort marker file into the output directory.

        This marker is intentionally minimal:
        - it avoids JAX synchronisation,
        - it avoids state serialisation,
        - it avoids MPI barriers inside the function body.

        It is safe to call in situations where the program is already in a partially-failing state.

        :param ctx: Active run context providing performed iteration count and abort reason.
        :return: None.
        :raises OSError: If the marker file cannot be created due to filesystem errors (best-effort; callers
            typically suppress failures).
        """

        try:
            os.makedirs(self.output_path, exist_ok=True)
            marker_path = os.path.join(self.output_path, f"{self.hash}_ABORTED.txt")
            with open(marker_path, "w", encoding="utf-8") as f:
                f.write(
                    "neuraLQX solver aborted by user.\n"
                    f"hash: {self.hash}\n"
                    f"performed_iters (approx): {ctx.performed_iters}\n"
                )
        except Exception:
            # marker failure should never block abort.
            pass

    @staticmethod
    def _abort_mpi_job(exit_code: int = 1, reason: str = "") -> None:
        """
        Hard-abort an MPI job to avoid deadlocks after rank-skewed interrupts.

        In multi-rank contexts, a KeyboardInterrupt may occur on some ranks but not others. Continuing
        execution would risk hanging inside MPI collectives (including those indirectly triggered by JAX).
        Hard-aborting the communicator is the most reliable safety mechanism.

        If MPI is not available or the job is effectively single-rank, this method falls back to a normal
        process exit.

        :param exit_code: Process exit code (commonly 130 for Ctrl+C).
        :param reason: Optional human-readable reason for the abort (informational only).
        :return: None.
        :raises SystemExit: Always terminates the current process (and typically the full MPI job).
        """

        # if we're not actually in multi-rank mode, just exit
        if not getattr(_mpi, "available", False) or getattr(_mpi, "n_nodes", 1) <= 1:
            exit(exit_code)

        try:
            # preferred: abort the active communicator
            if hasattr(_mpi, "comm") and hasattr(_mpi.comm, "Abort"):
                _mpi.comm.Abort(exit_code)
                return

            # fallback: abort world explicitly if exposed
            if hasattr(_mpi, "MPI") and hasattr(_mpi.MPI, "COMM_WORLD"):
                _mpi.MPI.COMM_WORLD.Abort(exit_code)
                return

        finally:
            # if MPI abort didn't immediately terminate, force kill this process
            exit(exit_code)

    def _finalise_run(self, ctx: _RunContext) -> None:
        """
        Finalise a run by mirroring driver state, computing final statistics, exporting state, and plotting.

        This method is the final phase of the run pipeline. It is designed to be robust in single-rank
        contexts, including after a graceful KeyboardInterrupt abort. In multi-rank KeyboardInterrupt cases
        it is typically not reached because the policy is to hard-abort the MPI job.

        Responsibilities
        ----------------

        - Mirror the driver-owned variational state into the solver and set state flags.

        - Determine the performed iteration count and update solver-visible ``n_iters``.

        - Maintain a cumulative ``_max_iters`` counter across resumed runs for plotting.

        - Compute a final constraint estimate and log it.

        - Compute trailing window averages (mean and R-hat) and log them.

        - Export a checkpoint (best-effort; failures are suppressed).

        - Export and optionally display HTML logs (rank-0).

        - Plot and save the minimisation curve (rank-0).

        - Shutdown live monitoring callback cleanly.

        - Update ``_solved`` status and synchronise ranks on exit.

        :param ctx: Run context carrying run configuration and runtime state (aborted/performed iters).
        :return: None.
        :raises RuntimeError: If required driver state is unavailable and finalisation cannot proceed.
        """

        event(
            "FINALISING_SIMULATION",
            tag="SOLVER:RUN",
            level=logging.INFO,
        )

        # mirror state from driver
        try:
            self._variational_state = self.vmc_driver.state
            self._vstate_flag = True
        except Exception:
            # if driver state is unavailable, we cannot proceed meaningfully
            return

        # determine performed iterations if not already known
        if ctx.performed_iters <= 0:
            ctx.performed_iters = int(
                getattr(self.vmc_driver, "step_count", ctx.n_iters)
            )

        # adjust solver's visible n_iters to reality
        self.n_iters = ctx.performed_iters

        # update max iterations aggregator
        if self._max_iters is None:
            self._max_iters = self.n_iters
        else:
            self._max_iters += self.n_iters

        # compute and log final constraint expectation safely
        self._final_constraint_estimate()

        # compute results averages and log
        self._log_trailing_averages(window=100)

        # Export a checkpoint in safe situations
        # - Always safe in single-rank.
        # - In multi-rank, this finaliser is typically not reached after Ctrl+C.
        try:
            self.export_state(marker=ctx.marker, silent=ctx.silent_print)
        except Exception:
            # avoid failing finalisation due to I/O
            pass

        # optional user-facing log display / plot
        if _mpi.is_global_master():
            if hasattr(self, "_logger") and self._logger is not None:
                # always export
                try:
                    self._logger.write_log_to_file(self.output_path)
                except Exception as exc:
                    self.printer.print(f"Failed to export HTML log: {exc}")

                # only display if not silent
                if not ctx.silent_print:
                    try:
                        self._logger.display_log()
                    except Exception as exc:
                        self.printer.print(f"Failed to display log: {exc}")

            try:
                self.plot_results(silent_plot=ctx.silent_plot)
            except Exception as exc:
                self.printer.print(f"Failed to plot/export results: {exc}")

        else:
            _mpi.barrier()

        # shut down live monitoring cleanly
        if ctx.live_callback is not None:
            try:
                ctx.live_callback.shutdown()
            except Exception:
                pass

        self._solved = not ctx.aborted and (ctx.performed_iters >= ctx.n_iters)

        # final barrier for orderly exit in normal cases
        _mpi.barrier()

    def _final_constraint_estimate(self) -> None:
        """
        Compute and store a robust final estimate of the model constraint expectation value.

        This helper evaluates the model constraint on the final variational state and stores:
        - ``self.final_constraint_stats``: the full Stats object returned by the backend,
        - ``self.final_constraint_mean``: a scalar Python value derived from the mean, with a robust
          handling of small imaginary components (numerical noise).

        If logging is available, it records both the full Stats object and the scalar mean value. If the
        model provides an exact diagonalisation ground energy, it is also logged when available.

        :return: None.
        :raises Exception: Any backend exception during expectation evaluation is suppressed by this helper
            (best-effort finalisation).
        """

        try:
            stats = self.variational_state.expect(self.lqx.constraint)
        except Exception:
            return

        self.final_constraint_stats = stats
        self.final_constraint_mean = self._coerce_mean_value(
            stats.mean,
            sigma=getattr(stats, "Sigma", None),
        )

        # log the full Stats object when possible
        if hasattr(self, "_logger") and self._logger is not None:
            try:
                self._logger.log("Network result", stats)
                self._logger.log(
                    "Network result (mean scalar)", self.final_constraint_mean
                )
            except Exception:
                pass

        if self.lqx.has_ground_energy:
            try:
                self._logger.log(
                    "Exact diagonalization result", self.lqx.ground_energy[0]
                )
            except Exception:
                pass

    @staticmethod
    def _coerce_mean_value(
        mean: Any,
        *,
        sigma: Any | None = None,
        atol: float = 1e-12,
        rtol: float = 1e-10,
        sigma_factor: float = 3.0,
    ) -> float | complex:
        """
        Convert an expectation mean value into a Python scalar robustly.

        Monte Carlo estimates and holomorphic models can sometimes yield small imaginary components even for
        observables that are theoretically real. This helper converts the mean into:

        - ``float`` if the value is real or if the imaginary part is negligible,
        - ``complex`` if a significant imaginary component remains.

        Decision rules
        --------------

        1) If the mean is not complex → return ``float(mean)``.

        2) If the mean is complex:

           - If ``sigma`` is provided and usable, treat ``imag`` as noise if
             ``abs(imag) <= sigma_factor * sigma``.

           - Otherwise, treat ``imag`` as noise if ``abs(imag) <= atol + rtol * abs(real)``.

           - If neither rule triggers, return a Python complex number.

        :param mean: Mean value returned by a Stats-like object; may be real or complex.
        :param sigma: Optional uncertainty estimate used to interpret imaginary components as noise.
        :param atol: Absolute tolerance threshold for considering imaginary parts negligible.
        :param rtol: Relative tolerance threshold for considering imaginary parts negligible.
        :param sigma_factor: Multiplier controlling how aggressively sigma suppresses imaginary parts.
        :return: A Python ``float`` or ``complex`` representing the coerced mean.
        :raises ValueError: If the mean cannot be converted into a numeric scalar.
        """

        arr = np.asarray(mean)

        if not np.iscomplexobj(arr):
            return float(arr)

        re = float(arr.real)
        im = float(arr.imag)

        # error-aware rule, if sigma is supplied and usable
        if sigma is not None:
            try:
                s = float(np.asarray(sigma))
                if s >= 0 and abs(im) <= sigma_factor * s:
                    return re
            except Exception:
                pass

        # pure tolerance rule
        if abs(im) <= atol + rtol * abs(re):
            return re

        return complex(re, im)

    def _log_trailing_averages(self, window: int = 100) -> None:
        """
        Compute and log trailing-window averages for key runtime metrics.

        This helper is intended to provide a stable summary of convergence behaviour by averaging over the
        last ``window`` iterations. When available, it logs:

        - The trailing average of the "Constraint" mean and an estimated standard error.
        - The trailing average of the "Constraint" R_hat diagnostic and an estimated standard error.

        If required log keys are missing or the log is empty, the method returns silently.

        :param window: Number of most recent iterations to include in the trailing window.
        :return: None.
        :raises Exception: All exceptions are suppressed to avoid breaking finalisation.
        """

        if not hasattr(self, "_logger") or self._logger is None:
            return

        try:
            data = self.log.data
            if "Constraint" not in data:
                return

            m = np.asarray(data["Constraint"]["Mean"])
            r = np.asarray(data["Constraint"]["R_hat"])

            if m.size == 0:
                return

            w = int(min(window, m.size))

            avg_m = np.mean(m[-w:])
            err_m = np.std(m[-w:]) / np.sqrt(w)

            if r.size >= w and w > 0:
                avg_r = np.mean(r[-w:])
                err_r = np.std(r[-w:]) / np.sqrt(w)
                r_str = f"{avg_r:.3f}({err_r:.3f})"
            else:
                r_str = "N/A"

            self._logger.log(
                [
                    "Average min<C> over last 100 iterations (stddev)",
                    "Average R_Hat over last 100 iterations (stddev)",
                ],
                [
                    f"{avg_m:.3f}({err_m:.3f})",
                    r_str,
                ],
            )
        except Exception:
            # don't let summary stats break finalisation
            return

    def continue_simulation(
        self,
        *,
        state_path: str | None = None,
        n_iters: int = 500,
        silent_print: bool = False,
        silent_plot: bool = False,
        timer: bool = False,
        live_monitoring: bool = False,
        callbacks: Callable[
            [int, dict, "AbstractVariationalDriver"], bool
        ] = lambda *x: True,
        observables: dict[str, "AbstractOperator"] | None = None,
        force_load_mpi: bool = False,
        **kwargs,
    ) -> None:
        """
        Continue a previously started simulation by resuming from memory or from a serialised checkpoint.

        This method supports two continuation sources:

        1) Resume from disk (``state_path`` provided):

           - Ensures a template variational state/driver exists (needed for safe deserialisation).

           - Imports a full :class:`~neuralqx.vqs.MCState` from disk in an MPI-safe manner.

           - Installs the loaded state into the solver and updates the driver reference.

        2) Resume from memory (``state_path`` is None):

           - If a driver exists, reuse it.

           - Otherwise, initialise the driver/state using current solver configuration.

        Log semantics
        -------------
        The runtime log is preserved across continuations unless it is missing, in which case a new log is
        created. Observables default to the previously registered set unless explicitly overridden.

        :param state_path: Optional path to a previously exported MCState checkpoint.
        :param n_iters: Number of additional optimisation iterations to execute.
        :param silent_print: If True, suppress printing of final log output.
        :param silent_plot: If True, suppress interactive plot display.
        :param timer: If True, enable timing/profiling integration in the driver.
        :param live_monitoring: If True, attach live monitoring callback for this continuation run.
        :param callbacks: Callback or list of callbacks invoked after each iteration.
        :param observables: Optional observable registry for this continuation run. If None, reuse the last
            registered observables (if any).
        :param force_load_mpi: If True, allow best-effort reconstruction even if MPI configuration differs
            from the environment used when saving.
        :param kwargs: Implementation-specific options. May include ``diagonal_shift`` override.
        :return: None.
        :raises RuntimeError: If continuation is requested but solver components are not initialised and
            cannot be initialised.
        :raises OSError: If ``state_path`` is provided but the file cannot be read.
        :raises ValueError: If the checkpoint is invalid or incompatible with the current environment.
        """

        # handle optional diagonal_shift override
        if "diagonal_shift" in kwargs and kwargs["diagonal_shift"] is not None:
            try:
                self.diagonal_shift = float(kwargs["diagonal_shift"])
            except Exception:
                # keep existing diagonal_shift if casting fails
                pass

        # ensure a runtime log exists (do NOT reset it here)
        if not hasattr(self, "_nk_log") or self._nk_log is None:
            self._nk_log = RuntimeLog()

        # If we are resuming from disk, we MUST ensure a template variational state exists before
        # calling import_state() because deserialize_MCState(...) uses self.variational_state
        if state_path is not None:

            # ensure the driver/state exist ONCE to provide a template
            # this should not overwrite the loaded state because we will
            # immediately replace it after import.
            if not getattr(self, "_driver_flag", False):
                self.initialize_vmc(
                    diagonal_shift=self.diagonal_shift
                )

            # now we should have a template variational state
            loaded_state = self.import_state(
                state_path,
                force_load_mpi=force_load_mpi,
            )

            # replace solver state with loaded one
            self._variational_state = loaded_state
            self._vstate_flag = True

            # make sure driver uses the loaded state
            try:
                self.vmc_driver._variational_state = loaded_state
            except Exception:
                pass

        else:
            # no state_path: ensure we have a driver if we are truly continuing from in-memory
            # state. If this is a fresh instance, initialise
            if not getattr(self, "_driver_flag", False):
                self.initialize_vmc(
                    diagonal_shift=getattr(self, "diagonal_shift", None)
                )

        # decide which observables to use for this continuation
        obs_to_use = (
            observables if observables is not None else self._registered_observables
        )

        # delegate the actual loop to run() in continuation mode
        self.run(
            n_iters,
            silent_plot=silent_plot,
            silent_print=silent_print,
            timer=timer,
            live_monitoring=live_monitoring,
            callbacks=callbacks,
            observables=obs_to_use,
            # tells _prepare_run not to reset/reattach fresh-run state
            _ct=True,
        )

    @errors_only(tag="SOLVER:EXPORT_STATE")
    def export_state(
        self,
        *,
        state: MCState | None = None,
        silent: bool = False,
        marker: str = "",
        **kwargs,
    ) -> str | None:
        """
        Export (serialise) a complete Monte Carlo variational state to disk.

        This method exports the full :class:`~neuralqx.vqs.MCState` (including sampler configuration/state)
        so that a run can be reproduced even if the checkpoint was taken before convergence.

        MPI synchronisation rationale
        -----------------------------
        Serialisation converts JAX arrays into host buffers. This conversion blocks until all outstanding
        device work is complete, which may include MPI-backed collectives issued during the last VMC step.
        To avoid deadlocks and ensure all ranks reach a consistent point:

        - All ranks serialise and synchronise on an MPI barrier.

        - Rank-0 writes the checkpoint to disk.

        - All ranks synchronise again before returning.

        File naming
        -----------
        The exported filename includes:

        - the solver hash,

        - a timestamp,

        - an optional ``marker`` suffix (e.g. "FinalState", "AbortedState").

        :param state: Optional explicit state to export. If None, export the solver’s current state.
        :param silent: If True, suppress user-facing confirmation messages.
        :param marker: Optional label appended to the exported filename for easy identification.
        :param kwargs: Implementation-specific export options (currently unused).
        :return: The exported file path if the implementation chooses to return it; otherwise ``None``.
        :raises RuntimeError: If no variational state exists and ``state`` is None.
        :raises OSError: If the checkpoint cannot be written to disk (rank-0).
        """

        # NOTE:
        # the serialisation done here uses flax.serialization.to_bytes() which converts every
        # jax.Array in vstate.variables to a NumPy buffer. That conversion blocks until all
        # outstanding JAX kernels that produced those arrays have finished (block_until_ready()
        # under the hood). Some of those kernels include mpi4jax collectives issued during the
        # last VMC step. They will not complete unless every rank reaches the corresponding MPI call
        # and flushes its own queue. Therefore, all states must serialise the state and wait at a
        # barrier, and then rank-0 proceeds to export

        # choose a valid MCState
        vstate = state or self._variational_state

        # serialise the current MCState
        serialised_state = serialize_MCState(vstate)

        # everyone synchronise here
        _mpi.barrier()

        # rank-0 only, workers skip export
        if _mpi.is_global_master():
            if marker != "":
                marker = "_" + marker

            datetime_str = datetime.now().strftime("%d%m%Y_%H%M%S")
            filename = os.path.join(
                self.output_path,
                f"{self.hash}_SerialisedState_{datetime_str}{marker}.mpack",
            )

            # save to disc
            save_to_file(serialised_state, filename)

            if not silent:
                self.printer.print("State serialised to disc.")

        # all other workers wait here
        _mpi.barrier()

        return None

    @errors_only(tag="SOLVER:IMPORT_STATE")
    def import_state(
        self,
        state_path: str,
        *,
        force_load_mpi: bool = False,
    ) -> MCState:
        r"""
        Import and deserialise a variational state from a checkpoint file.

        This method loads a full serialised :class:`~neuralqx.vqs.MCState` in an MPI-safe way:

        - Rank-0 reads the checkpoint bytes from disk.

        - The raw checkpoint payload is broadcast to all ranks.

        - Each rank reconstructs a new MCState using :func:`deserialize_MCState`.

        Template requirement
        --------------------
        Deserialisation may rely on a "template" MCState for shape/dtype/model reconstruction. This
        implementation expects that ``self.variational_state`` exists when calling this method. Continuation
        logic ensures a template is created before import if needed.

        :param state_path: Path to a previously exported checkpoint file.
        :param force_load_mpi: If True, force best-effort reconstruction even if MPI configuration differs
            from the environment used when saving.
        :return: A reconstructed :class:`~neuralqx.vqs.MCState` instance.
        :raises OSError: If the checkpoint file cannot be read on rank-0.
        :raises RuntimeError: If a required template state is not available for deserialisation.
        :raises ValueError: If the checkpoint payload is invalid or incompatible.
        """

        # synchronise every worker
        _mpi.barrier()

        # load the state from file, only rank-0
        if _mpi.is_global_master():
            nvs = load_from_file(state_path, raw=True)
        else:
            nvs = None

        # broadcast to all workers
        nvs = mpi_bcast(nvs, root=0)

        # reconstruct and return a new MCState
        return deserialize_MCState(
            self.variational_state, nvs, force_load_mpi=force_load_mpi
        )

    def plot_results(
        self,
        *,
        silent_plot: bool = False,
        with_inset: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot and save the main minimisation curve and export auxiliary run artefacts.

        This method produces:
        - a plot of the constraint expectation value versus iteration with error bars,
        - an optional inset zoom into the final iterations (useful for convergence inspection),
        - a saved image of the underlying graph when supported by the graph object,
        - a serialised copy of the runtime log data to disk.

        Plot content
        ------------

        - The primary curve uses the runtime log series under ``"Constraint"``:

          * x-axis: iteration index

          * y-axis: mean value

          * error bars: sigma

        - If the model provides an exact diagonalisation ground energy, a horizontal reference line is
          drawn.

        Inset behaviour
        --------------
        When ``with_inset=True``, the method:

        - zooms into the last ~20 iterations,

        - sanitises data via :func:`reject_outliers` for stable y-limits,

        - draws an inset with matching content and a marked region in the main plot.

        :param silent_plot: If True, suppress interactive plot display (figures are still saved).
        :param with_inset: If True, include an inset zoom of the final part of the trajectory.
        :param kwargs: Implementation-specific plotting options (currently unused).
        :return: None.
        :raises KeyError: If expected log keys (e.g. ``"Constraint"``) are missing from the runtime log.
        :raises OSError: If output images or log serialisation cannot be written to disk.
        """

        # set the dpi, a high dpi is used for publishing
        dpi = 300
        dpi_ratio = dpi / 100

        # create the filename for exporting
        dt_str = datetime.now().strftime("%d%m%Y_%H%M%S")
        image_filename = os.path.join(
            self.output_path, f"{self.hash}_ModelResults_{dt_str}.png"
        )

        graph_filename = os.path.join(
            self.output_path, f"{self.hash}_Graph_{dt_str}.png"
        )

        # export the graph
        try:
            self.graph.export(graph_filename, dpi=dpi)
        except Exception as exc:
            self.printer.print(f"Failed to export graph: {exc}")

        # the data from the NetKet log
        data_nn = self.log.data

        # create the figure
        fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)  # pylint: disable=W0612

        # check for ground energy
        if self.lqx.has_ground_energy:
            ge = self.lqx._ground_energy
        else:
            ge = None

        has_ge = ge is not None and not jnp.isnan(ge)

        # plot the exact diag. result if it exists
        if has_ge:
            ax.hlines(
                [ge],
                xmin=0,
                xmax=self._max_iters,
                color="#E25C47",
                label="ED Result",
                linewidth=2,
            )

        # plot the simulation data with error bars
        ax.errorbar(
            data_nn["Constraint"].iters,
            data_nn["Constraint"].Mean,
            yerr=data_nn["Constraint"].Sigma,
            label="neuraLQX",
            alpha=0.5,
            color="#255957",
            marker="o",
            markersize=3.5,
            capsize=5,
            linestyle="",
            ecolor="#70C1B3",
            elinewidth=1.5,
        )

        # customise the legend
        ax.legend(fontsize="x-large")

        # customise the axes
        ax.set_xlabel("Iteration", fontsize="xx-large")
        ax.set_ylabel(r"$\langle \hat{\mathcal{Q}}\rangle$", fontsize="xx-large")
        ax.tick_params(axis="both", which="both", labelsize="x-large")

        # add grid
        ax.grid(True, linestyle="--", alpha=0.7)

        # inset x-range
        max_x = self._max_iters
        min_x = max_x - 20

        # sanitise data
        cleaned_data_nn = reject_outliers(data_nn["Constraint"][min_x:max_x].Mean)

        # set y-axis the limits of the magnified plot
        # base bounds from NN data only
        if cleaned_data_nn.size == 0:
            # nothing finite to plot in inset
            # just skip the inset block
            with_inset = False
        else:
            data_min = float(np.min(cleaned_data_nn))
            data_max = float(np.max(cleaned_data_nn))

        if with_inset:
            # inset setup
            inset_position_100dpi = [170, -90, 850, 700]
            inset_position = [ip * dpi_ratio for ip in inset_position_100dpi]

            # create an inset_axes object for the magnified plot
            ax_ins = inset_axes(
                ax, width="30%", height="30%", bbox_to_anchor=inset_position
            )

            # plot the magnified part of the data
            # the simulation data
            ax_ins.errorbar(
                data_nn["Constraint"].iters,
                data_nn["Constraint"].Mean,
                yerr=data_nn["Constraint"].Sigma,
                label="NN",
                alpha=0.8,
                color="#255957",
                marker="o",
                markersize=5,
                capsize=5,
                linestyle="",
                ecolor="#70c1b3",
                elinewidth=1.5,
            )

            # and the ED data
            if has_ge:
                ax_ins.hlines(
                    [ge],
                    xmin=min_x,
                    xmax=max_x,
                    color="#E25C47",
                    label="ED Result",
                    linewidth=2,
                )

            if has_ge:
                min_y = min(data_min, ge) - 0.05
                max_y = max(data_max, ge) + 0.15
            else:
                min_y = data_min - 0.05
                max_y = data_max + 0.15

            # set the limits of the magnified plot
            ax_ins.set_xlim(min_x, max_x)
            ax_ins.set_ylim(min_y, max_y)

            # set the inset location
            mark_inset(ax, ax_ins, loc1=3, loc2=1, fc="none", ec="0.6", linestyle="--")

            # add a grid to the magnified plot
            ax_ins.grid(True, linestyle="--", alpha=0.7)

            # ticks
            ax_ins.minorticks_on()

        # ticks
        ax.minorticks_on()

        # save image to disc
        plt.savefig(image_filename, dpi=dpi)

        # serialise the simulation data to disc
        self.log.serialize(
            os.path.join(self.output_path, f"{self.hash}_LogData_{dt_str}")
        )

        # display plot if requested
        if (not silent_plot) and (_mpi.n_nodes == 1):
            plt.show()
        else:
            plt.close()

    def plot_observables(
        self,
        *observables: str,
        **kwargs,
    ) -> None:
        r"""
        Plot one or more logged observables from the runtime log.

        For each requested observable key, this method attempts to read:

        - ``Mean`` series,

        - ``Sigma`` series,

        and produces a plot of mean versus iteration with a shaded mean±sigma band.

        Observable lookup
        -----------------
        Observable data is retrieved from the solver runtime log using the provided keys. If an observable
        is missing, a ValueError is raised with a user-facing message.

        :param observables: One or more observable names/keys as stored in :attr:`log`.
        :param kwargs: Implementation-specific plotting options (currently unused).
        :return: None.
        :raises ValueError: If an observable key is not found in the runtime log.
        """

        # define some distinct colors for the plots
        colors = [
            "#255957",
            "#1F77B4",
            "#FF7F0E",
            "#2CA02C",
            "#D62728",
            "#9467BD",
            "#8C564B",
            "#E377C2",
        ]

        # check if the number of colors matches the number of observables
        if len(observables) > len(colors):
            raise ValueError("""
                Too many observables for the available colors. Add more colors or reduce the number of observables.
                """)

        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        # loop through the given observables and plot each one
        for idx, observable in enumerate(observables):
            try:
                # try to access mean and sigma from the RuntimeLog
                mean_values = self.log[observable]["Mean"]
                sigma_values = self.log[observable]["Sigma"]
                x = np.arange(len(mean_values))

                # plot the mean values with a line and markers
                ax.plot(
                    x,
                    mean_values,
                    label=rf"$\mathcal{{\hat{{O}}}} = {observable}$",
                    alpha=0.5,
                    color=colors[idx],
                    marker="o",
                    markersize=3.5,
                    linestyle="-",
                    linewidth=1.5,
                )

                # plot the shaded error region (mean ± sigma)
                ax.fill_between(
                    x,
                    mean_values - sigma_values,
                    mean_values + sigma_values,
                    color=colors[idx],
                    alpha=0.3,
                )
            except KeyError:
                # missing observable case
                raise ValueError(
                    f"Observable '{observable}' not found in solver.nk_log."
                )

        ax.minorticks_on()
        ax.set_xlabel("Iteration", fontsize="xx-large")
        ax.set_ylabel(r"$\langle \mathcal{\hat{O}} \rangle$", fontsize="xx-large")
        ax.tick_params(axis="both", which="both", labelsize="x-large")
        ax.legend(fontsize="x-large")

        # add grid
        ax.grid(True, linestyle="--", alpha=0.7)

        # display the plot
        # dev: save the plots maybe?
        plt.show()

    @required_kwargs(
        "sampler_type",
        conditional_args={
            "sampler_type": {
                "Metropolis Exchange": ["d_max"],
                "Metropolis Hamiltonian": ["Hamiltonian"],
                "Weighted Sampler": ["rules", "probabilities"],
                "Parallel Tempering Local": ["betas", "n_replicas"],
                "Parallel Tempering Exchange": ["betas", "n_replicas", "d_max"],
                "Parallel Tempering Weighted Sampler": [
                    "betas",
                    "n_replicas",
                    "rules",
                    "probabilities",
                ],
                "Parallel Tempering U1 Gauge Sampler": ["betas", "n_replicas"],
                "Parallel Tempering Random U1 Gauge Sampler": ["betas", "n_replicas"],
                "Parallel Tempering U1 Plaquette Sampler": ["betas", "n_replicas"],
                "Metropolis K Local": ["n_flips"],
                "Metropolis Multi Hamiltonian": ["Hamiltonian"],
                "Parallel Tempering Hamiltonian list": ["Hamiltonian"],
            }
        },
    )
    def set_sampler(
        self,
        sampler_type: str,
        *,
        number_of_chains: int = 16,
        number_of_sweeps: int = 20,
        machine_power: int = 2,
        reset_chains: bool = True,
        number_of_samples: int = 512,
        **kwargs,
    ) -> None:
        """
        Configure and attach a sampler to the solver.

        This method builds a sampler using the neuraLQX sampler factory and stores:

        - the primary sampler used by the variational state,

        - an auxiliary sampler copy (when provided by the factory),

        - the sampler initialisation kwargs for later reconstruction/logging,

        - the number of samples requested per chain.

        Gauge sampler compatibility
        ---------------------------
        If the requested sampler name indicates a gauge sampler (substring "gauge"), this method enforces
        that the underlying Hilbert space is gauge invariant. Otherwise it raises NotImplementedError to
        avoid silent misuse.

        Logging
        -------
        Sampler metadata is exported to the solver logger. Sampler kwargs are also recorded under
        "Sampler Configs", excluding keys that are already covered by the sampler metadata.

        :param sampler_type: Human-readable sampler identifier supported by neuraLQX.
        :param number_of_chains: Total number of Monte Carlo chains to use.
        :param number_of_sweeps: Number of sweeps per sampling step.
        :param machine_power: Exponent used in the sampling probability distribution (backend-specific).
        :param reset_chains: If True, reset chains at each iteration (backend-specific).
        :param number_of_samples: Number of samples per chain.
        :param kwargs: Sampler-specific parameters (e.g. ``d_max``, ``rules``, ``probabilities``,
            ``Hamiltonian``, temperature ladder, etc.).
        :return: None.
        :raises NotImplementedError: If a gauge sampler is requested for a non-gauge-invariant Hilbert
            space.
        :raises ValueError: If sampler configuration parameters are invalid for the chosen sampler type.
        """

        # erase the log if we are setting a new sampler after setting an old one
        if self._sampler_flag:
            self._logger.erase_parent_content("Sampler Configs")
            self._sampler_flag = False

        # the check of whether the requested sampler is available or not is now done in the Sampler
        # class, so it is skipped here.
        # we now check if the requested sampler is a GI sampler, but we do not have a GI space
        if "gauge" in sampler_type.lower():
            if not self.hilbert.is_gauge_invariant:
                raise NotImplementedError(f"""

                    Currently, the `{sampler_type}` sampler is only implemented for gauge invariant Hilbert spaces.

                    This sampler will be available for general, non gauge invariant Hilbert
                     spaced in a future release. If you need to use it before then,  get in touch with us, we are
                     interested!

                    """)

        # create the sampler object
        # kwargs will have sampler specific kwargs such as d_max, rules, probaiblities, etc.
        sampler_builder = SamplerNQX(
            sampler_type=sampler_type,
            number_of_chains=number_of_chains,
            number_of_sweeps=number_of_sweeps,
            machine_pow=machine_power,
            reset_chains=reset_chains,
            number_of_samples=number_of_samples,
            **kwargs,
        )

        # set the sampler n_states
        self._n_samples = number_of_samples

        # build the sampler
        self._sampler, self._aux_sampler, self._sampler_kwargs = sampler_builder.build(
            hilbert=self.hilbert.hilbert,
            hamiltonian=kwargs.get("Hamiltonian", self.lqx.constraint),
            graph=self.graph.dual_nk_graph,
        )

        # logging below
        fields, values = sampler_builder.export_info()
        self._logger.log(fields, values)

        # skip these keys as they are already exported in the sampler's metadata
        skip_keys = [
            "n_chains_per_rank",
            "sweep_size",
            "machine_pow",
            "reset_chains",
        ]

        for k, v in self._sampler_kwargs.items():
            if k not in skip_keys:
                self._logger.add_field("Sampler Configs", str(k))
                self._logger.log(str(k), str(v))

        # mark the sampler as set
        self._sampler_flag = True

        # confirm to the user that the sampler is set
        self.printer.print(f"Sampler `{sampler_type}` initialized.")

    def set_optimizer(
        self,
        optimizer_type: str = "Adam",
        *,
        use_sr: bool,
        preconditioner_solver: str = "Conjugate Gradient",
        preconditioner: AbstractLinearPreconditioner | None = None,
        learning_rate: float | None = None,
        scheduler_type: str | None = None,
        scheduler_parameters: dict[str, Any] | None = None,
        diagonal_shift: float | None = None,
        **kwargs,
    ) -> None:
        """
        Configure and attach an optimiser and the SR preconditioner linear solver.

        This method constructs:

        - an Optax optimiser via the neuraLQX :class:`~neuralqx.optimizer.Optimizer` builder,

        - a linear solver backend for stochastic reconfiguration (SR) via :class:`~neuralqx.optimizer.solvers.Solvers`,

        - and stores a diagonal shift value used later when constructing the SR preconditioner.

        Logging
        -------
        Optimiser metadata and solver metadata are exported to the solver logger under "Optimizer Configs".

        :param optimizer_type: Name of the optimiser supported by neuraLQX (e.g. "Adam").
        :param preconditioner_solver: Name of the linear solver used inside SR (e.g. "Conjugate Gradient").
        :param preconditioner: The preconditioner object used in the natural gradient descent, if not specified SR is
          used by default.
        :param learning_rate: Optional constant learning rate. If None, a sensible default is chosen by the
            optimiser builder.
        :param scheduler_type: Optional learning-rate schedule identifier.
        :param scheduler_parameters: Optional mapping of schedule parameters passed to the schedule builder.
        :param diagonal_shift: Optional diagonal shift regularisation used by SR. If None, defaults to 0.1.
        :param kwargs: Additional optimiser-specific configuration parameters forwarded to the builder.
        :param use_sr: if True, a stochastic reconfiguration (SR) preconditioner will be used
        :return: None.
        :raises ValueError: If optimiser or schedule configuration is invalid.
        :raises RuntimeError: If the optimiser or solver backend cannot be constructed.
        """

        # erase the log if we are setting a new sampler after setting an old one
        if self._opt_flag:
            self._logger.erase_parent_content("Optimizer Configs")
            self._opt_flag = False

        # create the optimiser based on the params
        opt_build = Optimizer(
            optimizer_name=optimizer_type,
            learning_rate=learning_rate,
            scheduler_type=scheduler_type,
            scheduler_kwargs=scheduler_parameters,
            **kwargs,
        )

        # build and set the optimiser
        self._optimizer = opt_build.build()

        # get optimiser metadata and update the logger
        for f, v in zip(*opt_build.export_info()):
            # add the field
            self._logger.add_field("Optimizer Configs", f)

            # log the values
            self._logger.log(f, v)

        # we also create the solver for the SR preconditioner
        sol_build = Solvers(
            solver_name=preconditioner_solver,
        )

        self._preconditioner_solver = sol_build.build()

        # get the solver metadata and update the logger
        for (
            f,
            v,
        ) in zip(*sol_build.export_info()):
            # add the field
            self._logger.add_field("Optimizer Configs", f)

            # log the values
            self._logger.log(f, v)

        # save the diagonal shift for the VMC later
        self.diagonal_shift = diagonal_shift or 0.1

        # set the optimiser to initialised
        self._opt_flag = True

        # check for collision
        if preconditioner is not None and use_sr:
            warnings.warn(
                "`use_sr=True` but a preconditioner was supplied. Ignoring the `use_sr=True` option "
                "and the provided preconditioner will be used.",
                RuntimeWarning,
                stacklevel=2,
            )

        # set the preconditioner
        if preconditioner is not None:
            self._preconditioner = preconditioner

        # set the SR option internally
        self._use_sr = use_sr

        # inform the user of successful init of the optimiser
        self.printer.print(f"Optimizer `{optimizer_type}` initialized.")

    def set_network(
        self,
        network: "FlaxModule",
        *,
        diff_invariant: bool = False,
        symmetries: Sequence[Any] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ) -> None:
        """
        Configure and attach a neural network Ansatz to the solver.

        This method:

        - records network type and attributes to the solver logger,

        - stores a copy of the original network for internal use when needed,

        - optionally wraps the network with a group/symmetry projector when running a diffeomorphism
          invariant simulation,

        - stores the final network module and updates internal flags.

        Diffeomorphism-invariant mode
        -----------------------------
        If ``diff_invariant=True``, the solver marks the run as diffeomorphism invariant and wraps the
        network using :func:`~neuralqx.nn.projectors.group_projector.wrap_model`. The wrapper applies a
        group-averaging/projection procedure over the provided symmetry descriptors.

        :param network: Neural network module (typically a subclass of ``flax.linen.Module``).
        :param diff_invariant: If True, enable diffeomorphism-invariant (group-averaged) simulation mode.
        :param symmetries: Optional symmetry descriptors used by the projection wrapper.
        :param kwargs: Implementation-specific parameters (currently unused).
        :param chunk_size: If specified, the expectation values and gradients are computed in a chunked manner where the
          largest chunk has size ``chunk_size``
        :return: None.
        :raises ValueError: If symmetry configuration is inconsistent with the graph/model.
        :raises RuntimeError: If the network wrapper cannot be constructed.
        """

        # erase the log if we are setting a new sampler after setting an old one
        if self._network_flag:
            self._logger.erase_parent_content("Network Configs")
            self._network_flag = False

        # get and log the network name:
        _net_name = (
            network.__name__
            if hasattr(network, "__name__")
            else network.__class__.__name__
        )
        self._logger.log("Network type", _net_name)

        # get and log the network attributes
        log_module_attributes(self, network)

        # dev: check if we still need this
        # save a copy for _gexpect()
        self._gnetwork = network

        # save the diffeo invariant request and project the selected network
        self._diff_inv = diff_invariant
        if self.diffeomorphism_invariant_simulation is True:
            network = wrap_model(
                network,
                symmetries=symmetries,
                graph=self.graph,
                characters=None,
            )

        # set the network in the solver
        self._network = network

        # change the flag
        self._network_flag = True

        self._chunk_size = chunk_size if chunk_size is not None else None

        # inform the user of successful init
        self.printer.print(
            f"{'Diffeomorphism Invariant ' if self.diffeomorphism_invariant_simulation else ''}"
            f"Network `{_net_name}` initialized."
        )

    def expect(
        self,
        operator: (
            list[Union["AbstractOperator", "AbstractObservable"]]
            | Union["AbstractOperator", "AbstractObservable"]
        ),
        *,
        n_samples: int | None = None,
        n_chains: int | None = None,
        use_same_variables: bool = True,
        use_same_sampler: bool = True,
        **kwargs,
    ) -> "Stats":
        """
        Compute the expectation value of an operator or list of operators.

        This method provides explicit sampling semantics to avoid accidental misuse.

        Supported modes
        ---------------

        1) Alias mode (default):
           If only ``operator`` is provided, this is a direct alias to
           ``variational_state.expect(operator)``.

        2) Alias mode with temporary sample-count override:
           If ``n_samples`` is provided and ``n_chains`` is not provided, this method temporarily overrides
           ``variational_state.n_samples``, warns the user that samples will be regenerated, evaluates the
           expectation, and restores the original sample count.

        Unsupported mode (planned)
        --------------------------
        If ``n_chains`` is provided, the intended behaviour is to build a temporary state with a modified
        sampler and draw fresh samples. This mode is not implemented and currently raises
        NotImplementedError.

        Ignored arguments
        ----------------
        In alias modes, arguments that would only apply to "fresh sampling" (such as ``use_same_variables``,
        ``use_same_sampler``, and additional kwargs) are ignored with a warning.

        :param operator: Operator/observable or a list of operators/observables to evaluate.
        :param n_samples: Optional number of samples per chain (alias mode override).
        :param n_chains: Optional number of chains (fresh sampling request; not implemented).
        :param use_same_variables: Intended for fresh sampling mode; ignored in alias modes.
        :param use_same_sampler: Intended for fresh sampling mode; ignored in alias modes.
        :param kwargs: Additional options; ignored in alias modes.
        :return: A Stats-like object containing the estimated expectation value(s).
        :raises RuntimeError: If the variational state is not initialised.
        :raises NotImplementedError: If ``n_chains`` is provided (fresh sampling mode not implemented).
        """

        # guard: ensure a variational state exists
        if not getattr(self, "_vstate_flag", False):
            raise RuntimeError(
                "Cannot compute expectation value: variational state is not initialised.\n\n"
                "Try calling Solver.initialize_vmc() after setting the network, sampler and "
                "optimizer."
            )

        # fresh sampling
        if n_chains is not None:
            raise NotImplementedError(
                "Expectation with explicit `n_chains` is not yet implemented. "
                "This mode will construct a temporary state with a modified sampler "
                "in a future release."
            )

        # alias to variational_state.expect

        # warn if user supplied parameters that will be ignored
        ignored_args = []

        if kwargs:
            ignored_args.extend(kwargs.keys())

        if not use_same_variables:
            ignored_args.append("use_same_variables")

        if not use_same_sampler:
            ignored_args.append("use_same_sampler")

        if ignored_args:
            warnings.warn(
                "The following arguments are ignored when calling `expect` in alias mode: "
                f"{', '.join(sorted(set(ignored_args)))}",
                RuntimeWarning,
                stacklevel=2,
            )

        # handle optional n_samples override
        if n_samples is not None:
            warnings.warn(
                "Calling `Solver.expect(..., n_samples=...)` temporarily overrides the number of "
                "samples in the current variational state. This will discard and regenerate "
                "the existing Monte Carlo samples.",
                RuntimeWarning,
                stacklevel=2,
            )

            # save original value to restore later
            original_n_samples = getattr(self.variational_state, "n_samples", None)

            try:
                self.variational_state.n_samples = int(n_samples)
                return self.variational_state.expect(operator)
            finally:
                # Restore original sample count
                if original_n_samples is not None:
                    self.variational_state.n_samples = original_n_samples

        # pure alias mode
        return self.variational_state.expect(operator)

    def initialize_vmc(self, *args, **kwargs) -> None:
        """
        Initialise the variational state and VMC driver.

        This method constructs the objects required to execute optimisation:

        1) Validates configuration:
           Requires that sampler, optimiser, and network have been configured via:

           - :meth:`set_sampler`

           - :meth:`set_optimizer`

           - :meth:`set_network`

        2) Builds the Monte Carlo variational state:
           Constructs :class:`~neuralqx.vqs.MCState` using the configured sampler and network and stores it
           in the solver. It also records the number of trainable parameters and the ratio of parameter
           count to Hilbert dimension (when estimable).

        3) Builds the VMC driver:
           Constructs :class:`~neuralqx.driver.VMC` using the model constraint as the optimisation target,
           the configured optimiser, and an SR preconditioner:

           - diagonal shift taken from :attr:`diagonal_shift`,

           - holomorphic flag inferred via NetKet's holomorphic diagnostic,

           - linear solver backend taken from :attr:`preconditioner_solver`.

        :param args: Implementation-specific initialisation options (currently unused).
        :param kwargs: Implementation-specific initialisation options (may be ignored by this implementation).
        :return: None.
        :raises PermissionError: If sampler, optimiser, or network have not been configured.
        :raises RuntimeError: If the variational state or driver cannot be constructed.
        """

        if not self._sampler_flag or not self._opt_flag or not self._network_flag:
            raise PermissionError(f"""

                You cannot run the VMC driver without setting the sampler, optimizer as well as the network!

                Try using the `{self.set_sampler.__name__}()`, `{self.set_optimizer.__name__}()`  and the
                 `{self.set_network.__name__}()` functions with valid parameters first.

                """)

        # set the variational state
        self._variational_state = MCState(
            self.sampler,
            self.network,
            n_samples=self._n_samples,
            is_group_averaged=self.diffeomorphism_invariant_simulation,
            seed=self.seed,
            sampler_seed=self.seed // 2,
            chunk_size=self.chunk_size,
        )

        is_holomorphic = is_probably_holomorphic(
            self._variational_state._apply_fun,
            self._variational_state.parameters,
            self._variational_state.samples,
            self._variational_state.model_state,
        )

        # log the number of network parameters
        self._logger.log(
            "Number of network parameters", self._variational_state.n_parameters
        )

        # log the ratio of networkParams / dim_H
        self._logger.add_field("Network Configs", "Params(network) / dim(Hilbert)")
        if self.hilbert.is_indexable:
            dim_H = self.hilbert.dimensions
        else:
            if self.hilbert.tiny_hilbert.is_indexable:
                dim_H = (
                    self.hilbert.tiny_hilbert.n_states**self.hilbert.gauge_dimensions
                )
            else:
                dim_H = (
                    (2 * self.hilbert.cutoff + 1) ** len(self.graph.edges)
                ) ** self.hilbert.gauge_dimensions
        self._logger.log(
            "Params(network) / dim(Hilbert)",
            f"{humanize.number.scientific(100 * self._variational_state.n_parameters / dim_H, precision = 4)} %",
        )

        # set the flag
        self._vstate_flag = True

        # check the preconditioner
        if self.preconditioner is None:
            # default to SR if requested
            if self._use_sr:
                self.preconditioner = SR(
                    # diagonal shift should be set if we have set the optimiser, in a sense the flag
                    # check above is a soft guard
                    diag_shift=self.diagonal_shift,
                    holomorphic=bool(is_holomorphic),
                    solver=self.preconditioner_solver,
                )
            else:
                # identity preconditioner
                self.preconditioner = identity_preconditioner

        # add the preconditioner to the log
        self._logger.add_field("Optimizer Configs", "Preconditioner")
        self._logger.log("Preconditioner", type(self.preconditioner).__name__)

        # init the driver
        self._vmc_driver = VMC(
            self.lqx.constraint,
            self.optimizer,
            variational_state=self._variational_state,
            preconditioner=self.preconditioner,
        )

        # set the flag
        self._driver_flag = True

    def __repr__(self):
        return f"Solver(is_initialised={self.is_initialised}, seed={self.seed}, hash={self.hash})"
