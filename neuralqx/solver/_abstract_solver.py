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
Base solver abstraction for neuraLQX optimisation drivers.

This module defines :class:`~neuralqx.solvers.AbstractSolver`, the common orchestration layer used by
all neuraLQX “solver” implementations. A solver glues together:

- a model interface (:class:`~neuralqx.lqx.AbstractLqxInterface`),
- the graph and gauge structure (:class:`~neuralqx.graph.core.AbstractGraph`,
  :class:`~neuralqx.gauge_groups.AbstractGaugeGroup`),
- a Hilbert-space interface (:class:`~neuralqx.hilbert.AbstractHilbertInterface`),
- a variational state (:class:`~neuralqx.vqs.MCState`),
- a sampler (:class:`netket.sampler.Sampler`),
- an Optax optimiser (:class:`optax.GradientTransformation`),
- a VMC-style driver (:class:`~neuralqx.driver.VMC`),
- runtime logging and run metadata (MPI, platform info, run hash, output paths).

Concrete solvers should implement the abstract methods to provide a consistent user-facing workflow:
configure sampler/optimiser/network -> initialise VMC -> run/continue -> export/import -> plot results.
"""

import abc
import logging
import os.path
import secrets
import platform

from collections.abc import Sequence
from collections.abc import Callable

from typing import Any
from typing import Union
from typing import TYPE_CHECKING

import humanize
from netket.logging import RuntimeLog
from netket.optimizer.preconditioner import AbstractLinearPreconditioner

from optax._src.base import GradientTransformation

from neuralqx.debug import event
from neuralqx.driver import VMC
from neuralqx.gauge_groups import AbstractGaugeGroup
from neuralqx.graph.core import AbstractGraph
from ..lqx.abstract_lqx_interface import AbstractLqxInterface
from neuralqx.hilbert import AbstractHilbertInterface
from neuralqx.utils.io.printing import NQXPrinter
from neuralqx.utils.io.loggers import Logger
from neuralqx.utils.misc.auth import get_hash
from neuralqx.vqs import MCState
from neuralqx.utils import distributed as _dist
from neuralqx.utils.io.runtime_loggers import DeferredRuntimeLog


from netket.sampler import Sampler

import jax

if TYPE_CHECKING:
    # these imports are only for type checking/docs
    # concrete implementations may use Flax/NetKet types or neuraLQX wrappers
    from flax.linen import Module as FlaxModule
    from netket.operator import AbstractOperator
    from netket.stats.mc_stats import Stats
    from netket.driver import AbstractVariationalDriver


class AbstractSolver(abc.ABC):
    """
    Abstract base class for neuraLQX solvers.

    A solver is responsible for orchestrating the full optimisation workflow around a given LQX model:
    configuration, initialisation, execution, logging, checkpointing, and result visualisation.

    This base class provides:
    - consistent wiring of model components (graph, gauge group, Hilbert interface),
    - reproducible run identity via a unique hash and seed,
    - output directory management with MPI-safe directory creation and optional cleanup,
    - a unified logging surface via a runtime log object,
    - a set of “initialisation flags” that gate access to derived objects (sampler, optimiser, network,
      driver, variational state).

    Concrete implementations must provide the operational methods such as:
    - constructing samplers/optimisers/networks,
    - building the variational state and VMC driver,
    - running and resuming optimisation loops,
    - exporting/importing state,
    - plotting results and observables.
    """

    _graph: AbstractGraph
    """The graph the LQX model is defined on."""

    _hilbert: AbstractHilbertInterface
    """The Hilbert space interface the LQX model is defined on."""

    _gauge_group: AbstractGaugeGroup
    """The gauge group to be used in the LQX model"""

    _lqx: AbstractLqxInterface
    """The LQX interface implementing the model to be solved."""

    _printer: Any | None
    """A user-facing printing helper."""

    _hash: str
    """A unique hash for the current simulation (used for output logs)."""

    _seed: int
    """A seed for the current simulation."""

    _output_path: str
    """The output path for the current simulation."""

    _aux_path: str
    """The subdirectory name for the current simulation in the output directory."""

    _solved: bool
    """A flag to determine if the simulation was completed."""

    _opt_flag: bool
    """A flag to determine whether an optimiser has been set."""

    _sampler_flag: bool
    """A flag to determine whether a sampler has been set."""

    _network_flag: bool
    """A flag to determine whether a network has been set."""

    _driver_flag: bool
    """A flag to determine whether a VMC driver has been set."""

    _vstate_flag: bool
    """A flag to determine whether a variational state has been set."""

    _registered_observables: dict | None
    """A dictionary of observables (if available) observed during training."""

    _max_iters: int | None
    """The maximum number of iterations across all resumed training runs."""

    _variational_state: MCState
    """The variational state of this solver."""

    _nk_log: RuntimeLog
    """A NetKet runtime log for logging simulation data."""

    _diagonal_shift: float
    """The diagonal shift for the stochastic reconfiguration."""

    _n_iters: int
    """The number of iterations conducted in a simulation."""

    _vmc_driver: VMC
    """The VMC driver handling the optimisation loop."""

    _network: "FlaxModule"
    """The network used in the variational Ansatz."""

    _sampler: Sampler
    """The NetKet sampler being used by the solver."""

    _optimizer: GradientTransformation
    """The optax optimizer used by the Solver."""

    _preconditioner_solver: Any
    """The linear solver for stochastic reconfiguration"""

    _preconditioner: AbstractLinearPreconditioner | None = None
    """The preconditioner used for the natural gradient descent."""

    _diff_inv: bool
    """If True, the simulation was conducted using a diffeomorphism invariant state."""

    _n_samples: int
    """The number of samples used in the Monte Carlo process."""

    _sampler_kwargs: dict
    """The initialisation parameters of the current sampler."""

    _aux_sampler: Sampler
    """A copy of the sampler used by this solver."""

    _chunk_size: int | None
    """The chunk size used by the solver to compute expectation values and gradients."""

    _use_sr: bool
    """A flag to determine whether SR is to be used as a preconditioner."""

    _DEFAULT_OUTPUT_PATH: str = "Output Data"
    """The default output path for the simulation results."""

    def __init__(
        self,
        lqx: AbstractLqxInterface,
        output_path: str | None = None,
        auxiliary_path: str | None = None,
        *,
        clean_up: bool | None = False,
        seed: int | None = None,
    ):
        event(
            msg="SOLVER CREATED",
            tag="SOLVER:INIT",
            level=logging.INFO,
            output_path=output_path,
            seed=seed,
        )
        """
        Abstract base class for all neuraLQX solvers.

        A solver is responsible for orchestrating:
            - the model
            - the Hilbert space
            - the underlying graph and gauge structure,
            - the choice of sampler, optimiser and network,
            - the construction of a VMC  driver,
            - the training/minimisation loop,
            - logging, checkpointing and visualisation

        Concrete implementations should provide:
            - consistent, reproducible initialisation,
            - robust state export/import semantics,
            - support for MPI multi-rank runs,
            - an internal printer/logger abstraction,
            - meaningful error messages when methods are used out of order

        :param lqx: The LQG model implementing the :class:`~neuralqx.lqx.AbstractLqxInterface`.
          This object is expected to expose at least:
            - a Hilbert space interface implementing :class:`~neuralqx.hilbert.AbstractHilbertInterface`
            - a constraint accessible through `lqx.constraint`
            - a graph implementing :class:`~neuralqx.graph.AbstractGraph`.
            - a gauge group implementing :class:`~neuralqx.gauge_groups.AbstractGaugeGroup`.
        :param output_path: Base directory where the solver may write all outputs (logs, plots,
          serialised states, metadata). If ``None``, a concrete solver may choose a sensible default
        :param auxiliary_path: Extra path components used to group outputs (for example in HPC
          runs). Concrete implementations may interpret this as a seed folder, job-id folder, or any
          other auxiliary namespace.
        :param clean_up: If ``True``, a concrete solver may optionally remove stale/empty
          directories from previous runs in the specified ``output_path``
        """

        # local private attributes
        self._lqx = lqx
        self._hilbert = lqx.hilbert
        self._graph = lqx.graph
        self._gauge_group = lqx.gauge_group

        # setup a Rich console panel printer
        self._printer = NQXPrinter()

        # this solver assigns two key "run identity" values:
        #
        #   1) self._hash : a unique identifier used for output directory names
        #   2) self._seed : the user-facing solver seed used to seed MCState / samplers
        #
        # in an MPI run, the constructor is executed on every rank. Therefore, naively doing
        #
        #     self._hash = get_hash()
        #     self._seed = seed or secrets.randbits(30)
        #
        # will generate different values on different ranks
        #
        # - The solver's output path is constructed on every rank:
        #
        #       self._output_path = os.path.join(self.output_path, str(self.hash))
        #
        #   if each rank has a different hash, then each rank has a different notion of where
        #   outputs "should" live, and that is a correctness bug waiting to happen
        #
        # - the solver seed being different across ranks is confusing and can bite if we
        #   ever use self.seed outside of NetKet (e.g. metadata, deterministic branching, or
        #   naming conventions)
        #
        # in concrete solvers, we do:
        #
        #     MCState(..., seed=self.seed, sampler_seed=self.seed // 2)
        #
        # those are Python integers. NetKet's internal seeding utilities broadcast the integer
        # seed from the root rank and then split it per rank. That means NetKet will behave
        # deterministically even if non-root ranks hold "wrong" values
        #
        # however, relying on that implicit behaviour makes the solver itself inconsistent
        # broadcasting both hash and seed here gives us
        #
        # - a consistent output paths and metadata across ranks
        # - clear semantics: "the solver run has one hash and one seed"
        # - NetKet still does its own correct per-rank PRNG splitting downstream
        #
        # therefore,
        # - only the global master generates the hash and default seed (if the user didn't provide one)
        # - broadcast those values to every rank

        # generate hash on rank-0 only, then broadcast to all ranks
        # this guarantees that `self.hash` and any derived paths are identical everywhere
        _hash = get_hash() if _dist.is_global_master() else None
        self._hash = _dist.bcast(_hash, root=0)

        # determine the solver seed:
        # - if user provided `seed`, we respect it
        # - otherwise we generate a random seed once on rank-0 and broadcast it
        #
        # note: we keep the seed as a Python int to match NetKet's expected API and to keep
        # the behaviour identical in single-rank runs
        if seed is not None:
            _seed = int(seed)
        else:
            _seed = secrets.randbits(30) if _dist.is_global_master() else None
        self._seed = _dist.bcast(_seed, root=0)

        # check the output path
        if output_path is None:
            # default to the default path
            self._output_path = os.path.join(os.getcwd(), self._DEFAULT_OUTPUT_PATH)
        else:
            # user provided one, sanitise first
            if not isinstance(output_path, str) or not output_path:
                raise ValueError(
                    "You entered an invalid path for your output data directory. Please verify "
                    "that the path is correct and try again."
                )
            self._output_path = os.path.join(os.getcwd(), output_path)

        # check the auxiliary path
        if auxiliary_path is not None:
            self._aux_path = auxiliary_path
            self._output_path = os.path.join(self._output_path, str(auxiliary_path))

        # create the output data path
        # avoid rave conditions, only rank-0 does this
        if _dist.is_global_master():
            os.makedirs(
                os.path.join(self.output_path, str(self.hash)),
                exist_ok=True,
            )

        # make sure the path exists for everyone
        _dist.barrier()

        # remove any directories which are from previous simulations that have been stopped (e.g.
        # empty directories in output path)
        # this works only on rank-0 for MPI enabled
        if _dist.is_global_master():
            if clean_up:
                self._clean_up()
        _dist.barrier()

        # now we can set the final output directory including the simulation hash
        self._output_path = os.path.join(self.output_path, str(self.hash))

        # create an internal logger
        self._logger = Logger(self.hash, solver_seed=self.seed)

        # set up initialisation flags
        self._solved = False
        self._opt_flag = False
        self._sampler_flag = False
        self._network_flag = False
        self._driver_flag = False
        self._vstate_flag = False

        # internals
        self._registered_observables = None
        self._max_iters = None
        self._nk_log = DeferredRuntimeLog()

        # logging
        all_basis_states = (
            f"Min: {self.hilbert.hilbert.allowed_basis_states.start}, "
            f"Max: {self.hilbert.hilbert.allowed_basis_states.start + (self.hilbert.hilbert.allowed_basis_states.length - 1) * self.hilbert.hilbert.allowed_basis_states.step}, "
            f"Step: {self.hilbert.hilbert.allowed_basis_states.step}"
        )

        self._logger.log(
            [
                "Gravity model",
                "Gauge group",
                "Edges/Graph",
                "Number of vertices",
                "Number of edges",
                "Minimal loops",
                "Number of minimal loops",
                "Allowed degrees of freedom",
                "Hilbert space dimension",
                "Finite Hilbert space",
                "Indexable Hilbert space",
                "Gauge invariant Hilbert space",
            ],
            [
                self.lqx.model_name,
                self.gauge_group.name,
                self.graph.edges,
                self.graph.n_vertices,
                self.graph.n_edges,
                self.graph.minimal_loops() or None,
                len(self.graph.minimal_loops()),
                all_basis_states,
                humanize.number.scientific(self.hilbert.dimensions, precision=4),
                self.hilbert.is_finite,
                self.hilbert.is_indexable,
                self.hilbert.is_gauge_invariant,
            ],
        )

        # logging distributed runtime data (JAX process semantics)
        rt = _dist.runtime_info()
        local_gpu_count = len([d for d in jax.devices() if d.platform == "gpu"])
        gathered_gpu_counts = _dist.allgather(local_gpu_count)
        try:
            global_gpu_count = int(sum(int(x) for x in gathered_gpu_counts))
        except Exception:
            global_gpu_count = int(local_gpu_count)

        hostnames = _dist.allgather(platform.node())
        try:
            n_hosts = len(set(hostnames))
            processes_on_this_host = sum(1 for h in hostnames if h == platform.node())
        except Exception:
            n_hosts = 1
            processes_on_this_host = 1

        cpus_per_process = int(os.cpu_count() or 1)

        self._logger.log(
            [
                "Backend",
                "Distributed enabled",
                "Process index",
                "Process count",
                "Local process index",
                "Number of hosts",
                "Processes on this host",
                "CPUs per process",
                "Total number of CPUs",
                "Available GPUs",
                "JAX version",
                "Python implementation",
                "Python version",
            ],
            [
                rt.backend,
                bool(rt.size > 1),
                rt.rank,
                rt.size,
                rt.local_rank,
                n_hosts,
                processes_on_this_host,
                cpus_per_process,
                rt.size * cpus_per_process,
                str(global_gpu_count),
                getattr(jax, "__version__", "Unknown"),
                platform.python_implementation(),
                platform.python_version(),
            ],
        )

    def _clean_up(self):
        """Delete all empty directories in the output path, except the current run directory."""

        # rank-0 only
        if not _dist.is_global_master():
            return

        current_run_dir = os.path.join(self.output_path, str(self.hash))

        for root, dirs, files in os.walk(self.output_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)

                # never delete the current run directory
                if os.path.abspath(dir_path) == os.path.abspath(current_run_dir):
                    continue

                if not os.listdir(dir_path):
                    os.rmdir(dir_path)

    @property
    def seed(self) -> int:
        """Random seed used by the solver."""
        return self._seed

    @property
    def chunk_size(self) -> int | None:
        """Chunk size used by the variational state to compute expectation values and gradients."""
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, chunk_size: int) -> None:
        """Chunk size used by the variational state to compute expectation values and gradients."""
        self._chunk_size = chunk_size
        if self._driver_flag:
            self.variational_state.chunk_size = chunk_size

    @property
    def preconditioner(self):
        """The preconditioner used in the natural gradient descent."""
        return self._preconditioner

    @preconditioner.setter
    def preconditioner(self, preconditioner) -> None:
        """The preconditioner used in the natural gradient descent."""
        self._preconditioner = preconditioner

    @property
    def hash(self) -> str:
        """Unique hash/identifier for this solver run."""
        return self._hash

    @property
    def output_path(self) -> str:
        """Base output directory for this solver instance."""
        return self._output_path

    @property
    def auxiliary_path(self) -> str | None:
        """Optional auxiliary path component."""
        return self._aux_path

    @property
    def hilbert(self) -> AbstractHilbertInterface:
        """Hilbert space associated to the model being solved."""
        return self._hilbert

    @property
    def lqx(self) -> AbstractLqxInterface:
        """The active LQX model used by this solver."""
        return self._lqx

    @property
    def gauge_group(self) -> AbstractGaugeGroup | None:
        """Gauge group associated with the model."""
        return self._gauge_group

    @property
    def graph(self) -> AbstractGraph:
        """The graph the LQX model takes place on."""
        return self._graph

    @property
    def variational_state(self) -> MCState:
        """The active variational state."""
        if not self._vstate_flag:
            raise AttributeError("""
                The variational state is available only after running the simulation. If you really
                wish to use the variational state before running the simulation, try using
                    >>> your_solver_object.initialize_vmc()
                after setting the sampler, optimiser and the network.
                """)
        return self._variational_state

    @property
    def sampler(self) -> Sampler:
        """The active sampler instance."""
        if not self._sampler_flag:
            raise AttributeError("""
                The sampler is available only after running the simulation. If you really
                wish to use the sampler before running the simulation, try using
                    >>> Solver.initialize_vmc()
                after setting the sampler, optimiser and the network.
                """)
        return self._sampler

    @property
    def optimizer(self) -> GradientTransformation:
        """The active optimiser instance."""
        if not self._opt_flag:
            raise AttributeError("""
                The optimizer is available only after running the simulation. If you really
                wish to use the optimizer before running the simulation, try using
                    >>> Solver.initialize_vmc()
                after setting the sampler, optimiser and the network.
                """)
        return self._optimizer

    @property
    def network(self) -> "FlaxModule":
        """The active neural network being used in the variational Ansatz."""
        if not self._network_flag:
            raise AttributeError("""
                The neural network is available only after running the simulation. If you really
                wish to use the neural network before running the simulation, try using
                    >>> Solver.initialize_vmc()
                after setting the sampler, optimiser and the network.
                """)
        return self._network

    @property
    def vmc_driver(self) -> VMC:
        """The variational Monte Carlo driver that executes the optimisation loop."""
        if not self._driver_flag:
            raise AttributeError("""
                The VMC driver is available only after running the simulation. If you really
                wish to use the VMC driver before running the simulation, try using
                    >>> Solver.initialize_vmc()
                after setting the sampler, optimiser and the network.
                """)
        return self._vmc_driver

    @property
    def log(self) -> Any:
        """
        A runtime log object. This provides access to time series data such as:
            - energy means
            - variances
            - R-hat diagnostics
            - arbitrary user observables
        The interface may match NetKet's ``RuntimeLog`` or a project specific wrapper.
        """
        return self._nk_log

    @property
    def printer(self) -> Any:
        """User-facing output helper."""
        return self._printer

    @property
    def is_initialised(self) -> bool:
        """
        Whether the solver is ready to run a simulation.

        :return: True if the solver is ready to run, False otherwise.
        """
        return (
            self._opt_flag
            and self._sampler_flag
            and self._network_flag
            and self._driver_flag
        )

    @property
    def is_solved(self) -> bool:
        """
        Whether a simulation has been completed using this solver.

        :return: True if the solver has already completed a simulation, False otherwise.
        """
        return self._solved

    @property
    def diagonal_shift(self) -> float:
        return self._diagonal_shift

    @diagonal_shift.setter
    def diagonal_shift(self, val):
        self._diagonal_shift = val

    @property
    def n_iters(self) -> int:
        return self._n_iters

    @n_iters.setter
    def n_iters(self, val):
        self._n_iters = val

    @property
    def preconditioner_solver(self):
        """The linear solver used by stochastic reconfiguration in this solver."""
        if not self._opt_flag:
            raise AttributeError("""
                The linear solver for the preconditioner is available only after running the  simulation. If you really
                 wish to use the optimizer before running the simulation,  try using
                    >>> Solver.initialize_vmc()
                after setting the sampler, optimiser and the network.
                """)
        return self._preconditioner_solver

    @property
    def diffeomorphism_invariant_simulation(self):
        """Returns True if a diff. invariant state was used in the simulation."""
        if not self._opt_flag:
            raise AttributeError("""
                The flag is available only after running the simulation. Try using
                    >>> Solver.initialize_vmc()
                after setting the sampler, optimiser and the network, to access it.
                """)
        return self._diff_inv

    @abc.abstractmethod
    def run(
        self,
        n_iters: int,
        *,
        silent_print: bool,
        silent_plot: bool,
        timer: bool,
        live_monitoring: bool,
        callbacks: Callable[
            [int, dict, "AbstractVariationalDriver"], bool
        ] = lambda *x: True,
        observables: dict[str, "AbstractOperator"] | None = None,
        **kwargs,
    ) -> None:
        """
        Begin a simulation/optimisation run.

        This method should:
            1. Validate that the solver is initialised
            2. Ensure a VMC driver and runtime log exist
            3. Run the optimisation for ``n_iters`` iterations.
            4. Update internal state references (variational state, log, etc.)
            5. Optionally export a final checkpoint and display results.

        :param n_iters: Number of optimisation iterations to execute.
        :param silent_print: If True, suppress printing of final log output.
        :param silent_plot: if True, suppress the display of minimisation curve.
        :param timer: If True, enable timing/profiling integration if available.
        :param live_monitoring: if True, enable live monitoring of the optimisation
        :param callbacks: A list of Callbacks invoked after each iteration. The callable must
          accept ``(iteration_index, log_data_dict, driver)`` and return a boolean. If False is
          returned from any callback, the simulation terminates.
        :param observables: An optional dictionary of observables to be monitored at every step
          of the optimisation process.
        :param kwargs: Implementation-specific options.
        """

    @abc.abstractmethod
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
        **kwargs,
    ) -> None:
        """
        Continue a previously started simulation.

        Any concrete solver may support resuming from:
            - the last in-memory state, or
            - a serialised state loaded from ``state_path``

        :param state_path: A path to a previously exported variational state. If None, resume from
          the most recently available in-memory state.
        :param n_iters: Number of optimisation iterations to execute.
        :param silent_print: If True, suppress printing of final log output.
        :param silent_plot: if True, suppress the display of minimisation curve.
        :param timer: If True, enable timing/profiling integration if available.
        :param live_monitoring: if True, enable live monitoring of the optimisation
        :param callbacks: A list of Callbacks invoked after each iteration. The callable must
          accept ``(iteration_index, log_data_dict, driver)`` and return a boolean. If False is
          returned from any callback, the simulation terminates.
        :param observables: An optional dictionary of observables to be monitored at every step
          of the optimisation process.
        :param kwargs: Implementation-specific options.
        """

    @abc.abstractmethod
    def export_state(
        self,
        *,
        state: MCState | None = None,
        silent: bool = False,
        marker: str = "",
        **kwargs,
    ) -> str | None:
        """
        Export a serialised variational state to disk.

        A concrete implementation may choose to serialise:
            - only variational parameters,
            - a complete variational state including sampler configuration,
            - additional metadata needed for reproducibility

        :param state: Explicit state object to export. If None, export the solver's current
          variational state.
        :param silent: If True, suppress user-facing confirmation messages.
        :param marker: Optional string appended to the exported filename.
        :param kwargs: Implementation-specific options

        :return: The path of the exported file if the implementation chooses to return it,
          otherwise ``None``.
        """

    @abc.abstractmethod
    def import_state(
        self,
        state_path: str,
        *,
        force_load_mpi: bool = False,
    ) -> MCState:
        r"""
        Import and deserialise a variational state from disk.

        Note:
            It is recommended that implementations return a new state object rather than mutating
            the current one silently, unless explicitly documented.

        :param state_path: Path to a previously exported state.
        :param force_load_mpi: If True, force a best-effort reconstruction even if the MPI
          configuration differs from the environment used when saving. Concrete implementations
          must document the exact behaviour.

        :return: A reconstructed state object.
        """

    @abc.abstractmethod
    def plot_results(
        self,
        *,
        silent_plot: bool = False,
        with_inset: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot and save the main minimisation curve for the simulation.

        :param silent_plot: If True, suppress the display of the plot.
        :param with_inset: If True, show an inset zoom (if the concrete implementation supports it).
        :param kwargs: Implementation-specific plotting options.
        """

    @abc.abstractmethod
    def plot_observables(
        self,
        *observables: str,
        **kwargs,
    ) -> None:
        r"""
        Plot one or more logged observables.

        :param observables: Names/keys of observables as stored in :attr:`log`.
        :param kwargs: Implementation-specific plotting options.
        """

    @abc.abstractmethod
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
        Configure and attach a sampler to the solver instance.

        This method should not start sampling, it should only build and configure and eventually
        store the sampler object and any metadata needed later.

        :param sampler_type: Human-readable sampler identifier supported by neuraLQX
        :param number_of_chains: Total number of Monte Carlo chains to use.
        :param number_of_sweeps: Number of sweeps per sampling step.
        :param machine_power: The power used in the probability distribution  # dev: check
        :param reset_chains:  If True, the sampler resets the chains at every iteration.
        :param number_of_samples: Number of samples per chain.
        :param kwargs: Sampler-specific parameters
        """

    @abc.abstractmethod
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
        Configure and attach an optimiser to the solver.

        This method may also configure a stochastic reconfiguration (SR) preconditioner and its
        linear solver if the backend supports it.

        :param optimizer_type: Name of the optimizer supported by neuraLQX.
        :param preconditioner_solver: Name of the linear solver used inside SR, supported by
          neuraLQX
        :param preconditioner: The preconditioner object used in the natural gradient descent, if not specified SR is
          used by default.
        :param learning_rate: Constant learning rate. If None, a sensible default should be chosen.
        :param scheduler_type: Name of the learning rate schedule supported by neuraLQX.
        :param scheduler_parameters: Keyword arguments passed to the schedule builder.
        :param diagonal_shift: Optional diagonal shift for the SR-preconditioning.
        :param kwargs: Additional optimiser-specific configuration parameters.
        :param use_sr: if True, a stochastic reconfiguration (SR) preconditioner will be used
        """

    @abc.abstractmethod
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
        Configure and attach a neural network to the variational state stored in the solver.

        :param network: A neural network object (subclass of flax.linen.Module)
        :param diff_invariant: If True, the variational state should be diffeomorphism projected
          onto the chosen symmetry subspace.
        :param symmetries: A list of symmetry descriptors used for group projection.
        :param chunk_size: If specified, the expectation values and gradients are computed in a chunked manner where the
          largest chunk has size ``chunk_size``
        :param kwargs: Implementation-specific parameters
        """

    @abc.abstractmethod
    def expect(
        self,
        operator: Union[list["AbstractOperator"], "AbstractOperator"],
        *,
        n_samples: int | None = None,
        n_chains: int | None = None,
        use_same_variables: bool = True,
        use_same_sampler: bool = True,
        **kwargs,
    ) -> "Stats":
        """
        Compute the expectation value of an operator.

        This method should implement the ability to compute the expectation values using freshly
        generated samples, independent of the last optimisation step.

        Concrete solvers should implement two modes:
            - Use the existing :attr:`variational_state` samples if no other parameters are
              specified
            - Build a temporary state with the same sampler/network and draw new samples with the
              given number of samples divided by the given number of chains.

        :param operator: The (list of) operator/observable whose expectation value is requested.
        :param n_samples: If provided, number of samples per chain for a fresh sampling mode.
        :param n_chains: If provided, number of chains for a fresh sampling mode.
        :param use_same_variables: If True, reuse the trained network variables/parameters in a
          temporary state.
        :param use_same_sampler: if True, simply increase the number of samples in the current
          sampler and not draw from a fresh sampler.
        :param kwargs: Implementation-specific expectation options.
        :return: The expectation value of the specified (list of) operator
        """

    @abc.abstractmethod
    def initialize_vmc(self, *args, **kwargs) -> None:
        """
        Initialise the VMC driver and variational state.

        Concrete implementations should:
            - validate that sampler, optimiser and network are configured
            - construct the variational state
            - construct the VMC-like driver
            - attach any convenience hooks
            - update internal initialisation flags

        :param args: Implementation-specific options
        :param kwargs: Implementation-specific options
        """

    def __repr__(self):
        return "AbstractSolver()"
