#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Multi state solver utilities for training and monitoring multiple variational states.

This module provides serialization helpers for `MultiMCState` and a `Solver` subclass
that trains several `MCState` instances jointly using `MultiStateVMC`. It supports
per state expectation evaluation, multi state export and import, per state logging of
final constraint estimates, and plotting of per state optimisation curves when present
in the runtime log.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime

from typing import Any
from typing import List
from typing import Optional
from typing import Union
from typing import Sequence
from typing import TYPE_CHECKING

import humanize
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from neuralqx.solver.solver import Solver
from neuralqx.solver.solver import reject_outliers

from neuralqx.vqs import MCState

from neuralqx.utils import distributed as _dist
from neuralqx.utils.serialization import load_from_file
from neuralqx.utils.serialization import save_to_file
from neuralqx.utils.parsing import log_module_attributes
from neuralqx.nn.projectors.group_projector import wrap_model

from neuralqx.vqs.mc.mc_state.state import serialize_MCState
from neuralqx.vqs.mc.mc_state.state import deserialize_MCState

from neuralqx.experimental.vqs.mc.mc_state.mtmh_state import MultiMCState
from neuralqx.experimental.driver.mvmc import MultiStateVMC

from netket.optimizer import SR
from netket.utils import is_probably_holomorphic

if TYPE_CHECKING:
    from flax.linen import Module as FlaxModule
    from netket.operator import AbstractOperator
    from netket.experimental.observable import AbstractObservable
    from netket.stats.mc_stats import Stats


def serialize_MultiMCState(vstate: MultiMCState) -> dict:
    """
    Serialize a :class:`~neuralqx.experimental.vqs.MultiMCState` into a plain Python dictionary.

    The resulting dictionary is suitable for writing to disk with the neuraLQX
    serialization utilities. Each contained :class:`~neuralqx.experimental.vqs.MultiMCState` is serialized using
    ``serialize_MCState``.

    :param vstate: Multi state variational state to serialize.
    :return: Dictionary representation of the multi state, including per state payloads.
    """

    return {
        "kind": "MultiMCState",
        "n_states": vstate.n_states,
        "states": [serialize_MCState(s) for s in vstate.states],
        # optional top-level distributed info, per-state dict already has metadata too
        "distributed_info": _dist.get_distributed_info_dict(),
    }


def deserialize_MultiMCState(
    template: MultiMCState,
    state_dict: dict,
    *,
    force_load_mpi: bool = False,
) -> MultiMCState:
    """
    Deserialize a :class:`~neuralqx.experimental.vqs.MultiMCState` from a dictionary into a new
    :class:`~neuralqx.experimental.vqs.MultiMCState` instance.

    The function uses an existing `template` multi state to provide structure and
    implementation details required by `deserialize_MCState`. For backward compatibility,
    a dictionary that represents a single `MCState` is accepted and wrapped into a
    single state :class:`~neuralqx.experimental.vqs.MultiMCState`.

    :param template: Existing multi state used as a template for reconstructing each state.
    :param state_dict: Dictionary payload previously produced by serialization utilities.
    :param force_load_mpi: If True, attempt to load MPI related fields even if they differ
        from the current run configuration.
    :return: Newly reconstructed :class`~neuralqx.experimental.vqs.MultiMCState`.
    :raises ValueError: If the number of saved states does not match `template.n_states`.
    """

    # Backward compatibility: if a single MCState dict was saved, wrap it as one state
    if "states" not in state_dict:
        new_single = deserialize_MCState(
            template.states[0], state_dict, force_load_mpi=force_load_mpi
        )
        return MultiMCState([new_single])

    saved_states = state_dict["states"]
    if len(saved_states) != template.n_states:
        raise ValueError(
            f"Loaded MultiMCState has n_states={len(saved_states)} but current template has "
            f"n_states={template.n_states}. Make sure you set the same number of networks before loading."
        )

    new_states: List[MCState] = []
    for i, (tmpl, sd) in enumerate(zip(template.states, saved_states)):
        new_states.append(deserialize_MCState(tmpl, sd, force_load_mpi=force_load_mpi))

    return MultiMCState(new_states)


class MultiSolver(Solver):
    """
    Solver subclass that trains multiple variational states jointly.

    This solver is a drop in variant of :class:`~neuralqx.solver.Solver` that runs
    :class:`~neuralqx.experimental.driver.MultiStateVMC` with a
    :class:`~neuralqx.experimental.vqs.MultiMCState` backend. It supports configuring multiple networks, building one
    :class:`~neuralqx.vqs.MCState` per network during VMC initialization, evaluating expectations per state
    or for all states, exporting and importing multi state checkpoints, and logging
    final constraint results both aggregated and per state.

    :return: None.
    """

    _lambda_ortho: float
    """The orthogonalization factor between the networks."""

    @property
    def lambda_ortho(self) -> float:
        """
        Return the orthogonalization strength used between the networks.

        :return: Orthogonalization factor used by the multi state driver.
        """

        return self._lambda_ortho

    def set_network(
        self,
        network: Union["FlaxModule", Sequence["FlaxModule"]],
        *,
        diff_invariant: bool = False,
        symmetries: Optional[Sequence[Any]] = None,
        lambda_ortho: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Configure one or more networks for a multi state simulation.

        A single model or a sequence of models can be provided. The solver stores the models
        as a list and logs both an aggregate network summary and per state network type
        entries. When diffeomorphism invariance is enabled, each model is wrapped using the
        configured symmetry information.

        :param network: A single Flax module or a sequence of Flax modules, one per state.
        :param diff_invariant: If True, enable diffeomorphism invariant wrapping of the models.
        :param symmetries: Optional symmetry data forwarded to the model wrapper.
        :param lambda_ortho: Optional override for the orthogonalization strength used during
            multi state optimisation.
        :param kwargs: Additional keyword arguments accepted for forward compatibility.
        :return: None.
        """

        # erase old network section if re-setting
        if getattr(self, "_network_flag", False):
            self._logger.erase_parent_content("Network Configs")
            self._network_flag = False

        # normalise to list
        if isinstance(network, (list, tuple)):
            networks = list(network)
        else:
            networks = [network]

        # optionally wrap each model for diffeo projection
        self._diff_inv = diff_invariant
        if self.diffeomorphism_invariant_simulation is True:
            networks = [
                wrap_model(
                    net,
                    symmetries=symmetries,
                    graph=self.graph,
                    characters=None,
                )
                for net in networks
            ]

        self._network = networks
        self._gnetwork = networks[0]

        # logger, aggregate + per-state fields (unique keys!)
        net_names = [
            (net.__name__ if hasattr(net, "__name__") else net.__class__.__name__)
            for net in networks
        ]

        # existing field: "Network type"
        self._logger.log(
            "Network type", f"MultiNetwork(N={len(networks)}): " + ", ".join(net_names)
        )

        # add and log number of networks (unique key)
        self._logger.add_field("Network Configs", "Number of networks")
        self._logger.log("Number of networks", len(networks))

        # per-state network types
        for i, nm in enumerate(net_names):
            k = f"State[{i}] Network type"
            self._logger.add_field("Network Configs", k)
            self._logger.log(k, nm)

        # log module attributes ONCE (to avoid collisions/overwrites)
        # if we really need per-state attrs, we can prefix keys, but that requires changing log_module_attributes
        try:
            log_module_attributes(self, networks[0])
        except Exception:
            pass

        if lambda_ortho is not None:
            self._lambda_ortho = lambda_ortho
        else:
            self._lambda_ortho = 1.0

        # mark flag + inform user
        self._network_flag = True
        self.printer.print(
            f"{'Diffeomorphism Invariant ' if self.diffeomorphism_invariant_simulation else ''}"
            f"MultiNetwork with {len(networks)} states initialized."
        )

    def expect(
        self,
        operator: Union[
            List[Union["AbstractOperator", "AbstractObservable"]],
            Union["AbstractOperator", "AbstractObservable"],
        ],
        *,
        state_idx: Optional[int] = None,
        n_samples: Optional[int] = None,
        n_chains: Optional[int] = None,
        use_same_variables: bool = True,
        use_same_sampler: bool = True,
        **kwargs,
    ) -> Union["Stats", List["Stats"]]:
        """
        Compute expectation values in a multi state aware way.

        If `state_idx` is None, compute expectations for all states and return a list of Stats.
        If `state_idx` is an integer, compute the expectation for that single state and return
        a single Stats object. An optional `n_samples` can be provided to temporarily override
        the number of samples used by the selected state or states, which discards and
        regenerates Monte Carlo samples.

        :param operator: Operator or observable, or a list of them, to evaluate.
        :param state_idx: Optional state index to select a single state. If None, evaluate all
            states.
        :param n_samples: Optional temporary sample count override for the evaluation.
        :param n_chains: Not supported in this method.
        :param use_same_variables: Accepted for API compatibility and ignored.
        :param use_same_sampler: Accepted for API compatibility and ignored.
        :param kwargs: Additional keyword arguments accepted for API compatibility and ignored.
        :return: A Stats object if `state_idx` is provided, otherwise a list of Stats.
        :raises RuntimeError: If the variational state has not been initialised.
        :raises NotImplementedError: If `n_chains` is provided.
        :raises IndexError: If `state_idx` is out of range for the current number of states.
        """

        if not getattr(self, "_vstate_flag", False):
            raise RuntimeError(
                "Cannot compute expectation value: variational state is not initialised.\n\n"
                "Try calling initialize_vmc() after setting the network, sampler and optimizer."
            )

        if n_chains is not None:
            raise NotImplementedError(
                "Expectation with explicit `n_chains` is not yet implemented."
            )

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

        # Helper: run expect on a specific state (with optional temporary n_samples override)
        def _expect_one(idx: int):
            st = self.variational_state.states[idx]
            if n_samples is None:
                return st.expect(operator)

            warnings.warn(
                "Calling `MultiSolver.expect(..., n_samples=...)` temporarily overrides the "
                "number of samples in the selected variational state(s). This will discard and "
                "regenerate existing Monte Carlo samples.",
                RuntimeWarning,
                stacklevel=2,
            )

            original = getattr(st, "n_samples", None)
            try:
                st.n_samples = int(n_samples)
                return st.expect(operator)
            finally:
                if original is not None:
                    st.n_samples = original

        # single state
        if state_idx is not None:
            if state_idx < 0 or state_idx >= self.variational_state.n_states:
                raise IndexError(
                    f"state_idx={state_idx} out of range for n_states={self.variational_state.n_states}"
                )
            return _expect_one(state_idx)

        # all states
        return [_expect_one(i) for i in range(self.variational_state.n_states)]

    def initialize_vmc(self, *args, **kwargs) -> None:
        """
        Initialise the multi state variational state and VMC driver.

        This method requires that sampler, optimizer, and networks have already been set. It
        builds one `MCState` per network with distinct seeds, collects holomorphicity flags,
        constructs a `MultiMCState`, logs parameter counts and ratios, and then creates a
        `MultiStateVMC` driver with a per state SR preconditioner.

        :param args: Positional arguments accepted for compatibility.
        :param kwargs: Keyword arguments. If present, `lambda_ortho` overrides the stored
            orthogonalization factor.
        :return: None.
        :raises PermissionError: If sampler, optimizer, or networks are not configured.
        :raises ValueError: If no networks are available for initialization.
        """

        if not self._sampler_flag or not self._opt_flag or not self._network_flag:
            raise PermissionError(f"""
                You cannot run the VMC driver without setting the sampler, optimizer as well as 
                the network!

                Try using set_sampler(), set_optimizer() and set_network() first.
                """)

        # normalize networks
        networks = self.network
        if not isinstance(networks, (list, tuple)):
            networks = [networks]
        if len(networks) == 0:
            raise ValueError("No networks were provided to MultiSolver.set_network().")

        # build one MCState per network
        states: List[MCState] = []
        holo_flags: List[bool] = []

        for i, net in enumerate(networks):
            st = MCState(
                self.sampler,
                net,
                n_samples=self._n_samples,
                is_group_averaged=self.diffeomorphism_invariant_simulation,
                seed=int(self.seed + 1337 * i),
                sampler_seed=int(self.seed // 2 + 7331 * i),
            )
            states.append(st)

            try:
                is_holo = is_probably_holomorphic(
                    st._apply_fun,
                    st.parameters,
                    st.samples,
                    st.model_state,
                )
            except Exception:
                is_holo = False
            holo_flags.append(bool(is_holo))

        self._variational_state = MultiMCState(states)

        per_state_params = [int(s.n_parameters) for s in states]
        total_params = int(sum(per_state_params))

        # existing field: total params
        self._logger.log("Number of network parameters", total_params)

        # add per-state param fields
        for i, npar in enumerate(per_state_params):
            k = f"State[{i}] Number of network parameters"
            self._logger.add_field("Network Configs", k)
            self._logger.log(k, npar)

        # compute dim_H
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

        # total ratio
        ratio_total = 100.0 * total_params / float(dim_H)
        self._logger.add_field(
            "Network Configs", "Params(network_total) / dim(Hilbert)"
        )
        self._logger.log(
            "Params(network_total) / dim(Hilbert)",
            f"{humanize.number.scientific(ratio_total, precision=4)} %",
        )

        # per-state ratios
        for i, npar in enumerate(per_state_params):
            k = f"State[{i}] Params(network) / dim(Hilbert)"
            self._logger.add_field("Network Configs", k)
            self._logger.log(
                k,
                f"{humanize.number.scientific(100.0 * npar / float(dim_H), precision=4)} %",
            )

        self._vstate_flag = True

        # pick SR holomorphic conservatively, True only if ALL states holomorphic
        # dev: SR for each state means SR can be differently holomorphic for each state, this needs a fix
        holomorphic = bool(all(holo_flags))

        # build MultiStateVMC driver
        lambda_ortho = float(kwargs.get("lambda_ortho", self.lambda_ortho))

        self._vmc_driver = MultiStateVMC(
            variational_state=self._variational_state,
            hamiltonian=self.lqx.constraint,
            optimizer=self.optimizer,
            preconditioner=SR(
                diag_shift=self.diagonal_shift,
                holomorphic=holomorphic,
                solver=self.preconditioner_solver,
            ),
            lambda_ortho=lambda_ortho,
        )

        self._driver_flag = True

    def _final_constraint_estimate(self) -> None:
        """
        Log final constraint estimates for a multi state variational state.

        This override handles `MultiMCState.expect` returning a list of Stats. It stores the
        per state stats and per state scalar means, logs an aggregated entry for backward
        compatibility, and also logs per state results under unique keys. If an exact
        diagonalisation reference energy is available, it is logged as well.

        :return: None.
        """

        try:
            stats_list = self.variational_state.expect(self.lqx.constraint)
        except Exception:
            return

        # ensure list-like
        if not isinstance(stats_list, list):
            stats_list = [stats_list]

        self.final_constraint_stats = stats_list
        self.final_constraint_mean = [
            self._coerce_mean_value(s.mean, sigma=getattr(s, "Sigma", None))
            for s in stats_list
        ]

        # Backward compatible: log aggregated "Network result" as the list
        if hasattr(self, "_logger") and self._logger is not None:
            try:
                self._logger.log("Network result", stats_list)
            except Exception:
                pass

            # per-state logs into unique keys under Optimization Results
            for i, st in enumerate(stats_list):
                k = f"State[{i}] Network result"
                try:
                    self._logger.add_field("Optimization Results", k)
                except Exception:
                    # field might already exist in some rerun scenario
                    pass
                try:
                    self._logger.log(k, st)
                except Exception:
                    pass

            # exact diag
            if self.lqx.has_ground_energy:
                try:
                    self._logger.log(
                        "Exact diagonalization result", self.lqx.ground_energy[0]
                    )
                except Exception:
                    pass

    def export_state(
        self,
        *,
        state: Optional[MultiMCState] = None,
        silent: bool = False,
        marker: str = "",
        **kwargs,
    ) -> Optional[str]:
        """
        Export a `MultiMCState` checkpoint to disk in an MPI safe manner.

        The method serializes each contained `MCState` and writes a single multi state payload
        to the solver output directory on the global MPI master rank. Other ranks participate
        via barriers and perform no file IO.

        :param state: Optional `MultiMCState` to export. If None, exports the current solver state.
        :param silent: If True, suppress user facing printing on successful export.
        :param marker: Optional string appended to the filename for easier identification.
        :param kwargs: Additional keyword arguments accepted for forward compatibility.
        :return: None.
        :raises TypeError: If the provided or current state is not a `MultiMCState`.
        """

        vstate = state or self._variational_state
        if not isinstance(vstate, MultiMCState):
            raise TypeError("MultiSolver.export_state expects a MultiMCState.")

        serialised_state = serialize_MultiMCState(vstate)

        _dist.barrier()

        if _dist.is_global_master():
            os.makedirs(self.output_path, exist_ok=True)
            if marker:
                marker = "_" + marker
            datetime_str = datetime.now().strftime("%d%m%Y_%H%M%S")
            filename = os.path.join(
                self.output_path,
                f"{self.hash}_SerialisedMultiState_{datetime_str}{marker}.mpack",
            )
            save_to_file(serialised_state, filename)

            if not silent and self.printer is not None:
                self.printer.print("MultiState serialised to disk.")

        _dist.barrier()
        return None

    def import_state(
        self,
        state_path: str,
        *,
        force_load_mpi: bool = False,
    ) -> MultiMCState:
        """
        Import a `MultiMCState` checkpoint from disk.

        The payload is loaded on the global MPI master rank and broadcast to all ranks. This
        method requires that a template `MultiMCState` already exists, which is created by
        calling `initialize_vmc` after configuring networks, sampler, and optimiser.

        :param state_path: Path to the serialized checkpoint file.
        :param force_load_mpi: If True, attempt to load MPI related fields even if they differ
            from the current run configuration.
        :return: Reconstructed `MultiMCState` instance.
        :raises RuntimeError: If no `MultiMCState` template exists in the solver.
        """

        _dist.barrier()

        if _dist.is_global_master():
            nvs = load_from_file(state_path, raw=True)
        else:
            nvs = None

        nvs = _dist.mpi_bcast(nvs, root=0)

        if not isinstance(self.variational_state, MultiMCState):
            raise RuntimeError(
                "MultiSolver.import_state requires an existing MultiMCState template. "
                "Call initialize_vmc() after setting networks/sampler/optimizer before loading."
            )

        return deserialize_MultiMCState(
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
        Plot and save the minimisation curves for the simulation.

        This method exports a graph image when possible, then plots the objective curve with
        error bars. If multi state series are present in the runtime log, it plots one curve
        per state. Otherwise it falls back to a single objective curve. An optional inset view
        can be added for the final iterations, and the runtime log is serialized alongside the
        saved plot image.

        :param silent_plot: If True, do not display the plot interactively.
        :param with_inset: If True, include an inset showing the final portion of the curves.
        :param kwargs: Additional keyword arguments accepted for forward compatibility.
        :return: None.
        """

        # dev: this is still WIP...

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

        def _is_stats_series(obj) -> bool:
            # RuntimeLog stores series with .iters, .Mean, .Sigma etc
            return (
                hasattr(obj, "iters") and hasattr(obj, "Mean") and hasattr(obj, "Sigma")
            )

        # Prefer multi-state keys Constraint/s{i}
        constraint_series = []
        # data_nn behaves like a dict
        for k in list(data_nn.keys()):
            if isinstance(k, str) and k.startswith("Constraint ("):
                try:
                    s = data_nn[k]
                    if _is_stats_series(s):
                        constraint_series.append((k, s))
                except Exception:
                    pass

        # sort by state index if possible (Constraint/s0, Constraint/s1, ...)
        def _state_index(name: str) -> int:
            # name like "Constraint/s3"
            try:
                suffix = name.split("(")[-1]
                if suffix.startswith("s"):
                    return int(suffix[1:])
            except Exception:
                pass
            return 10**9

        constraint_series.sort(key=lambda kv: _state_index(kv[0]))

        # fallback to single-state "Constraint"
        if (
            not constraint_series
            and "Constraint" in data_nn
            and _is_stats_series(data_nn["Constraint"])
        ):
            constraint_series = [("Constraint", data_nn["Constraint"])]

        if not constraint_series:
            self.printer.print("No 'Constraint' series found in log, skipping plot.")
            return

        for name, series in constraint_series:
            # label nicely: "State 0", "State 1", ...
            if name == "Constraint":
                lbl = "neuraLQX"
            else:
                idx = _state_index(name)
                lbl = f"State {idx}" if idx != 10**9 else name

            ax.errorbar(
                series.iters,
                series.Mean,
                yerr=series.Sigma,
                label=lbl,
                alpha=0.6,
                marker="o",
                markersize=3.5,
                capsize=5,
                linestyle="",
                elinewidth=1.5,
            )

        # customise the legend
        ax.legend(fontsize="x-large")

        # customise the axes
        ax.set_xlabel("Iteration", fontsize="xx-large")
        ax.set_ylabel(r"$\langle \hat{C}\rangle$", fontsize="xx-large")
        ax.tick_params(axis="both", which="both", labelsize="x-large")

        # add grid
        ax.grid(True, linestyle="--", alpha=0.7)

        if with_inset:
            inset_position_100dpi = [170, -90, 850, 700]
            inset_position = [ip * dpi_ratio for ip in inset_position_100dpi]

            max_x = self._max_iters
            min_x = max_x - 20

            ax_ins = inset_axes(
                ax, width="30%", height="30%", bbox_to_anchor=inset_position
            )

            # plot all state curves in inset
            for name, series in constraint_series:
                if name == "Constraint":
                    lbl = "NN"
                else:
                    idx = _state_index(name)
                    lbl = f"State {idx}" if idx != 10**9 else name

                ax_ins.errorbar(
                    series.iters,
                    series.Mean,
                    yerr=series.Sigma,
                    label=lbl,
                    alpha=0.8,
                    marker="o",
                    markersize=5,
                    capsize=5,
                    linestyle="",
                    elinewidth=1.2,
                )

            # ED line in inset
            if has_ge:
                ax_ins.hlines(
                    [ge],
                    xmin=min_x,
                    xmax=max_x,
                    color="#E25C47",
                    label="ED Result",
                    linewidth=2,
                )

            # compute y-bounds using all states in [min_x:max_x]
            all_vals = []
            for _, series in constraint_series:
                try:
                    # series supports slicing like series[min_x:max_x]
                    window_vals = np.asarray(series[min_x:max_x].Mean)
                    if window_vals.size:
                        all_vals.append(reject_outliers(window_vals))
                except Exception:
                    pass

            if all_vals:
                cleaned = np.concatenate(all_vals)
                data_min = float(np.min(cleaned))
                data_max = float(np.max(cleaned))
            else:
                # fallback: use first series
                cleaned = reject_outliers(np.asarray(constraint_series[0][1].Mean))
                data_min = float(np.min(cleaned))
                data_max = float(np.max(cleaned))

            if has_ge:
                min_y = min(data_min, float(ge)) - 0.05
                max_y = max(data_max, float(ge)) + 0.15
            else:
                min_y = data_min - 0.05
                max_y = data_max + 0.15

            ax_ins.set_xlim(min_x, max_x)
            ax_ins.set_ylim(min_y, max_y)

            mark_inset(ax, ax_ins, loc1=3, loc2=1, fc="none", ec="0.6", linestyle="--")

            ax_ins.grid(True, linestyle="--", alpha=0.7)
            ax_ins.minorticks_on()

        ax.minorticks_on()

        # save image to disc
        plt.savefig(image_filename, dpi=dpi)

        # serialise the simulation data to disc
        self.log.serialize(
            os.path.join(self.output_path, f"{self.hash}_LogData_{dt_str}")
        )

        # display plot if requested
        if (not silent_plot) and (_dist.n_nodes == 1):
            plt.show()
        else:
            plt.close()

    # TODO: do something about this
    def __repr__(self) -> str:
        return "MultiSolver()"
