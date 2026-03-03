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

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Any, List, Optional, Sequence, TYPE_CHECKING, Union

import humanize
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from flax import linen as nn

from neuralqx.solver.solver import Solver
from neuralqx.solver.solver import reject_outliers

from neuralqx.vqs import MCState

from neuralqx.utils import mpi as _mpi
from neuralqx.utils.serialization import load_from_file
from neuralqx.utils.serialization import save_to_file
from neuralqx.utils.parsing import log_module_attributes
from neuralqx.nn.projectors.group_projector import wrap_model

from neuralqx.vqs.mc.mc_state.state import serialize_MCState
from neuralqx.vqs.mc.mc_state.state import deserialize_MCState

try:
    from neuralqx.experimental.vqs.mc.mc_state.stmh_state import (
        STMultiMCState,
    )
except Exception:  # pragma: no cover
    from neuralqx.experimental.vqs.mc.mc_state import STMultiMCState

try:
    from neuralqx.experimental.driver.stmh_multi_vmc import SingleTrunkMultiHeadVMC
except Exception:  # pragma: no cover
    from neuralqx.experimental.driver import SingleTrunkMultiHeadVMC

from netket.optimizer import SR
from netket.utils import is_probably_holomorphic
from netket.utils.mpi import mpi_bcast

if TYPE_CHECKING:
    from flax.linen import Module as FlaxModule
    from netket.operator import AbstractOperator
    from netket.experimental.observable import AbstractObservable
    from netket.stats.mc_stats import Stats


class _LocalSTMHHeadView(nn.Module):
    """
    Lightweight compatibility wrapper selecting one head from a shared ST-MH model.

    The wrapped model must support ``model(x, head=int)`` and return a scalar output per sample.
    This mirrors the ``STMHHeadView`` interface so the solver can work even if the wrapper module
    is not imported from a specific package path.
    """

    base: nn.Module
    head: int

    @nn.compact
    def __call__(self, x):
        try:
            return self.base(x, head=int(self.head))
        except TypeError as exc:
            raise TypeError(
                "The provided ST-MH base network does not appear to support a `head=` keyword. "
                "Pass explicit `head_models=` to `set_network(...)` or use the STMH wrapper model."
            ) from exc


def serialize_SharedParamsMultiMCState(vstate: SharedParamsMultiMCState) -> dict:
    """
    Serialize a :class:`SharedParamsMultiMCState` into a plain Python dictionary.

    Each head ``MCState`` is serialized using ``serialize_MCState``. Parameters are shared, but we
    intentionally keep the per-head payload format because it is robust and backward-compatible.
    """

    return {
        "kind": "SharedParamsMultiMCState",
        "n_states": vstate.n_states,
        "canonical_state": int(getattr(vstate, "canonical_state", 0)),
        "sync_model_state": bool(getattr(vstate, "_sync_model_state", False)),
        "states": [serialize_MCState(s) for s in vstate.states],
        "mpi_info": (
            _mpi.get_mpi_info_dict() if hasattr(_mpi, "get_mpi_info_dict") else None
        ),
    }


def deserialize_SharedParamsMultiMCState(
    template: SharedParamsMultiMCState,
    state_dict: dict,
    *,
    force_load_mpi: bool = False,
) -> SharedParamsMultiMCState:
    """
    Deserialize a shared-parameter multi-head state from a checkpoint dictionary.

    :param template: Existing ``SharedParamsMultiMCState`` used to reconstruct each contained ``MCState``.
    :param state_dict: Raw dict loaded from disk.
    :param force_load_mpi: Forwarded to ``deserialize_MCState``.
    """

    # Backward compatibility: allow a single MCState payload.
    if "states" not in state_dict:
        new_single = deserialize_MCState(
            template.states[0], state_dict, force_load_mpi=force_load_mpi
        )
        return SharedParamsMultiMCState(
            [new_single],
            canonical_state=int(getattr(template, "canonical_state", 0)),
            sync_model_state=bool(getattr(template, "_sync_model_state", False)),
        )

    saved_states = state_dict["states"]
    if len(saved_states) != template.n_states:
        raise ValueError(
            f"Loaded SharedParamsMultiMCState has n_states={len(saved_states)} but current template "
            f"has n_states={template.n_states}. Make sure you set the same number of heads before loading."
        )

    new_states: List[MCState] = []
    for tmpl, sd in zip(template.states, saved_states):
        new_states.append(deserialize_MCState(tmpl, sd, force_load_mpi=force_load_mpi))

    out = SharedParamsMultiMCState(
        new_states,
        canonical_state=int(
            state_dict.get("canonical_state", getattr(template, "canonical_state", 0))
        ),
        sync_model_state=bool(
            state_dict.get(
                "sync_model_state", getattr(template, "_sync_model_state", False)
            )
        ),
    )
    out.broadcast_from_canonical()
    return out


class STMultiSolver(Solver):
    """
    Solver for the single-trunk multi-head (ST-MH) variational ansatz.

    Compared to ``MultiSolver`` (MT-MH / independent networks), this solver builds one shared
    multi-head Flax model and exposes one ``MCState`` per head using head-selector wrappers. The
    variational state is a :class:`SharedParamsMultiMCState`, and optimization is performed by
    :class:`SingleTrunkMultiHeadVMC`, which aggregates energy and orthogonality gradients into a
    single shared parameter update.
    """

    _lambda_ortho: float

    @property
    def lambda_ortho(self) -> float:
        return self._lambda_ortho

    @property
    def n_heads(self) -> int:
        return int(getattr(self, "_n_heads", 0))

    def _build_head_models(
        self, base_model: "FlaxModule", n_heads: int
    ) -> list["FlaxModule"]:
        # Self-contained head wrappers; avoids depending on where STMHHeadView is exported.
        return [
            _LocalSTMHHeadView(base=base_model, head=i) for i in range(int(n_heads))
        ]

    def set_network(
        self,
        network: "FlaxModule",
        *,
        n_heads: Optional[int] = None,
        head_models: Optional[Sequence["FlaxModule"]] = None,
        diff_invariant: bool = False,
        symmetries: Optional[Sequence[Any]] = None,
        lambda_ortho: Optional[float] = None,
        canonical_state: int = 0,
        sync_model_state: bool = False,
        **kwargs,
    ) -> None:
        """
        Configure a shared ST-MH network and (optionally) explicit per-head view models.

        :param network: Shared ST-MH base model (typically your wrapper that supports ``head=...``).
        :param n_heads: Number of heads. If omitted, we try to infer it from ``network.n_heads``.
        :param head_models:
            Optional explicit scalar-output head models, one per head. If provided, these are used
            directly and ``network`` is treated as the shared base model only for logging.
        :param diff_invariant, symmetries:
            Same semantics as in ``MultiSolver``. For ST-MH, diffeo wrapping is applied to the
            *head models* during ``initialize_vmc``.
        :param lambda_ortho: Orthogonality/fidelity penalty strength.
        :param canonical_state: Forwarded to ``SharedParamsMultiMCState``.
        :param sync_model_state: Forwarded to ``SharedParamsMultiMCState``.
        """

        if getattr(self, "_network_flag", False):
            self._logger.erase_parent_content("Network Configs")
            self._network_flag = False

        self._diff_inv = diff_invariant
        self._symmetries = symmetries
        self._stmh_base_network = network
        self._head_models_user = list(head_models) if head_models is not None else None

        if self._head_models_user is not None:
            inferred_n_heads = len(self._head_models_user)
            if n_heads is not None and int(n_heads) != inferred_n_heads:
                raise ValueError(
                    f"n_heads={n_heads} does not match len(head_models)={inferred_n_heads}."
                )
            self._n_heads = inferred_n_heads
        else:
            if n_heads is None:
                n_heads = getattr(network, "n_heads", None)
            if n_heads is None:
                raise ValueError(
                    "For ST-MH, `set_network(...)` requires `n_heads` unless the model exposes an "
                    "`n_heads` attribute."
                )
            self._n_heads = int(n_heads)
            if self._n_heads <= 0:
                raise ValueError("n_heads must be >= 1.")

        self._stmh_canonical_state = int(canonical_state)
        self._stmh_sync_model_state = bool(sync_model_state)

        # Keep compatibility with base Solver expectations.
        self._network = network
        self._gnetwork = network

        base_name = (
            network.__name__
            if hasattr(network, "__name__")
            else network.__class__.__name__
        )
        self._logger.log(
            "Network type", f"SingleTrunkMultiHead(N={self._n_heads}): {base_name}"
        )

        self._logger.add_field("Network Configs", "Number of heads")
        self._logger.log("Number of heads", self._n_heads)

        if self._head_models_user is not None:
            for i, hm in enumerate(self._head_models_user):
                nm = hm.__name__ if hasattr(hm, "__name__") else hm.__class__.__name__
                k = f"Head[{i}] Network view type"
                self._logger.add_field("Network Configs", k)
                self._logger.log(k, nm)

        try:
            log_module_attributes(self, network)
        except Exception:
            pass

        self._lambda_ortho = float(lambda_ortho) if lambda_ortho is not None else 1.0
        self._network_flag = True

        self.printer.print(
            f"{'Diffeomorphism Invariant ' if self.diffeomorphism_invariant_simulation else ''}"
            f"ST-MH network with {self._n_heads} heads initialized."
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
        """Multi-head aware expectation helper mirroring ``MultiSolver.expect``."""

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

        def _expect_one(idx: int):
            st = self.variational_state.states[idx]
            if n_samples is None:
                return st.expect(operator)

            warnings.warn(
                "Calling `STMultiSolver.expect(..., n_samples=...)` temporarily overrides the "
                "number of samples in the selected head state(s). This will discard and "
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

        if state_idx is not None:
            if state_idx < 0 or state_idx >= self.variational_state.n_states:
                raise IndexError(
                    f"state_idx={state_idx} out of range for n_states={self.variational_state.n_states}"
                )
            return _expect_one(state_idx)

        return [_expect_one(i) for i in range(self.variational_state.n_states)]

    def initialize_vmc(self, *args, **kwargs) -> None:
        """
        Initialise the shared-parameter ST-MH variational state and driver.

        Builds one ``MCState`` per head (head-selector view), then wraps them in
        ``SharedParamsMultiMCState`` and constructs ``SingleTrunkMultiHeadVMC``.
        """

        if not self._sampler_flag or not self._opt_flag or not self._network_flag:
            raise PermissionError("""
                You cannot run the VMC driver without setting the sampler, optimizer as well as 
                the network!

                Try using set_sampler(), set_optimizer() and set_network() first.
                """)

        base_model = self.network
        n_heads = int(self._n_heads)
        if n_heads <= 0:
            raise ValueError(
                "No heads configured. Call set_network(..., n_heads=K) first."
            )

        # Build or use explicit head models
        if self._head_models_user is not None:
            head_models = list(self._head_models_user)
        else:
            head_models = self._build_head_models(base_model, n_heads)

        if len(head_models) != n_heads:
            raise ValueError(f"Expected {n_heads} head models, got {len(head_models)}.")

        # Apply diffeomorphism projection to each head view (scalar-output models)
        if self.diffeomorphism_invariant_simulation is True:
            head_models = [
                wrap_model(
                    hm,
                    symmetries=getattr(self, "_symmetries", None),
                    graph=self.graph,
                    characters=None,
                )
                for hm in head_models
            ]

        head_states: List[MCState] = []
        holo_flags: List[bool] = []

        for i, model_i in enumerate(head_models):
            st = MCState(
                self.sampler,
                model_i,
                n_samples=self._n_samples,
                is_group_averaged=self.diffeomorphism_invariant_simulation,
                seed=int(self.seed + 1337 * i),
                sampler_seed=int(self.seed // 2 + 7331 * i),
            )
            head_states.append(st)

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

        self._variational_state = STMultiMCState(
            head_states,
            canonical_state=self._stmh_canonical_state,
            sync_model_state=self._stmh_sync_model_state,
        )

        # Shared parameter count (true optimized parameter dimension)
        shared_params = int(head_states[self._stmh_canonical_state].n_parameters)
        nominal_replicated = int(sum(int(s.n_parameters) for s in head_states))

        self._logger.log("Number of network parameters", shared_params)

        self._logger.add_field("Network Configs", "Nominal replicated parameters")
        self._logger.log("Nominal replicated parameters", nominal_replicated)

        for i, st in enumerate(head_states):
            k = f"Head[{i}] Number of view parameters"
            self._logger.add_field("Network Configs", k)
            self._logger.log(k, int(st.n_parameters))

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

        self._logger.add_field("Network Configs", "Params(shared) / dim(Hilbert)")
        self._logger.log(
            "Params(shared) / dim(Hilbert)",
            f"{humanize.number.scientific(100.0 * shared_params / float(dim_H), precision=4)} %",
        )

        self._logger.add_field(
            "Network Configs", "Params(nominal_replicated) / dim(Hilbert)"
        )
        self._logger.log(
            "Params(nominal_replicated) / dim(Hilbert)",
            f"{humanize.number.scientific(100.0 * nominal_replicated / float(dim_H), precision=4)} %",
        )

        self._vstate_flag = True

        holomorphic = bool(all(holo_flags))
        lambda_ortho = float(kwargs.get("lambda_ortho", self.lambda_ortho))
        energy_weights = kwargs.get("energy_weights", None)
        enforce_machine_pow_2 = bool(kwargs.get("enforce_machine_pow_2", True))
        preconditioner_state_index = int(kwargs.get("preconditioner_state_index", 0))

        self._vmc_driver = SingleTrunkMultiHeadVMC(
            variational_state=self._variational_state,
            hamiltonian=self.lqx.constraint,
            optimizer=self.optimizer,
            preconditioner=SR(
                diag_shift=self.diagonal_shift,
                holomorphic=holomorphic,
                solver=self.preconditioner_solver,
            ),
            lambda_ortho=lambda_ortho,
            energy_weights=energy_weights,
            enforce_machine_pow_2=enforce_machine_pow_2,
            preconditioner_state_index=preconditioner_state_index,
        )

        self._driver_flag = True

    def _final_constraint_estimate(self) -> None:
        """Log final per-head constraint estimates (shared-parameter ST-MH version)."""

        try:
            stats_list = self.variational_state.expect(self.lqx.constraint)
        except Exception:
            return

        if not isinstance(stats_list, list):
            stats_list = [stats_list]

        self.final_constraint_stats = stats_list
        self.final_constraint_mean = [
            self._coerce_mean_value(s.mean, sigma=getattr(s, "Sigma", None))
            for s in stats_list
        ]

        if hasattr(self, "_logger") and self._logger is not None:
            try:
                self._logger.log("Network result", stats_list)
            except Exception:
                pass

            for i, st in enumerate(stats_list):
                k = f"Head[{i}] Network result"
                try:
                    self._logger.add_field("Optimization Results", k)
                except Exception:
                    pass
                try:
                    self._logger.log(k, st)
                except Exception:
                    pass

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
        state: Optional[SharedParamsMultiMCState] = None,
        silent: bool = False,
        marker: str = "",
        **kwargs,
    ) -> Optional[str]:
        """Export a shared-parameter ST-MH checkpoint to disk."""

        vstate = state or self._variational_state
        if not isinstance(vstate, SharedParamsMultiMCState):
            raise TypeError(
                "STMultiSolver.export_state expects a SharedParamsMultiMCState."
            )

        serialised_state = serialize_SharedParamsMultiMCState(vstate)

        _mpi.barrier()
        if _mpi.is_global_master():
            os.makedirs(self.output_path, exist_ok=True)
            if marker:
                marker = "_" + marker
            datetime_str = datetime.now().strftime("%d%m%Y_%H%M%S")
            filename = os.path.join(
                self.output_path,
                f"{self.hash}_SerialisedSTMHState_{datetime_str}{marker}.mpack",
            )
            save_to_file(serialised_state, filename)

            if not silent and self.printer is not None:
                self.printer.print("ST-MH shared state serialised to disk.")

        _mpi.barrier()
        return None

    def import_state(
        self,
        state_path: str,
        *,
        force_load_mpi: bool = False,
    ) -> SharedParamsMultiMCState:
        """Import a shared-parameter ST-MH checkpoint from disk."""

        _mpi.barrier()
        if _mpi.is_global_master():
            nvs = load_from_file(state_path, raw=True)
        else:
            nvs = None

        nvs = mpi_bcast(nvs, root=0)

        if not isinstance(self.variational_state, SharedParamsMultiMCState):
            raise RuntimeError(
                "STMultiSolver.import_state requires an existing SharedParamsMultiMCState template. "
                "Call initialize_vmc() after setting network/sampler/optimizer before loading."
            )

        return deserialize_SharedParamsMultiMCState(
            self.variational_state,
            nvs,
            force_load_mpi=force_load_mpi,
        )

    def plot_results(
        self,
        *,
        silent_plot: bool = False,
        with_inset: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot and save optimization curves.

        This mirrors ``MultiSolver.plot_results`` but labels the per-state curves as heads.
        """

        dpi = 300
        dpi_ratio = dpi / 100

        dt_str = datetime.now().strftime("%d%m%Y_%H%M%S")
        image_filename = os.path.join(
            self.output_path, f"{self.hash}_ModelResults_{dt_str}.png"
        )
        graph_filename = os.path.join(
            self.output_path, f"{self.hash}_Graph_{dt_str}.png"
        )

        try:
            self.graph.export(graph_filename, dpi=dpi)
        except Exception as exc:
            self.printer.print(f"Failed to export graph: {exc}")

        data_nn = self.log.data
        fig, ax = plt.subplots(
            figsize=(12, 8), dpi=dpi
        )  # pylint: disable=unused-variable

        if self.lqx.has_ground_energy:
            ge = self.lqx._ground_energy
        else:
            ge = None
        has_ge = ge is not None and not jnp.isnan(ge)

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
            return (
                hasattr(obj, "iters") and hasattr(obj, "Mean") and hasattr(obj, "Sigma")
            )

        constraint_series = []
        for k in list(data_nn.keys()):
            if isinstance(k, str) and k.startswith("Constraint"):
                try:
                    s = data_nn[k]
                    if _is_stats_series(s):
                        constraint_series.append((k, s))
                except Exception:
                    pass

        def _head_index(name: str) -> int:
            # Matches names such as "Constraint (head 0)" or fallback variants.
            try:
                lower = name.lower()
                if "head" in lower:
                    tail = lower.split("head", 1)[1]
                    digits = "".join(ch for ch in tail if ch.isdigit())
                    if digits:
                        return int(digits)
                if name.startswith("Constraint ("):
                    digits = "".join(ch for ch in name if ch.isdigit())
                    if digits:
                        return int(digits)
            except Exception:
                pass
            return 10**9

        constraint_series.sort(key=lambda kv: _head_index(kv[0]))

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
            if name == "Constraint":
                lbl = "neuraLQX"
            else:
                idx = _head_index(name)
                lbl = f"Head {idx}" if idx != 10**9 else name

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

        ax.legend(fontsize="x-large")
        ax.set_xlabel("Iteration", fontsize="xx-large")
        ax.set_ylabel(r"$\langle \hat{C}\rangle$", fontsize="xx-large")
        ax.tick_params(axis="both", which="both", labelsize="x-large")
        ax.grid(True, linestyle="--", alpha=0.7)

        if with_inset:
            inset_position_100dpi = [170, -90, 850, 700]
            inset_position = [ip * dpi_ratio for ip in inset_position_100dpi]

            max_x = self._max_iters
            min_x = max_x - 20

            ax_ins = inset_axes(
                ax, width="30%", height="30%", bbox_to_anchor=inset_position
            )

            for name, series in constraint_series:
                idx = _head_index(name)
                lbl = (
                    "NN"
                    if name == "Constraint"
                    else (f"Head {idx}" if idx != 10**9 else name)
                )
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

            if has_ge:
                ax_ins.hlines(
                    [ge],
                    xmin=min_x,
                    xmax=max_x,
                    color="#E25C47",
                    label="ED Result",
                    linewidth=2,
                )

            all_vals = []
            for _, series in constraint_series:
                try:
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
        plt.savefig(image_filename, dpi=dpi)
        self.log.serialize(
            os.path.join(self.output_path, f"{self.hash}_LogData_{dt_str}")
        )

        if (not silent_plot) and (_mpi.n_nodes == 1):
            plt.show()
        else:
            plt.close()

    # TODO: do something about this
    def __repr__(self) -> str:
        return "STMultiSolver()"
