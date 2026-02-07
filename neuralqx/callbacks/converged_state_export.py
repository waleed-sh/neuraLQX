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
This module defines a NetKet-compatible callback for Variational Monte Carlo (VMC)
optimisation runs. The callback monitors a chosen loss statistic (e.g. mean or
error estimate) and triggers an export of the current Monte Carlo state whenever
the monitored value reaches (or falls below) a user-defined target threshold.

The exported state is intended to be a fully serialised `MCState` (including the
sampler state), written via the solver's export facility (typically to `.mpack`
format). This enables checkpointing "good" optimisation states at the moment a
desired loss level is achieved.

The callback is implemented as a `netket.utils.struct.Pytree`, making it
compatible with NetKet's driver callback mechanism and JAX/pytrees where
appropriate.

Notes
-----
- The callback always returns ``True`` to indicate that optimisation should
  continue, it is a logging/export side-effect hook rather than a stopping
  criterion.
- Only a fixed set of monitoring keys is accepted to avoid silent typos and to
  align with the statistics provided by NetKet log entries.
"""

from typing import Any
from typing import TYPE_CHECKING

import numpy as np
from netket.utils import struct

from neuralqx.utils.errors import IncorrectMonitoringValueError

if TYPE_CHECKING:
    from neuralqx.solver import AbstractSolver

_ALLOWED_MONITOR = ["Mean", "Sigma", "R_hat", "TauCorr"]


class StateExportCallback(struct.Pytree, mutable=True):
    """
    Export the serialised MCState when the monitored loss statistic reaches a target threshold.

    The callback reads the current loss entry from `log_data[driver._loss_name]`, extracts the
    requested statistic (configured via `monitor`), and compares it to `target`. If the value
    is less than or equal to `target`, it exports the driver's current state through the
    provided `Solver` instance. The export marker includes step and diagnostic statistics.

    :param target: Target threshold for the monitored loss statistic. When the monitored value
        is <= this threshold, an export is triggered.
    :param solver: Instance of `neuralqx.solver.Solver` used to perform the serialised export
        (must provide `export_serialized_state(...)`).
    :param monitor: Name of the statistic to read from the driver's loss log entry. Must be one
        of ["Mean", "Sigma", "R_hat", "TauCorr"]. Defaults to "Mean".
    :raises IncorrectMonitoringValueError: If `monitor` is not one of the allowed monitoring keys.
    """

    monitor: str
    """The loss statistic to track (mean, variance, sigma)"""

    target: float
    """The target value for the loss statistic to scan for"""

    _solver: Any
    r"""An instance of :class:`~neuralqx.solver.Solver`"""

    def __init__(
        self,
        target: float,
        solver: "AbstractSolver",
        monitor: str = "Mean",
    ):
        """
        Construct the callback.

        :param target: Target threshold for the monitored loss statistic.
        :param solver: Solver instance used to export the serialised state.
        :param monitor: Loss statistic key to monitor, must be one of
            ["Mean", "Sigma", "R_hat", "TauCorr"].
        :raises IncorrectMonitoringValueError: If `monitor` is not in the allowed set.
        """

        if monitor not in _ALLOWED_MONITOR:
            raise IncorrectMonitoringValueError(monitor, _ALLOWED_MONITOR)

        self.target = target
        self.monitor = monitor
        self._solver = solver

    def __call__(self, step, log_data, driver) -> bool:
        """
        Evaluate the callback at a given optimisation step and export if the target is reached.

        This function always returns `True` because it is an export/logging callback and does
        not request early termination of the optimisation.

        :param step: Current optimisation step index provided by the driver.
        :param log_data: Logging container holding the loss entry under `driver._loss_name`.
            The loss entry is expected to expose attributes including the configured `monitor`
            key, as well as "Sigma", "Variance", and "R_hat" for marker diagnostics.
        :param driver: NetKet driver instance. Must expose `_loss_name` (key into `log_data`)
            and `state` (the MCState to be exported).
        :return: Always True, indicating the driver should continue.
        :raises KeyError: If `driver._loss_name` is not present in `log_data`.
        :raises AttributeError: If the loss entry does not expose the requested monitoring
            attribute or expected diagnostic attributes.
        """

        # get the loss from the log_data and index the most recent value
        mean = np.real(getattr(log_data[driver._loss_name], self.monitor))
        rhat = np.real(getattr(log_data[driver._loss_name], "R_hat"))
        sigma = np.real(getattr(log_data[driver._loss_name], "Sigma"))
        variance = np.real(getattr(log_data[driver._loss_name], "Variance"))

        if mean <= self.target:
            # the default neuraLQX export function will ensure that only the master process performs I/O
            self._solver.export_state(
                marker="callback_export_at_step_"
                + str(step)
                + f"__loss__[{mean:.5f} ± {sigma:.5f} [σ²={variance:.5f}, R̂={rhat:.5f}]]",
                silent=True,
                state=driver.state,
            )

        return True
