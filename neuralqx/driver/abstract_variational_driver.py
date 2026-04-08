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

# Copyright 2021 The NetKet Authors - All rights reserved.
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
A subclass of NetKet's AbstractVariationalDriver which provides variational inference with profiling.
"""

import logging
import numbers

from collections.abc import Iterable
from collections.abc import Callable

from tqdm.auto import tqdm

from netket.driver.abstract_variational_driver import (
    AbstractVariationalDriver as NKDriver,
)
from netket.driver.abstract_variational_driver import _to_iterable
from netket.driver.abstract_variational_driver import apply_gradient
from netket.logging import AbstractLog
from netket.logging import JsonLog
from netket.operator._abstract_observable import AbstractObservable
from neuralqx.utils import distributed as _dist
from netket.utils import timing
from netket.jax import tree_cast

from neuralqx.debug import trace
from neuralqx.debug import timeit
from neuralqx.profile import step as prof_step
from neuralqx.profile import section as prof_section

CallbackT = Callable[[int, dict, "AbstractVariationalDriver"], bool]


class AbstractVariationalDriver(NKDriver):

    def iter(self, n_steps: int, step: int = 1):
        """
        Returns a generator which advances the VMC optimization, yielding
        after every `step_size` steps.

        Args:
            n_steps: The total number of steps to perform (this is
                equivalent to the length of the iterator)
            step: The number of internal steps the simulation
                is advanced between yielding from the iterator

        Yields:
            int: The current step.
        """
        for _ in range(0, n_steps, step):
            for i in range(0, step):

                # mark every optimiser step as a "VMC step"
                # step_count is monotonic, already what we want to identify steps
                with prof_step(self.step_count, name="VMC", cat="vmc"):
                    # forward/backward triggers sampling + expect + grad
                    with prof_section("forward_backward (dispatch)", cat="vmc"):
                        self._dp = self._forward_and_backward()

                    if i == 0:
                        yield self.step_count

                    self._step_count += 1

                    with prof_section("update_parameters (dispatch)", cat="vmc"):
                        self.update_parameters(self._dp)

    @trace(tag="VMCDRIVER:RUN")
    def run(
        self,
        n_iter: int,
        out: AbstractLog | Iterable[AbstractLog] | str | None = (),
        obs: dict[str, AbstractObservable] | None = None,
        step_size: int = 1,
        show_progress: bool = True,
        save_params_every: int = 50,  # for default logger
        write_every: int = 50,  # for default logger
        callback: CallbackT | Iterable[CallbackT] = lambda *x: True,
        timeit: bool = False,
    ):
        """
        Runs this variational driver, updating the weights of the network stored in
        this driver for `n_iter` steps and dumping values of the observables `obs`
        in the output `logger`.

        It is possible to control more specifically what quantities are logged, when to
        stop the optimisation, or to execute arbitrary code at every step by specifying
        one or more callbacks, which are passed as a list of functions to the keyword
        argument `callback`.

        Callbacks are functions that follow this signature:

        .. Code::

            def callback(step, log_data, driver) -> bool:
                ...
                return True/False

        If a callback returns True, the optimisation continues, otherwise it is stopped.
        The `log_data` is a dictionary that can be modified in-place to change what is
        logged at every step. For example, this can be used to log additional quantities
        such as the acceptance rate of a sampler.

        Loggers are specified as an iterable passed to the keyword argument `out`. If only
        a string is specified, this will create by default a :class:`nk.logging.JsonLog`.
        To know about the output format check its documentation. The logger object is
        also returned at the end of this function so that you can inspect the results
        without reading the json output.

        When running among multiple MPI ranks/Jax devices, the logging logic is executed
        on all nodes, but only root-rank loggers should write to files or do expensive I/O
        operations.

        .. note::

            Before NetKet 3.15, loggers where automatically 'ignored' on non-root ranks.
            However, starting with NetKet 3.15 it is the responsability of a logger to
            check if it is executing on a non-root rank, and to 'do nothing' if that is
            the case.

            The change was required to work correctly and efficiently with sharding. It will
            only affect users that were defining custom loggers themselves.

        Args:
            n_iter: the total number of iterations to be performed during this run.
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
                If this argument is a string, it will be used as output prefix for the standard JSON logger.
            obs: An iterable containing all observables that should be computed
            step_size: Every how many steps should observables be logged to disk (default=1)
            callback: Callable or list of callable callback functions to stop training given a condition
            show_progress: If true displays a progress bar (default=True)
            save_params_every: Every how many steps the parameters of the network should be
                serialized to disk (ignored if logger is provided)
            write_every: Every how many steps the json data should be flushed to disk (ignored if
                logger is provided)
            timeit: If True, provide timing information.
        """

        if not isinstance(n_iter, numbers.Number):
            raise ValueError(
                "n_iter, the first positional argument to `run`, must be a number!"
            )

        if obs is None:
            obs = {}

        # if out is a path, create an overwriting Json Log for output
        if isinstance(out, str):
            out = JsonLog(out, "w", save_params_every, write_every)
        elif out is None:
            out = ()

        loggers = _to_iterable(out)
        callbacks = _to_iterable(callback)
        callback_stop = False

        with timing.timed_scope(force=timeit) as timer:
            with tqdm(
                total=n_iter,
                disable=not show_progress or not self._is_root,
                dynamic_ncols=True,
            ) as pbar:
                old_step = self.step_count
                first_step = True

                for step in self.iter(n_iter, step_size):

                    with prof_section("estimate (dispatch)", cat="vmc"):
                        log_data = self.estimate(obs)

                    with prof_section("log_additional_data (dispatch)", cat="vmc"):
                        self._log_additional_data(log_data, step)

                    # If the cost-function is defined then store it in the log payload.
                    # Avoid formatting it into the progress bar unless the bar is actually
                    # visible, because stringifying Stats can force host transfers.
                    if self._loss_stats is not None:
                        if show_progress and self._is_root:
                            pbar.set_postfix_str(
                                self._loss_name + "=" + str(self._loss_stats)
                            )
                        log_data[self._loss_name] = self._loss_stats

                    # Execute callbacks before loggers because they can append to log_data
                    with prof_section("callbacks (dispatch)", cat="vmc"):
                        for callback in callbacks:
                            if not callback(step, log_data, self):
                                callback_stop = True

                    with prof_section("loggers (dispatch)", cat="vmc"):
                        with timing.timed_scope(name="loggers"):
                            for logger in loggers:
                                logger(self.step_count, log_data, self.state)

                    if len(callbacks) > 0:
                        if _dist.mpi_any(callback_stop):
                            break

                    # Reset the timing of tqdm after the first step, to ignore compilation time
                    if first_step:
                        first_step = False
                        pbar.unpause()

                    # Update the progress bar
                    pbar.update(self.step_count - old_step)
                    old_step = self.step_count

                # Final update so that it shows up filled.
                pbar.update(self.step_count - old_step)

        # flush at the end of the evolution so that final values are saved to
        # file
        for logger in loggers:
            logger.flush(self.state)

        if timeit:
            self._timer = timer
            if self._is_root:
                print(timer)

        return loggers

    @timeit(tag="VMDRIVER:FORWARD_BACKWARD", warn_ms=10000, level=logging.DEBUG)
    def _forward_and_backward(self):
        """
        Compute the current loss statistics and a preconditioned gradient update direction.

        This method resets the variational state statistics, evaluates the objective and its
        gradient for the stored Hamiltonian operators, applies the configured preconditioner,
        and casts the resulting update direction to match the parameter PyTree structure and
        dtype conventions.

        :return: The preconditioned gradient PyTree to be used by the optimiser.
        """

        with prof_section("state.reset (dispatch)", cat="vmc"):
            self.state.reset()

        # Compute the local energy estimator and average Energy
        with prof_section("expect_and_grad (sync)", cat="vmc") as sec:
            self._loss_stats, self._loss_grad = sec.sync(
                self.state.expect_and_grad(self._ham)  # pylint: disable=W0201
            )

        # if it's the identity it does
        # self._dp = self._loss_grad
        with prof_section("preconditioner (sync)", cat="vmc") as sec:
            self._dp = sec.sync(
                self.preconditioner(self.state, self._loss_grad, self.step_count)
            )

        # If parameters are real, then take only real part of the gradient (if it's complex)
        with prof_section("tree_cast (dispatch)", cat="vmc"):
            self._dp = tree_cast(self._dp, self.state.parameters)

        return self._dp

    def update_parameters(self, dp):
        """
        Updates the parameters of the machine using the optimizer in this driver

        Args:
            dp: the pytree containing the updates to the parameters
        """
        with prof_section("apply_gradient (sync)", cat="vmc") as sec:
            self._optimizer_state, self.state.parameters = sec.sync(
                apply_gradient(
                    self._optimizer.update,
                    self._optimizer_state,
                    dp,
                    self.state.parameters,
                )
            )
