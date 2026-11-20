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


"""Deterministic ownership of automatic driver captures and iteration boundaries."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

from . import _state
from ._instrumentation import region
from .annotations import _NULL
from .options import ProfileOptions
from .session import Profiler


class _DisabledRun:
    """Shared inactive controller with no per-iteration configuration reads."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def step(self, index):
        """Return the shared inactive iteration context."""
        return _NULL


_DISABLED = _DisabledRun()


class _Completion:
    """Keep independent observable outputs alive until their device work completes."""

    def __init__(self, driver, extra_results):
        self._driver = driver
        """Owner of the current parameter, gradient, optimizer, and sampler arrays."""

        self._extra_results = extra_results
        """Optional callback exposing observable and callback outputs in log data."""

        self._pending = {}
        """Unfinished array leaves keyed by identity, pruned without waiting."""

    def retain(self):
        """Retain unfinished log outputs before the next iteration replaces them.

        Readiness queries do not wait for completion. Keeping only unfinished
        leaves bounds retention by outstanding work rather than run length.
        """
        if self._extra_results is None:
            return
        import jax

        pending = {
            key: value for key, value in self._pending.items() if not value.is_ready()
        }
        for value in jax.tree_util.tree_leaves(self._extra_results()):
            if isinstance(value, jax.Array) and not value.is_ready():
                pending[id(value)] = value
        self._pending = pending

    def __call__(self):
        """Return live outputs, including independent work from earlier iterations."""
        return (
            self._driver._profile_completion_values(),
            tuple(self._pending.values()),
            None if self._extra_results is None else self._extra_results(),
        )


class _DriverCapture:
    """Borrow an explicit profiler or own the bounded lifetime of driver.run."""

    def __init__(self, driver, session, owned, completion):
        self._driver = driver
        """Driver whose current arrays and automatic result belong to this run."""

        self._session = session
        """Existing explicit session or the session opened for this run."""

        self._owned = owned
        """Whether this controller must enter and close the session."""

        self._completion = completion
        """Callback returning current independent result branches without reading devices."""

        self._iteration = 0
        """Number of attempted iterations, including rejected parameter updates."""

    def __enter__(self):
        if self._owned:
            self._session.__enter__()
        else:
            self._session._check_owner()
        return self

    def __exit__(self, kind, error, traceback):
        if self._owned:
            try:
                return self._session.__exit__(kind, error, traceback)
            finally:
                self._driver.profile_result = self._session.result
        return False

    @contextmanager
    def step(self, index):
        """Record one attempted iteration and then advance the capture schedule.

        Args:
            index: Accepted optimizer step count before this attempt. Rejected
                updates retain that count but still advance the profile schedule.

        Yields:
            Control inside the iteration region. Completion values are read only
            after its regions close, so a scheduled boundary cannot clip them.
        """
        failure = None
        try:
            with region(
                "vmc.step",
                category="neuralqx.driver",
                attributes={"step": index, "iteration": self._iteration},
                step_num=self._session._step,
            ):
                yield
        except BaseException as error:
            failure = error
            raise
        finally:
            self._iteration += 1
            try:
                self._completion.retain()
                self._session.step(wait_for=self._completion())
            except BaseException as cleanup:
                if failure is None:
                    raise
                failure.add_note(f"Profiling step completion also failed: {cleanup}")


def driver_capture(driver, *, automatic=True, extra_results=None):
    """Select one capture for a bounded driver run or borrow an explicit session.

    Args:
        driver: Driver providing current result arrays and profile_result storage.
        automatic: Whether config.profile may start a session. Iterators borrow
            explicit sessions because their lifetime can outlive iteration.
        extra_results: Optional callback returning pending observable and logging
            results that are independent of the parameter update.

    Returns:
        A controller that owns automatic capture, borrows an existing session,
        or performs no recording. An existing session's settings take precedence.
    """

    active = _state.active
    if active is not None:
        return _DriverCapture(driver, active, False, _Completion(driver, extra_results))
    if not automatic:
        return _DISABLED
    from neuralqx.config import config

    driver.profile_result = None
    if not config.get("profile"):
        return _DISABLED
    options = ProfileOptions.from_config(config)
    if options.level == "off":
        options = replace(options, level="basic")
    completion = _Completion(driver, extra_results)
    session = Profiler(options=options, wait_for=completion)
    return _DriverCapture(driver, session, True, completion)
