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


"""Progress-bar callbacks."""

from __future__ import annotations

from typing import Any

from .base import Callback
from .context import CallbackContext


class ProgressBar(Callback):
    """Low-overhead tqdm progress callback for driver runs."""

    priority = 10_000

    def __init__(
        self,
        total: int,
        *,
        every: int = 1,
        min_interval: float | None = 0.0,
        leave: bool = True,
    ) -> None:
        self.total = int(total)
        self.every = int(every)
        if self.every <= 0:
            raise ValueError("every must be positive.")
        self.min_interval = min_interval
        self.leave = bool(leave)
        self._bar: Any | None = None

    def run_started(self, context: CallbackContext) -> None:
        del context
        try:
            from tqdm.auto import tqdm
        except ImportError:
            self._bar = None
            return

        disable = False
        try:
            import jax

            disable = jax.process_index() != 0
        except Exception:
            disable = False

        kwargs: dict[str, Any] = {
            "total": self.total,
            "dynamic_ncols": True,
            "leave": self.leave,
            "disable": disable,
        }
        if self.min_interval is not None:
            kwargs["mininterval"] = float(self.min_interval)
        self._bar = tqdm(**kwargs)

    def step_finished(self, context: CallbackContext) -> None:
        bar = self._bar
        if bar is None:
            return
        next_n = int(getattr(bar, "n", 0)) + 1
        if _progress_should_refresh_postfix(bar, every=self.every, n=next_n):
            bar.set_postfix_str(f"{context.loss_name}={context.loss}", refresh=False)
        bar.update(1)

    def run_finished(self, context: CallbackContext) -> None:
        del context
        self.close()

    def run_failed(self, context: CallbackContext) -> None:
        del context
        self.close()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def _progress_should_refresh_postfix(
    progress: Any,
    *,
    every: int,
    n: int | None = None,
) -> bool:
    if n is None:
        n = getattr(progress, "n", None)
    total = getattr(progress, "total", None)
    if total is not None and n is not None and n >= total:
        return True
    if n is not None and int(n) % int(every) != 0:
        return False

    timer = getattr(progress, "_time", None)
    last_print_t = getattr(progress, "last_print_t", None)
    mininterval = getattr(progress, "mininterval", None)
    if timer is None or last_print_t is None or mininterval is None:
        return True
    return bool(timer() - last_print_t >= mininterval)


ProgressBarCallback = ProgressBar


__all__ = ["ProgressBar", "ProgressBarCallback"]
