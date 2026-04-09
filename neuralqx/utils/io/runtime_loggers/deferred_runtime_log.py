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

from __future__ import annotations

from typing import Any

from netket.logging.runtime_log import RuntimeLog
from netket.utils.history import HistoryDict
from netket.utils.history import accum_histories_in_tree


class DeferredRuntimeLog(RuntimeLog):
    """
    RuntimeLog variant that can defer history accumulation until flush.

    When ``defer_accumulation`` is enabled, per-step payloads are stored as raw entries and
    converted into History objects only when ``flush()``, ``serialize()`` or ``data`` access
    is requested.
    """

    def __init__(self):
        super().__init__()
        self._defer_accumulation = False
        self._pending_steps: list[int] = []
        self._pending_items: list[dict[str, Any]] = []

    @property
    def defer_accumulation(self) -> bool:
        """Whether per-step accumulation is deferred to flush."""
        return self._defer_accumulation

    def set_deferred_accumulation(self, enabled: bool) -> None:
        """
        Enable/disable deferred accumulation mode.

        Disabling deferred mode materializes any pending entries immediately.
        """
        if not enabled:
            self._materialize_pending()
        self._defer_accumulation = bool(enabled)

    def _materialize_pending(self) -> None:
        """Convert buffered entries into History objects."""
        if not self._pending_steps:
            return

        if self._data is None:
            self._data = HistoryDict()

        for step, item in zip(self._pending_steps, self._pending_items):
            self._data = accum_histories_in_tree(self._data, item, step=step)

        self._pending_steps.clear()
        self._pending_items.clear()

    def __call__(
        self,
        step: int,
        item: dict[str, Any],
        variational_state=None,
    ):
        if not self._defer_accumulation:
            super().__call__(step, item, variational_state)
            return

        self._pending_steps.append(step)
        self._pending_items.append(item)
        self._old_step = step

    @property
    def data(self) -> dict[str, Any]:
        self._materialize_pending()
        return super().data

    def __getitem__(self, key: str) -> Any:
        self._materialize_pending()
        return super().__getitem__(key)

    def flush(self, variational_state=None):
        self._materialize_pending()

    def serialize(self, path):
        self._materialize_pending()
        return super().serialize(path)
