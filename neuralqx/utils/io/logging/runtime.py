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


"""In-memory runtime logger."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .base import AbstractLogger
from .history import HistoryNode
from .history import append_to_history
from .history import flatten_history
from .history import get_history_path
from .plotting import plot_history
from .values import normalize_log_tree
from .values import serializable_log_tree


class RuntimeLogger(AbstractLogger):
    """Append-only in-memory logger for driver runs.

    The logger is intentionally simple: each leaf quantity is stored as a
    :class:`HistorySeries` with separate step and value lists. No full-tree copy
    is made per step.
    """

    def __init__(
        self,
        *,
        every: int = 1,
        log_on_all_processes: bool = False,
    ) -> None:
        every = int(every)
        if every <= 0:
            raise ValueError("every must be positive.")
        self.every = every
        self.log_on_all_processes = bool(log_on_all_processes)
        self._data: HistoryNode = {}
        self._last_step: int | None = None

    @property
    def data(self) -> HistoryNode:
        """Nested history tree accumulated by this logger."""
        return self._data

    @property
    def last_step(self) -> int | None:
        """Last step accepted by this logger."""
        return self._last_step

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __call__(
        self,
        step: int,
        item: Mapping[str, Any],
        variational_state: Any | None = None,
        *,
        driver: Any | None = None,
    ) -> None:
        del variational_state, driver
        step = int(step)
        if step % self.every != 0:
            return
        if not self.is_active_process:
            return
        append_to_history(self._data, normalize_log_tree(item), step=step)
        self._last_step = step

    def clear(self) -> None:
        """Clear all accumulated data."""
        self._data.clear()
        self._last_step = None

    def series(self, path: str | tuple[str, ...]) -> Any:
        """Return a nested history node or leaf series by path."""
        return get_history_path(self._data, path)

    def flatten(self) -> dict[tuple[str, ...], Any]:
        """Return leaf histories keyed by path tuples."""
        return flatten_history(self._data)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly nested dictionary."""
        return serializable_log_tree(_history_to_plain_tree(self._data))

    def save_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        """Serialize the current runtime log to JSON on the primary process."""
        path = Path(path)
        if path.suffix == "":
            path = path.with_suffix(".json")
        if self.is_active_process:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf8")
        return path

    def plot(
        self,
        keys: str | tuple[str, ...] | list[str | tuple[str, ...]] | None = None,
        *,
        field: str | None = "Mean",
        ax: Any | None = None,
        show: bool = False,
        title: str | None = None,
    ):
        """Plot selected numeric histories with matplotlib."""
        return plot_history(
            self,
            keys=keys,
            field=field,
            ax=ax,
            show=show,
            title=title,
        )

    def __repr__(self) -> str:
        keys = sorted("/".join(path) for path in self.flatten())
        return f"RuntimeLogger(keys={keys}, every={self.every})"


def _history_to_plain_tree(history: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in history.items():
        if hasattr(value, "to_dict") and callable(value.to_dict):
            out[key] = value.to_dict()
        elif isinstance(value, Mapping):
            out[key] = _history_to_plain_tree(value)
        else:
            out[key] = value
    return out


__all__ = ["RuntimeLogger"]
