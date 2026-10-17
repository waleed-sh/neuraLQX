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


"""Small append-only history containers used by runtime loggers."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
from neuralqx.utils.typing import HistoryNode


@dataclass(slots=True)
class HistorySeries:
    """Append-only scalar or array-valued history for one logged quantity."""

    steps: list[int] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)

    def append(self, step: int, value: Any) -> None:
        self.steps.append(int(step))
        self.values.append(value)

    @property
    def last(self) -> Any:
        if not self.values:
            raise IndexError("HistorySeries is empty.")
        return self.values[-1]

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(steps, values)`` as NumPy arrays."""
        return np.asarray(self.steps), np.asarray(self.values)

    def to_dict(self) -> dict[str, Any]:
        return {"iters": list(self.steps), "value": list(self.values)}

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __repr__(self) -> str:
        return f"HistorySeries(n={len(self.values)})"


def append_to_history(
    history: HistoryNode,
    item: Mapping[str, Any],
    *,
    step: int,
) -> HistoryNode:
    """Append a normalized mapping into an in-memory history tree."""
    for key, value in item.items():
        key = str(key)
        if isinstance(value, Mapping):
            child = history.get(key)
            if child is None:
                child = {}
                history[key] = child
            if isinstance(child, HistorySeries):
                raise TypeError(
                    f"Cannot log nested data under scalar history key {key!r}."
                )
            append_to_history(child, value, step=step)
        else:
            series = history.get(key)
            if series is None:
                series = HistorySeries()
                history[key] = series
            if not isinstance(series, HistorySeries):
                raise TypeError(
                    f"Cannot log scalar data under nested history key {key!r}."
                )
            series.append(step, value)
    return history


def flatten_history(
    history: Mapping[str, Any],
    *,
    prefix: tuple[str, ...] = (),
) -> dict[tuple[str, ...], HistorySeries]:
    """Return all leaf histories keyed by their path."""
    flat: dict[tuple[str, ...], HistorySeries] = {}
    for key, value in history.items():
        path = (*prefix, str(key))
        if isinstance(value, HistorySeries):
            flat[path] = value
        elif isinstance(value, Mapping):
            flat.update(flatten_history(value, prefix=path))
    return flat


def get_history_path(history: Mapping[str, Any], path: str | Iterable[str]) -> Any:
    """Return a node from a slash-separated or iterable history path."""
    parts = tuple(path.split("/")) if isinstance(path, str) else tuple(path)
    node: Any = history
    for part in parts:
        node = node[part]
    return node


__all__ = [
    "HistoryNode",
    "HistorySeries",
    "append_to_history",
    "flatten_history",
    "get_history_path",
]
