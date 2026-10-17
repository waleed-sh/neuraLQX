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


"""Matplotlib plotting helpers for runtime logs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .history import HistorySeries
from .history import flatten_history
from .history import get_history_path


def plot_history(
    log: Any,
    keys: str | tuple[str, ...] | list[str | tuple[str, ...]] | None = None,
    *,
    field: str | None = "Mean",
    ax: Any | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Plot numeric histories from a runtime log or history mapping."""
    from matplotlib import pyplot as plt

    history = getattr(log, "data", log)
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    selected = _select_series(history, keys=keys, field=field)
    for label, series in selected:
        steps, values = series.as_arrays()
        values = _numeric_values(values)
        if values is None:
            continue
        if np.iscomplexobj(values):
            ax.plot(steps, values.real, label=f"{label}.real")
            ax.plot(steps, values.imag, label=f"{label}.imag")
        else:
            ax.plot(steps, values, label=label)

    ax.set_xlabel("step")
    ax.set_ylabel("value")
    if title is not None:
        ax.set_title(title)
    if ax.lines:
        ax.legend()
    if show:
        plt.show()
    return figure, ax


def _select_series(
    history: Mapping[str, Any],
    *,
    keys: str | tuple[str, ...] | list[str | tuple[str, ...]] | None,
    field: str | None,
) -> list[tuple[str, HistorySeries]]:
    if keys is None:
        return [
            ("/".join(path), series)
            for path, series in flatten_history(history).items()
            if _numeric_values(np.asarray(series.values)) is not None
        ]
    if isinstance(keys, (str, tuple)):
        keys = [keys]
    selected = []
    for key in keys:
        node = get_history_path(history, key)
        path = tuple(key.split("/")) if isinstance(key, str) else tuple(key)
        if isinstance(node, HistorySeries):
            selected.append(("/".join(path), node))
            continue
        if field is not None and isinstance(node, Mapping) and field in node:
            child = node[field]
            if isinstance(child, HistorySeries):
                selected.append(("/".join((*path, field)), child))
                continue
        if isinstance(node, Mapping):
            selected.extend(
                ("/".join((*path, *child_path)), series)
                for child_path, series in flatten_history(node).items()
            )
    return selected


def _numeric_values(values: np.ndarray) -> np.ndarray | None:
    if values.dtype == object:
        try:
            values = np.asarray(values, dtype=np.complex128)
        except (TypeError, ValueError):
            try:
                values = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError):
                return None
    if not np.issubdtype(values.dtype, np.number):
        return None
    return values


__all__ = ["plot_history"]
