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


"""Detail-filtered package regions and stable compiled-operation names.

Runtime regions describe host work and never insert completion fences. Native
annotations put those regions beside dispatch and correlation events. Compiled
names are unconditional metadata so changing detail does not retrace a kernel.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from . import _state
from .annotations import _NULL
from .annotations import span

if TYPE_CHECKING:
    from collections.abc import Callable

_DETAIL = {"summary": 0, "detailed": 1, "developer": 2}


class _Region:
    """Pair an application region with the owning native collector's annotation."""

    def __init__(self, session, name, category, attributes, step_num=None):
        self._span = span(name, category=category, attributes=attributes)
        """Portable application region measuring host elapsed time."""

        self._native = None
        """Owning collector's optional annotation for native dispatch context."""

        collector = session._recording_window.collector
        if collector is not None:
            self._native = collector.annotation(
                name, attributes=attributes, step_num=step_num
            )

    def __enter__(self):
        self._span.__enter__()
        try:
            if self._native is not None:
                self._native.__enter__()
        except BaseException as error:
            self._span.__exit__(type(error), error, error.__traceback__)
            raise
        return None

    def __exit__(self, kind, error, traceback):
        try:
            if self._native is not None:
                self._native.__exit__(kind, error, traceback)
        except BaseException as cleanup:
            if error is None:
                raise
            error.add_note(f"Native profile annotation cleanup also failed: {cleanup}")
        finally:
            self._span.__exit__(kind, error, traceback)
        return False


def enabled(detail: str = "detailed") -> bool:
    """Return whether a package region at this detail is currently being retained.

    Args:
        detail: summary, detailed, or developer.

    Returns:
        False outside recording windows, without reading runtime configuration.
    """
    session = _state.active
    return (
        session is not None
        and session._recording_window is not None
        and session._detail_level >= _DETAIL[detail]
    )


def region(
    name, *, detail="summary", category="neuralqx", attributes=None, step_num=None
):
    """Return a detail-filtered host and native context without waiting on arrays.

    Args:
        name: Stable package operation label.
        detail: Minimum annotation detail required by this operation.
        category: Perfetto category identifying the subsystem.
        attributes: Bounded Python scalars, never array values or representations.
        step_num: Optional logical iteration number for native step identification.

    Returns:
        A context manager, or the shared inactive context outside capture.
    """
    session = _state.active
    if (
        session is None
        or session._recording_window is None
        or session._detail_level < _DETAIL[detail]
    ):
        return _NULL
    return _Region(session, name, category, attributes, step_num)


def instrument(name: str, *, detail: str = "detailed", category: str = "neuralqx"):
    """Decorate a Python dispatch boundary with a filtered package region.

    Args:
        name: Stable operation label shared by repeated calls.
        detail: Minimum annotation detail required by the wrapper.
        category: Perfetto subsystem category.

    Returns:
        Signature-preserving decorator. Put it outside jax.jit when measuring
        a compiled dispatch. Compiled internals use named_scope instead.
    """
    threshold = _DETAIL[detail]

    def decorate(function: Callable):
        @wraps(function)
        def wrapped(*args, **kwargs):
            session = _state.active
            if (
                session is None
                or session._recording_window is None
                or session._detail_level < threshold
            ):
                return function(*args, **kwargs)
            with _Region(session, name, category, None):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def operator_region(
    operator, index: int, *, operation: str = "connections", source_indices=None
):
    """Name one operator-list member without inspecting its array-valued fields.

    Args:
        operator: Operator instance. Only its Python class name is read.
        index: Stable zero-based index within the current term list.
        operation: Work actually performed for this term, such as connections.
        source_indices: Original flattened indices when identical terms share work.

    Returns:
        A detailed region named operator[index]/Class/operation. Work shared by
        several terms receives a separate shared region at its execution site.
    """
    if not enabled("detailed"):
        return _NULL
    kind = type(operator).__name__[:96]
    attributes = {
        "operator_index": index,
        "operator_type": kind,
        "operation": operation,
    }
    if source_indices is not None:
        attributes["source_indices"] = ",".join(map(str, source_indices))[:512]
    return region(
        f"operator[{index}]/{kind}/{operation}",
        detail="detailed",
        category="neuralqx.operator",
        attributes=attributes,
    )
