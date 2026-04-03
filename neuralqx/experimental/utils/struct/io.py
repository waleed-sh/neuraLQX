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

"""
Public serialization APIs for Struct and registered pytree types.

The implementation lives in :mod:`neuralqx.utils.struct._internals.io`.
This module intentionally exposes a compact, stable public surface while hiding
wire-format and storage details behind internal helpers.

Supported use cases:
    1. Convert runtime values into Python-inspection views (:func:`to_python`).
    2. Build in-memory portable payloads (:func:`to_state_dict`).
    3. Restore from in-memory payloads (:func:`from_state_dict`).
    4. Persist payloads to directory or ``.zip`` bundles (:func:`export`).
    5. Load persisted bundles back into objects (:func:`load`).

Struct I/O uses a two-part representation:
    - manifest: JSON-friendly structural metadata and object graph encoding.
    - arrays: compressed array blobs stored separately for efficiency.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._internals.io import export_impl
from ._internals.io import from_state_dict_impl
from ._internals.io import load_impl
from ._internals.io import to_python_value
from ._internals.io import to_state_dict_impl

__all__ = [
    "to_python",
    "to_state_dict",
    "from_state_dict",
    "export",
    "load",
]


def to_python(value: Any, *, include_opaque: bool = True) -> Any:
    """
    Convert a possibly nested value into Python-native inspection form.

    This helper recursively converts:
        - Struct objects into ``dict`` form,
        - registered pytrees into explicit structural views,
        - JAX/NumPy arrays into NumPy arrays.

    Args:
        value: Any value possibly containing structs/pytrees.
        include_opaque: Whether to include opaque struct fields in the resulting object graph.

    Returns:
        A Python-friendly representation preserving overall structure.

    Notes:
        ``to_python`` is designed for introspection/debugging, not for persistence.
        For portable storage use :func:`to_state_dict` or :func:`export`.
    """
    return to_python_value(value, include_opaque=include_opaque)


def to_state_dict(obj: Any) -> dict[str, Any]:
    """
    Encode an object graph into a portable state-dict payload.

    The returned mapping contains:
        - ``version``: schema version integer.
        - ``manifest``: JSON-serialisable structural manifest.
        - ``arrays``: array metadata (shape/dtype).
        - ``array_data``: in-memory NumPy arrays.

    Args:
        obj: Root object to encode. May be a Struct, a registered pytree class, or
            nested built-in containers of supported types.

    Returns:
        In-memory transport format suitable for :func:`from_state_dict`.

    Raises:
        SerializationError: if unsupported values are encountered and no adapter is registered.
    """
    return to_state_dict_impl(obj)


def from_state_dict(
    payload: Mapping[str, Any], *, expected_cls: type[Any] | None = None
) -> Any:
    """
    Reconstruct an object graph from :func:`to_state_dict` output.

    Args:
        payload: State payload produced by :func:`to_state_dict`.
        expected_cls: Optional type guard. If provided, serialised root object must be an
            instance/subclass of this type.

    Returns:
        Reconstructed object graph.

    Raises:
        SerializationError: If payload schema/version is invalid, required arrays are missing, class
            resolution fails, adapters are unavailable, or type guard check fails.
    """
    return from_state_dict_impl(payload, expected_cls=expected_cls)


def export(obj: Any, path: str | Path, *, overwrite: bool = False) -> Path:
    """
    Export object state to a directory bundle or ``.zip`` archive.

    Directory/zip output always includes:
        - ``manifest.json`` structural payload
        - ``arrays.npz`` compressed array buffer

    Args:
        obj: Object graph to export.
        path: Output location. Suffix ``.zip`` writes a compressed archive, any other
            path writes a directory bundle.
        overwrite: If ``False`` (default), existing output path raises ``FileExistsError``.

    Returns:
        Final output path.
    """
    return export_impl(obj, path, overwrite=overwrite)


def load(path: str | Path, *, expected_cls: type[Any] | None = None) -> Any:
    """
    Load a previously exported state bundle.

    Args:
        path: Either a directory path containing ``manifest.json`` and ``arrays.npz``,
            or a ``.zip`` file with the same contents.
        expected_cls: Optional runtime guard for expected root type.

    Returns:
        Decoded object graph.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        SerializationError: If bundle files are malformed, schema checks fail, or type resolution
            cannot be completed.
    """
    return load_impl(path, expected_cls=expected_cls)
