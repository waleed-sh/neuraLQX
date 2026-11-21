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
Internal serialization engine for Struct and registered pytree types.

Wire format
-----------
Every state payload is a dict with the following top-level keys:

    ``version`` (int)
        Schema version. Currently ``1``. Used by ``_migrate`` to apply
        forward-compatible transformations when loading older bundles.
    ``manifest`` (JSON-serialisable dict)
        Recursive structural description of the object graph. Arrays are
        referenced by name rather than embedded inline.
    ``arrays`` (dict)
        Array metadata (shape / dtype) for manifest validation.
    ``array_data`` (dict[str, np.ndarray])
        In-memory array buffers, keyed by the names used in the manifest.
        Omitted from on-disk bundles (arrays are stored separately in
        ``arrays.npz``).

Known limitations
-----------------
- ``set`` and ``frozenset`` elements are serialised as lists. Because Python
  set iteration order is non-deterministic, two serialisations of the same
  set may produce different ``items`` lists. Content-based checksums of the
  manifest are therefore unreliable for set-containing payloads.
- ``Mapping`` subclasses (``OrderedDict``, ``defaultdict``, etc.) are
  serialised as plain ``dict`` and deserialised as ``dict``. Subclass type
  information is lost on round-trip.
- ``bytes`` values are not serialisable. Fields that must store raw byte
  buffers should be marked ``serialize=False`` or converted to a supported
  type (e.g. a NumPy ``uint8`` array) before storing in a Struct field.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

import jax
from neuralqx.utils.errors import NeuralqxFileExistsError
from neuralqx.utils.errors import NeuralqxFileNotFoundError
from neuralqx.utils.errors import NeuralqxOSError
from neuralqx.utils.errors import NeuralqxPermissionError
from neuralqx.utils.errors import NeuralqxRuntimeError
from neuralqx.utils.errors import NeuralqxValueError
from neuralqx.utils.errors import SerializationError
from neuralqx.utils.errors import ValidationError

from ..fields import FieldKind
from ..pytree import resolve_pytree_spec
from ..registry import class_ref
from ..registry import resolve_class
from ._types import Array


def _struct_type() -> type[Any]:
    from ..base import Struct

    return Struct


def _serialize_dist_error(exc: Exception) -> dict[str, str]:
    """Serialise exception metadata for propagation between processes."""
    return {"type": type(exc).__name__, "message": str(exc)}


def _reraise_dist_error(payload: Mapping[str, Any], *, context: str) -> None:
    """Re-raise a shared exception payload on each participating process."""
    err = payload.get("error")
    if err is None:
        return
    if not isinstance(err, Mapping):
        raise NeuralqxRuntimeError(
            f"{context} failed with malformed distributed error payload: {err!r}"
        )

    name = str(err.get("type", "RuntimeError"))
    message = str(err.get("message", ""))
    known: dict[str, type[Exception]] = {
        "SerializationError": SerializationError,
        "ValidationError": ValidationError,
        "FileExistsError": NeuralqxFileExistsError,
        "FileNotFoundError": NeuralqxFileNotFoundError,
        "PermissionError": NeuralqxPermissionError,
        "ValueError": NeuralqxValueError,
        "RuntimeError": NeuralqxRuntimeError,
        "OSError": NeuralqxOSError,
    }
    known.update({kind.__name__: kind for kind in tuple(known.values())})
    exc_type = known.get(name)
    if exc_type is not None:
        raise exc_type(message)
    raise NeuralqxRuntimeError(
        f"{context} failed on a participating process ({name}): {message}"
    )


def _collective_call(function: Callable[[], Any], *, context: str) -> Any:
    """Propagate recoverable Python errors before starting the next collective."""
    if jax.process_count() == 1:
        return function()
    from jax.experimental import multihost_utils

    result = None
    message = b""
    try:
        result = function()
    except Exception as exc:
        error = _serialize_dist_error(exc)
        error["type"] = error["type"][:128]
        error["message"] = (
            f"{context}, process {jax.process_index()}: {error['message'][:1000]}"
        )
        message = json.dumps(error).encode("utf-8")
    buffer = np.zeros(8192, dtype=np.uint8)
    buffer[: len(message)] = np.frombuffer(message, dtype=np.uint8)
    errors = np.asarray(multihost_utils.process_allgather(buffer, tiled=False))
    for row in errors:
        if np.any(row):
            error = json.loads(bytes(row).rstrip(b"\x00"))
            _reraise_dist_error({"error": error}, context=context)
    return result


def _agree(value: Any, *, context: str) -> None:
    """Check control metadata before any shape-dependent array collective."""
    if jax.process_count() == 1:
        return
    from jax.experimental import multihost_utils

    digest = _collective_call(
        lambda: np.frombuffer(
            hashlib.sha256(json.dumps(value, sort_keys=True).encode()).digest(),
            dtype=np.uint8,
        ),
        context=context,
    )
    hashes = np.asarray(multihost_utils.process_allgather(digest, tiled=False))
    if not np.all(hashes == hashes[0]):
        raise SerializationError(f"Processes disagree on {context}.")


def _broadcast_json(value: Any) -> Any:
    """Broadcast a JSON control message from process zero."""
    if jax.process_count() == 1:
        return value
    from jax.experimental import multihost_utils

    data = _collective_call(
        lambda: json.dumps(value).encode() if jax.process_index() == 0 else b"",
        context="Struct metadata encoding",
    )
    size = int(
        multihost_utils.broadcast_one_to_all(np.array(len(data), dtype=np.uint32))
    )
    buffer = _collective_call(
        lambda: (
            np.frombuffer(data, dtype=np.uint8).copy()
            if jax.process_index() == 0
            else np.zeros(size, dtype=np.uint8)
        ),
        context="Struct metadata allocation",
    )
    return json.loads(bytes(multihost_utils.broadcast_one_to_all(buffer)))


def to_python_value(value: Any, *, include_opaque: bool = True) -> Any:
    """Convert nested struct/registered pytree values into python-native objects."""
    Struct = _struct_type()

    if isinstance(value, Struct):
        out = {}
        for name, spec in value.fields().items():
            if spec.kind is FieldKind.OPAQUE and not include_opaque:
                continue
            out[name] = to_python_value(
                getattr(value, name), include_opaque=include_opaque
            )
        return out

    spec = None
    ref = class_ref(type(value))
    try:
        spec = resolve_pytree_spec(ref)
    except SerializationError:
        spec = None

    if spec is not None and not isinstance(value, Struct):
        if spec.serializer is not None:
            return to_python_value(
                spec.serializer(value), include_opaque=include_opaque
            )

        children, aux = spec.flatten(value)
        return {
            "__kind__": "registered-pytree-view",
            "type": ref,
            "children": [
                to_python_value(v, include_opaque=include_opaque) for v in children
            ],
            "aux": to_python_value(aux, include_opaque=include_opaque),
        }

    if isinstance(value, (Array, np.ndarray)):
        return np.asarray(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            to_python_value(k, include_opaque=include_opaque): to_python_value(
                v, include_opaque=include_opaque
            )
            for k, v in value.items()
        }
    if isinstance(value, tuple):
        return tuple(to_python_value(v, include_opaque=include_opaque) for v in value)
    if isinstance(value, list):
        return [to_python_value(v, include_opaque=include_opaque) for v in value]
    if isinstance(value, set):
        return {to_python_value(v, include_opaque=include_opaque) for v in value}
    if isinstance(value, frozenset):
        return frozenset(
            to_python_value(v, include_opaque=include_opaque) for v in value
        )
    return value


class ArrayStore:
    """In-memory buffer for array payloads during state-dict construction."""

    def __init__(self, *, materialize: bool = True) -> None:
        self.counter = 0
        self.materialize = materialize
        self.arrays: dict[str, Any] = {}

    def put(self, arr: Any) -> dict[str, Any]:
        name = f"arr_{self.counter:06d}"
        self.counter += 1
        try:
            dtype = np.dtype(arr.dtype)
        except TypeError as exc:
            raise SerializationError(
                f"Cannot store array dtype {arr.dtype}. Convert typed PRNG keys "
                "with jax.random.key_data before serialising them."
            ) from exc
        if not (jax.dtypes.issubdtype(dtype, np.number) or dtype == np.dtype(bool)):
            raise SerializationError(f"Cannot store non-numeric array dtype {dtype}.")
        self.arrays[name] = np.asarray(arr) if self.materialize else arr
        return {
            "__kind__": "array",
            "name": name,
            "dtype": str(dtype),
            "shape": list(arr.shape),
        }


def encode_value(value: Any, store: ArrayStore) -> Any:
    """Encode supported runtime objects into manifest payloads."""
    Struct = _struct_type()

    if isinstance(value, Struct):
        fields_payload = {}
        for name, spec in value.fields().items():
            if not spec.should_serialize:
                continue
            fields_payload[name] = encode_value(getattr(value, name), store)
        return {
            "__kind__": "struct",
            "type": class_ref(type(value)),
            "fields": fields_payload,
        }

    spec = None
    ref = class_ref(type(value))
    try:
        spec = resolve_pytree_spec(ref)
    except SerializationError:
        spec = None

    if spec is not None and not isinstance(value, Struct):
        if spec.serializer is not None:
            return {
                "__kind__": "registered-pytree",
                "type": ref,
                "mode": "serializer",
                "payload": encode_value(spec.serializer(value), store),
            }

        children, aux = spec.flatten(value)
        return {
            "__kind__": "registered-pytree",
            "type": ref,
            "mode": "flatten",
            "children": [encode_value(v, store) for v in children],
            "aux": encode_value(aux, store),
        }

    if isinstance(value, (Array, np.ndarray)):
        return store.put(value)
    if isinstance(value, np.generic):
        return {
            "__kind__": "numpy-scalar",
            "dtype": str(value.dtype),
            "value": value.item(),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, complex):
        return {"__kind__": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return {"__kind__": "path", "value": str(value)}
    if isinstance(value, tuple):
        return {"__kind__": "tuple", "items": [encode_value(v, store) for v in value]}
    if isinstance(value, list):
        return {"__kind__": "list", "items": [encode_value(v, store) for v in value]}
    if isinstance(value, set):
        return {"__kind__": "set", "items": [encode_value(v, store) for v in value]}
    if isinstance(value, frozenset):
        return {
            "__kind__": "frozenset",
            "items": [encode_value(v, store) for v in value],
        }
    if isinstance(value, Mapping):
        return {
            "__kind__": "dict",
            "items": [
                [encode_value(k, store), encode_value(v, store)]
                for k, v in value.items()
            ],
        }
    raise SerializationError(
        f"Cannot serialize value of type {type(value).__name__}. "
        "Use serialize=False for this field, register a pytree adapter, or convert it to a supported type."
    )


def decode_value(
    payload: Any,
    arrays: Mapping[str, np.ndarray],
    *,
    expected_cls: type[Any] | None = None,
) -> Any:
    """Decode manifest payloads and array bundle into runtime objects."""
    if payload is None or isinstance(payload, (bool, int, float, str)):
        return payload
    if isinstance(payload, list):
        if expected_cls is not None:
            raise SerializationError(
                f"Expected {expected_cls.__name__}, but the serialized payload is a list."
            )
        return [decode_value(v, arrays) for v in payload]
    if not isinstance(payload, dict):
        raise SerializationError(
            f"Malformed serialized payload of type {type(payload).__name__}."
        )
    kind = payload.get("__kind__")
    if kind == "array":
        name = payload["name"]
        if name not in arrays:
            raise SerializationError(f"Missing array payload {name!r} in archive.")
        arr = arrays[name]
        expected_shape = tuple(payload.get("shape", []))
        expected_dtype = payload.get("dtype")
        if "shape" in payload and tuple(arr.shape) != expected_shape:
            raise SerializationError(
                f"Array {name!r} shape mismatch: expected {expected_shape}, found {arr.shape}."
            )
        if expected_dtype and str(arr.dtype) != expected_dtype:
            raise SerializationError(
                f"Array {name!r} dtype mismatch: expected {expected_dtype}, found {arr.dtype}."
            )
        return jax.device_put(arr)
    if kind == "numpy-scalar":
        return np.array(payload["value"], dtype=np.dtype(payload["dtype"])).item()
    if kind == "complex":
        return complex(payload["real"], payload["imag"])
    if kind == "path":
        return Path(payload["value"])
    if kind == "tuple":
        return tuple(decode_value(v, arrays) for v in payload["items"])
    if kind == "list":
        return [decode_value(v, arrays) for v in payload["items"]]
    if kind == "set":
        return {decode_value(v, arrays) for v in payload["items"]}
    if kind == "frozenset":
        return frozenset(decode_value(v, arrays) for v in payload["items"])
    if kind == "dict":
        return {
            decode_value(k, arrays): decode_value(v, arrays)
            for k, v in payload["items"]
        }
    if kind == "registered-pytree":
        spec = resolve_pytree_spec(payload["type"])
        mode = payload.get("mode", "flatten")
        if mode == "serializer":
            if spec.deserializer is None:
                raise SerializationError(
                    f"Registered type {payload['type']!r} requires a deserializer to load serializer payloads."
                )
            decoded_payload = decode_value(payload["payload"], arrays)
            return spec.deserializer(decoded_payload)

        children = [decode_value(v, arrays) for v in payload["children"]]
        aux = decode_value(payload["aux"], arrays)
        return spec.unflatten(aux, children)
    if kind == "struct":
        cls = resolve_class(payload["type"])
        if expected_cls is not None and not issubclass(cls, expected_cls):
            raise SerializationError(
                f"Expected {expected_cls.__name__}, found serialized {cls.__name__}."
            )

        if not hasattr(cls, "fields") or not hasattr(cls, "__struct_unflatten__"):
            raise SerializationError(
                f"Class {cls.__name__} is not a Struct-compatible type in this runtime."
            )

        values: dict[str, Any] = {}
        encoded_fields = payload["fields"]
        for name, spec in cls.fields().items():
            if spec.is_derived:
                continue
            if name in encoded_fields:
                values[name] = decode_value(encoded_fields[name], arrays)
            elif spec.has_default:
                values[name] = spec.make_default()
            elif not spec.should_serialize:
                raise SerializationError(
                    f"Field {name!r} was not serialized and has no default/default_factory."
                )
            else:
                raise SerializationError(
                    f"Missing serialized field {name!r} for {cls.__name__}."
                )
        return cls.__struct_unflatten__(values)
    raise SerializationError(f"Unknown serialized payload kind: {kind!r}")


_CURRENT_VERSION = 1


def _migrate(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Apply version-to-version schema migrations and return the updated payload.

    Each version bump should add an ``elif version == N`` branch here that
    transforms the payload in-place and bumps ``version`` to ``N + 1``.
    The function always returns a payload at ``_CURRENT_VERSION``.
    """
    return payload


def to_state_dict_impl(obj: Any, *, materialize: bool = True) -> dict[str, Any]:
    """Internal ``to_state_dict`` implementation."""
    store = ArrayStore(materialize=materialize)
    manifest = encode_value(obj, store)
    arrays = {
        name: {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
        for name, arr in store.arrays.items()
    }
    return {
        "version": 1,
        "manifest": manifest,
        "arrays": arrays,
        "array_data": store.arrays,
    }


def validate_array_manifest(
    payload: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    """Validate array names/dtypes/shapes expected by a manifest."""
    expected = payload.get("arrays", {})
    if not isinstance(expected, Mapping):
        raise SerializationError("Malformed 'arrays' section in state payload.")

    missing = [name for name in expected if name not in arrays]
    if missing:
        raise SerializationError(
            f"Missing array payload(s): {', '.join(sorted(missing))}"
        )

    for name, expected_meta in expected.items():
        if not isinstance(expected_meta, Mapping):
            continue
        arr = arrays[name]
        shape = tuple(expected_meta.get("shape", []))
        dtype = expected_meta.get("dtype")
        if "shape" in expected_meta and tuple(arr.shape) != shape:
            raise SerializationError(
                f"Array {name!r} shape mismatch: expected {shape}, found {arr.shape}."
            )
        if dtype and str(arr.dtype) != dtype:
            raise SerializationError(
                f"Array {name!r} dtype mismatch: expected {dtype}, found {arr.dtype}."
            )


def from_state_dict_impl(
    payload: Mapping[str, Any], *, expected_cls: type[Any] | None = None
) -> Any:
    """Internal ``from_state_dict`` implementation."""
    version = payload.get("version")
    if not isinstance(version, int) or version < 1 or version > _CURRENT_VERSION:
        raise SerializationError(
            f"Unsupported state dict version: {version!r}. "
            f"Expected 1..{_CURRENT_VERSION}."
        )
    payload = _migrate(payload)
    arrays = payload.get("array_data")
    if arrays is None:
        raise SerializationError("State dict is missing 'array_data'.")

    if not isinstance(arrays, Mapping):
        raise SerializationError("State dict 'array_data' must be a mapping.")

    np_arrays = {k: np.asarray(v) for k, v in arrays.items()}
    validate_array_manifest(payload, np_arrays)
    result = decode_value(payload["manifest"], np_arrays, expected_cls=expected_cls)
    if expected_cls is not None and not isinstance(result, expected_cls):
        raise SerializationError(
            f"Expected {expected_cls.__name__}, found serialized {type(result).__name__}."
        )
    return result


def write_bundle(
    path: Path, manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    """Write manifest and compressed arrays into a directory."""
    with path.joinpath("manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    with path.joinpath("arrays.npz").open("wb") as file:
        np.savez_compressed(file, **arrays)


def _write_export(path: Path, state: dict[str, Any], *, overwrite: bool) -> None:
    """Stage complete bundles and restore the previous directory on rename failure."""
    if (path.exists() or path.is_symlink()) and not overwrite:
        raise NeuralqxFileExistsError(f"Path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=path.parent, prefix=".tmp_struct_"))
    backup = staging / "previous"
    preserve_backup = False
    try:
        bundle = staging / "bundle"
        bundle.mkdir()
        manifest = {key: value for key, value in state.items() if key != "array_data"}
        write_bundle(bundle, manifest, state["array_data"])
        if path.suffix == ".zip":
            archive = staging / "bundle.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for item in sorted(bundle.iterdir()):
                    zf.write(item, arcname=item.name)
            bundle = archive
        if (path.exists() or path.is_symlink()) and not overwrite:
            raise NeuralqxFileExistsError(f"Path already exists: {path}")
        if path.suffix == ".zip":
            os.replace(bundle, path)
        else:
            if path.exists() or path.is_symlink():
                os.replace(path, backup)
            try:
                os.replace(bundle, path)
            except Exception:
                if backup.exists() or backup.is_symlink():
                    try:
                        os.replace(backup, path)
                    except Exception as exc:
                        preserve_backup = True
                        raise NeuralqxOSError(
                            f"Could not restore the previous bundle. It remains at {backup}."
                        ) from exc
                raise
    finally:
        if not preserve_backup:
            shutil.rmtree(staging, ignore_errors=True)


def export_impl(obj: Any, path: str | Path, *, overwrite: bool = False) -> Path:
    """Gather global arrays with JAX and write the bundle on process zero."""
    path = _collective_call(lambda: Path(path), context="Struct export path")
    state = _collective_call(
        lambda: to_state_dict_impl(obj, materialize=False),
        context="Struct export preparation",
    )
    arrays = state["array_data"]
    global_arrays = {
        name: isinstance(arr, jax.Array) and not arr.is_fully_addressable
        for name, arr in arrays.items()
    }
    _agree(
        {
            "path": str(path),
            "overwrite": overwrite,
            "manifest": {
                key: value for key, value in state.items() if key != "array_data"
            },
            "global_arrays": global_arrays,
        },
        context="Struct export metadata",
    )
    for name, arr in arrays.items():
        if global_arrays[name]:
            from jax.experimental import multihost_utils

            _collective_call(arr.block_until_ready, context="Struct array readiness")
            arr = multihost_utils.process_allgather(arr, tiled=True)
        arrays[name] = _collective_call(
            lambda: np.asarray(arr), context="Struct array transfer"
        )
    _collective_call(
        lambda: (
            _write_export(path, state, overwrite=overwrite)
            if jax.process_index() == 0
            else None
        ),
        context="Struct export",
    )
    return path


def _read_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise NeuralqxFileNotFoundError(path)
    if path.is_file() and path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("manifest.json", "r") as file:
                manifest = json.load(io.TextIOWrapper(file, encoding="utf-8"))
            with zf.open("arrays.npz", "r") as file:
                with np.load(io.BytesIO(file.read()), allow_pickle=False) as npz:
                    arrays = {key: npz[key] for key in npz.files}
    else:
        with (path / "manifest.json").open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        with np.load(path / "arrays.npz", allow_pickle=False) as npz:
            arrays = {key: npz[key] for key in npz.files}
    if not isinstance(manifest, Mapping):
        raise SerializationError("Malformed manifest payload.")
    metadata = manifest.get("arrays", {})
    if not isinstance(metadata, Mapping):
        raise SerializationError("Malformed 'arrays' section in state payload.")
    # NumPy stores extension dtypes such as bfloat16 as void data in NPZ files.
    # Recover their numerical dtype from the existing manifest, without a cast.
    for name, meta in metadata.items():
        arr = arrays.get(name)
        if arr is not None and arr.dtype.kind == "V" and isinstance(meta, Mapping):
            dtype = np.dtype(meta["dtype"])
            if dtype.itemsize == arr.dtype.itemsize and jax.dtypes.issubdtype(
                dtype, np.number
            ):
                arrays[name] = arr.view(dtype)
    return {**manifest, "array_data": arrays}


def load_impl(path: str | Path, *, expected_cls: type[Any] | None = None) -> Any:
    """Read on process zero and broadcast host buffers without dtype conversion."""
    path = _collective_call(lambda: Path(path), context="Struct load path")
    _agree(
        {
            "path": str(path),
            "expected_cls": class_ref(expected_cls) if expected_cls else None,
        },
        context="Struct load metadata",
    )
    state = _collective_call(
        lambda: _read_bundle(path) if jax.process_index() == 0 else None,
        context="Struct load",
    )
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        manifest = _broadcast_json(
            {key: value for key, value in state.items() if key != "array_data"}
            if state is not None
            else None
        )
        # Describe the actual buffers separately so malformed manifest metadata
        # is rejected by the same decoder on every process, after broadcasting.
        metadata = _broadcast_json(
            {
                name: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                for name, arr in state["array_data"].items()
            }
            if state is not None
            else None
        )
        arrays = {}
        for name, meta in metadata.items():
            arr = _collective_call(
                lambda: (
                    np.array(state["array_data"][name], copy=True, order="C")
                    if state is not None
                    else np.zeros(meta["shape"], dtype=meta["dtype"])
                ),
                context="Struct load buffer allocation",
            )
            # uint8 transport preserves int64 and float64 even with JAX X64 off.
            buffer = arr.reshape(-1).view(np.uint8)
            if buffer.size:
                buffer = multihost_utils.broadcast_one_to_all(buffer)
            arrays[name] = np.asarray(buffer).view(arr.dtype).reshape(arr.shape)
        state = {**manifest, "array_data": arrays}
    return _collective_call(
        lambda: from_state_dict_impl(state, expected_cls=expected_cls),
        context="Struct reconstruction",
    )
