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


"""Explicit, data-only recipes for reconstructing checkpoint components."""

import base64
import dataclasses
import math
from collections.abc import Callable
from collections.abc import Mapping
from threading import RLock

import numpy as np

import jax
from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError

from .errors import ReconstructionError


@dataclasses.dataclass(frozen=True)
class ComponentCodec:
    """A versioned recipe whose implementation is supplied by installed code.

    The checkpoint stores the stable name, version and encoded configuration.
    Loading resolves that name only through this registry. It does not import
    arbitrary module paths or execute source code from the saved file.
    """

    name: str
    """Stable identifier recorded in checkpoint metadata."""

    cls: type
    """Exact Python type accepted by this recipe."""

    version: int
    """Configuration schema version expected by the decoder."""

    encode: Callable
    """Function returning constructor configuration from a live component."""

    decode: Callable
    """Function constructing the component from decoded configuration."""


_BY_NAME = {}
_BY_TYPE = {}
_READY = False
_LOCK = RLock()


def register_codec(cls, *, name, encode, decode, version=1, replace=False):
    """Register a reconstruction recipe for a custom model, sampler or space.

    ``encode(object)`` returns a configuration made of supported values.
    ``decode(config)`` builds the component without its learned arrays, which
    are restored separately. Register the same codec before saving and loading.
    For example, a Linen module can use ``encode=lambda m: {"width": m.width}``
    and ``decode=lambda cfg: MyModel(**cfg)``. Models need not register a recipe
    when loading into an already constructed compatible variational state.
    """
    _initialize()
    return _register_codec(
        cls, name=name, encode=encode, decode=decode, version=version, replace=replace
    )


def _register_codec(cls, *, name, encode, decode, version=1, replace=False):
    if not isinstance(cls, type) or not isinstance(name, str) or not name:
        raise NeuralqxTypeError(
            "A component codec requires a Python type and a nonempty name."
        )
    if (
        not callable(encode)
        or not callable(decode)
        or type(version) is not int
        or version < 1
    ):
        raise NeuralqxTypeError(
            "Codec functions must be callable and its version a positive integer."
        )
    with _LOCK:
        if not replace and (name in _BY_NAME or cls in _BY_TYPE):
            raise NeuralqxValueError(
                f"A component codec already exists for {name!r} or {cls.__name__}."
            )
        old = _BY_TYPE.get(cls)
        if old is not None and old.cls is cls:
            _BY_NAME.pop(old.name, None)
            for registered_type, registered in tuple(_BY_TYPE.items()):
                if registered is old:
                    _BY_TYPE.pop(registered_type)
        old = _BY_NAME.get(name)
        if old is not None:
            for registered_type, registered in tuple(_BY_TYPE.items()):
                if registered is old:
                    _BY_TYPE.pop(registered_type)
        codec = ComponentCodec(name, cls, version, encode, decode)
        _BY_NAME[name] = _BY_TYPE[cls] = codec
    return codec


def _initialize():
    global _READY
    with _LOCK:
        if not _READY:
            from .builtins import install_codecs

            install_codecs()
            _READY = True


def encode_config(value):
    """Encode small static configuration without serializing executable objects."""
    _initialize()
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        return (
            value if math.isfinite(value) else {"kind": "float", "value": repr(value)}
        )
    if isinstance(value, complex):
        return {
            "kind": "complex",
            "real": encode_config(value.real),
            "imag": encode_config(value.imag),
        }
    if isinstance(value, np.dtype) or (
        isinstance(value, type) and issubclass(value, np.generic)
    ):
        return {"kind": "dtype", "name": str(np.dtype(value))}
    if isinstance(value, type):
        try:
            dtype = np.dtype(value)
        except TypeError:
            dtype = None
        if dtype is not None and not dtype.hasobject:
            return {"kind": "dtype", "name": str(dtype)}
    if isinstance(value, bytes):
        return {"kind": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        is_jax = isinstance(value, jax.Array)
        weak_type = bool(getattr(value, "weak_type", False))
        if isinstance(value, jax.Array) and not value.is_fully_addressable:
            if not value.is_fully_replicated:
                raise ReconstructionError(
                    "Sharded data belongs in state arrays, not a constructor recipe."
                )
            value = value.addressable_shards[0].data
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ReconstructionError(
                "Object arrays cannot be stored in a constructor recipe."
            )
        return {
            "kind": "array",
            "jax": is_jax,
            "scalar": isinstance(value, np.generic),
            "weak_type": weak_type,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": encode_config(array.tolist()),
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "items": [encode_config(v) for v in value],
        }
    if isinstance(value, Mapping):
        from flax.core import FrozenDict

        return {
            "kind": "frozen_mapping" if isinstance(value, FrozenDict) else "mapping",
            "items": [[encode_config(k), encode_config(v)] for k, v in value.items()],
        }
    codec = _BY_TYPE.get(type(value))
    if codec is not None:
        return {
            "kind": "component",
            "name": codec.name,
            "version": codec.version,
            "config": encode_config(codec.encode(value)),
        }
    if callable(value):
        from neuralqx.nn.activations import activation_spec
        from neuralqx.nn.activations import available_activations
        from neuralqx.nn.activations import get_activation

        spec = activation_spec(value)
        if spec.name in available_activations() and get_activation(spec.name) is value:
            return {
                "kind": "activation",
                "name": spec.name,
                "elementwise": spec.elementwise,
                "supports_complex": spec.supports_complex,
            }
        split = getattr(value, "_neuralqx_reim_config", None)
        if split is not None:
            return {
                "kind": "reim",
                "function": encode_config(split[0]),
                "options": encode_config(split[1]),
            }
    raise ReconstructionError(
        f"No reconstruction codec for {type(value).__module__}.{type(value).__qualname__}. Supply the component when loading or register_codec for its type."
    )


def decode_config(value):
    """Decode configuration using only explicitly registered recipe implementations."""
    _initialize()
    if value is None or type(value) in (bool, str, int, float):
        return value
    if not isinstance(value, dict):
        raise ReconstructionError("Malformed component configuration.")
    kind = value.get("kind")
    if kind == "float":
        return float(value["value"])
    if kind == "complex":
        return complex(decode_config(value["real"]), decode_config(value["imag"]))
    if kind == "dtype":
        return np.dtype(value["name"])
    if kind == "bytes":
        return base64.b64decode(value["data"], validate=True)
    if kind in ("tuple", "list"):
        result = [decode_config(v) for v in value["items"]]
        return tuple(result) if kind == "tuple" else result
    if kind in ("mapping", "frozen_mapping"):
        result = {decode_config(k): decode_config(v) for k, v in value["items"]}
        if kind == "frozen_mapping":
            from flax.core import FrozenDict

            return FrozenDict(result)
        return result
    if kind == "array":
        array = np.asarray(decode_config(value["data"]), dtype=np.dtype(value["dtype"]))
        if list(array.shape) != value["shape"] or array.dtype.hasobject:
            raise ReconstructionError("Malformed array in component configuration.")
        if value.get("jax"):
            from .trees import replicated_sharding
            from .trees import restore_weak_type

            if jax.dtypes.canonicalize_dtype(array.dtype) != array.dtype:
                raise ReconstructionError(
                    f"Enable jax_enable_x64 to restore {array.dtype} configuration arrays."
                )
            array = jax.make_array_from_callback(
                array.shape, replicated_sharding(), lambda index: array[index]
            )
            array = restore_weak_type(array, value.get("weak_type", False))
        return array[()] if value.get("scalar") else array
    if kind == "component":
        codec = _BY_NAME.get(value["name"])
        if codec is None or codec.version != value["version"]:
            raise ReconstructionError(
                f"Missing codec or unsupported codec version for {value['name']!r}."
            )
        return codec.decode(decode_config(value["config"]))
    if kind == "activation":
        from neuralqx.nn.activations import activation_spec

        spec = activation_spec(value["name"])
        if (
            spec.elementwise != value["elementwise"]
            or spec.supports_complex != value["supports_complex"]
        ):
            raise ReconstructionError(
                f"The activation contract for {spec.name!r} has changed."
            )
        return spec.function
    if kind == "reim":
        from neuralqx.nn import reim

        return reim(decode_config(value["function"]), **decode_config(value["options"]))
    raise ReconstructionError(f"Unknown configuration entry {kind!r}.")


def component_description(value):
    """Describe a component even when automatic reconstruction is unavailable."""
    description = {"class": f"{type(value).__module__}.{type(value).__qualname__}"}
    if callable(value) and hasattr(value, "__qualname__"):
        description["definition"] = f"{value.__module__}.{value.__qualname__}"
    try:
        description["recipe"] = encode_config(value)
    except ReconstructionError as error:
        description["recipe"] = None
        description["requires_template"] = error.message
    return description
