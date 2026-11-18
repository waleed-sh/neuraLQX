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


"""Construction-time registration of wavefunction projectors."""

import inspect
import keyword
import re
from functools import wraps
from threading import RLock

from neuralqx.utils.struct import StructABCMeta

_REGISTRY = {}
_LOCK = RLock()
_RESERVED = {
    "Projector",
    "DiagonalProjector",
    "RowProjector",
    "NonVacuum",
    "ProjectedNNX",
    "ProjectedLinen",
    "ProjectedCallable",
    "field",
    "register_projector",
    "get_projector",
    "available_projectors",
    "base",
    "registry",
    "wrappers",
    "vacuum",
    "utils",
}


def _default_name(cls):
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", cls.__name__)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def register_projector(cls=None, *, name=None, replace=False):
    """Register a projector class and return it unchanged.

    Concrete subclasses register automatically under their snake-case class
    name, or the explicit ``name=`` supplied in the class declaration. Use
    this function to register an additional alias. Names become decorators
    at ``neuralqx.nn.projectors.<name>``. Replacing a name is explicit so an
    import cannot silently change another model's construction.
    """

    def register(projector_cls):
        from .base import Projector

        if not isinstance(projector_cls, type) or not issubclass(
            projector_cls, Projector
        ):
            raise TypeError("Only Projector subclasses can be registered.")
        if inspect.isabstract(projector_cls):
            raise TypeError("An abstract projector cannot be registered.")
        key = _default_name(projector_cls) if name is None else name
        if (
            not isinstance(key, str)
            or not key.isidentifier()
            or keyword.iskeyword(key)
            or key.startswith("_")
            or key in _RESERVED
        ):
            raise ValueError(f"Invalid or reserved projector name {key!r}.")
        with _LOCK:
            old = _REGISTRY.get(key)
            if old is not None and old is not projector_cls and not replace:
                raise ValueError(f"Projector {key!r} is already registered.")
            _REGISTRY[key] = projector_cls
        return projector_cls

    return register if cls is None else register(cls)


def get_projector(name, **kwargs):
    """Construct a registered projector with the supplied field values."""
    with _LOCK:
        try:
            cls = _REGISTRY[name]
        except KeyError:
            raise KeyError(
                f"Unknown projector {name!r}. Available: {available_projectors()}"
            ) from None
    return cls(**kwargs)


def available_projectors():
    """Return registered names in alphabetical order."""
    with _LOCK:
        return tuple(sorted(_REGISTRY))


def projector_decorator(name):
    """Resolve a registry entry into a configured or bare model decorator."""
    with _LOCK:
        cls = _REGISTRY[name]

    @wraps(cls, updated=())
    def decorate(model=None, **kwargs):
        projector = cls(**kwargs)
        return projector if model is None else projector(model)

    return decorate


class ProjectorMeta(StructABCMeta):
    """Finish the immutable pytree schema before registering concrete classes."""

    def __new__(
        mcls,
        class_name,
        bases,
        namespace,
        *,
        name=None,
        register=True,
        replace=False,
        **kwargs,
    ):
        return super().__new__(mcls, class_name, bases, namespace, **kwargs)

    def __init__(
        cls,
        class_name,
        bases,
        namespace,
        *,
        name=None,
        register=True,
        replace=False,
        **kwargs,
    ):
        super().__init__(class_name, bases, namespace, **kwargs)
        if register and not inspect.isabstract(cls):
            register_projector(cls, name=name, replace=replace)
