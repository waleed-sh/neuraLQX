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


import importlib
import pkgutil

from collections.abc import Iterable
from collections.abc import MutableMapping

from types import ModuleType
from typing import Any


def auto_export(
    package_name: str,
    namespace: MutableMapping[str, Any],
    *,
    modules: Iterable[str] | None = None,
    include_modules: bool = False,
) -> tuple[str, ...]:
    """Imports public names from submodules into a package namespace.

    Args:
        package_name: Fully qualified package name, for example
            ``"neuralqx.utils.typing"``.
        namespace: Package ``globals()`` dictionary to populate.
        modules: Optional iterable of child module names. When omitted, all
            non-private immediate modules in the package directory are used.
        include_modules: Whether imported module objects should also be exported
            under their short names.

    Returns:
        Public names exported into ``namespace`` in deterministic order.

    Raises:
        RuntimeError: If two modules try to export different objects under the
            same public name.
    """
    package = importlib.import_module(package_name)
    module_names = tuple(modules) if modules is not None else _discover_modules(package)

    exported: list[str] = []
    seen: set[str] = set()

    for module_name in module_names:
        module = importlib.import_module(f"{package_name}.{module_name}")
        if include_modules:
            _export_name(namespace, exported, seen, module_name, module)

        names = getattr(module, "__all__", None)
        if names is None:
            names = tuple(name for name in vars(module) if not name.startswith("_"))

        for name in names:
            _export_name(namespace, exported, seen, name, getattr(module, name))

    return tuple(exported)


def _discover_modules(package: ModuleType) -> tuple[str, ...]:
    paths = getattr(package, "__path__", None)
    if paths is None:
        return ()
    return tuple(
        info.name
        for info in sorted(pkgutil.iter_modules(paths), key=lambda item: item.name)
        if not info.ispkg and not info.name.startswith("_")
    )


def _export_name(
    namespace: MutableMapping[str, Any],
    exported: list[str],
    seen: set[str],
    name: str,
    value: Any,
) -> None:
    existing = namespace.get(name)
    if name in namespace and existing is not value:
        raise RuntimeError(f"Cannot auto-export duplicate public name {name!r}.")

    namespace[name] = value
    if name not in seen:
        exported.append(name)
        seen.add(name)


__all__ = [
    "auto_export",
]
