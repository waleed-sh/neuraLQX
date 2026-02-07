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
Utilities for controlling public module namespaces and export behavior.
"""

from types import ModuleType
import sys
from typing import Iterable


def prune_module_attributes(
    module_name: str,
    *,
    allow: Iterable[str] = (),
) -> None:
    """
    Remove imported submodules from a module's public namespace.

    This function mutates the target module by removing attributes that correspond to imported
    submodules, except those explicitly allowed.

    :param module_name: Fully-qualified module name to prune.
    :param allow: Attribute names that should be preserved even if they are submodules.
    """

    module = sys.modules.get(module_name)
    if module is None:
        raise KeyError(f"Module '{module_name}' is not loaded.")

    keep = set(allow)

    for attr in list(vars(module)):
        if attr.startswith("_") or attr in keep:
            continue

        value = getattr(module, attr)
        if isinstance(value, ModuleType):
            delattr(module, attr)
            fq_name = f"{module_name}.{attr}"
            sys.modules.pop(fq_name, None)


def define_public_api(module_name: str) -> None:
    """
    Populate a module's ``__all__`` attribute based on its public symbols.

    All non-private attributes currently defined on the module are added to ``__all__`` and
    explicitly re-bound to the module namespace.
    """

    module = sys.modules.get(module_name)
    if module is None:
        raise KeyError(f"Module '{module_name}' is not loaded.")

    public = []

    for name, obj in vars(module).items():
        if not name.startswith("_"):
            public.append(name)
            setattr(module, name, obj)

    module.__all__ = sorted(set(public))
