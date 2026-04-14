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

from __future__ import annotations

from typing import Any
from typing import Iterable
from typing import MutableMapping

from neuralqx import cfg
from neuralqx.utils.deprecation import deprecate_public_api

_NKDSL_DOCS_URL = "https://nkdsl.readthedocs.io/en/latest/?badge=latest"
_SYMBOLIC_INTERNAL_MODULE_PREFIXES = ("neuralqx.experimental.operators.symbolic",)


def _symbolic_deprecation_reason(target: str) -> str:
    return (
        f"`{target}` belongs to the experimental symbolic API.\n\n"
        "The experimental symbolic API is deprecated in favour of nkDSL, "
        "the independent DSL for both NetKet and neuraLQX. nkDSL offers a larger range of "
        "capabilities than what is currently offered by the experimental symbolic API of "
        "neuraLQX.\n"
        "This symbolic API will be removed in the next release.\n\n"
        "Please migrate to nkDSL and see its docs to get started:\n"
        f"{_NKDSL_DOCS_URL}"
    )


def deprecate_symbolic_public_api(
    namespace: MutableMapping[str, Any],
    exports: Iterable[str] | None = None,
    *,
    module_name: str | None = None,
    warn_on_module_import: bool = False,
) -> None:
    """
    Applies symbolic-specific messaging to the generic public API deprecator.
    """

    # Module-level warnings are useful for users, but under the strict test
    # warning policy they can turn into collection-time errors before tests run.
    effective_warn_on_module_import = warn_on_module_import and not cfg.TESTING

    deprecate_public_api(
        namespace=namespace,
        exports=exports,
        module_name=module_name,
        reason=_symbolic_deprecation_reason,
        warn_on_module_import=effective_warn_on_module_import,
        internal_module_prefixes=_SYMBOLIC_INTERNAL_MODULE_PREFIXES,
    )
