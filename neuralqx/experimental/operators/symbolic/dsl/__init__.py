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


"""User-facing declarative symbolic operator DSL."""

from neuralqx.experimental.operators.symbolic._deprecation import (
    deprecate_symbolic_public_api,
)

from .context import ExpressionContext
from .op import DOperator

from .rewrite import Update
from .rewrite import affine
from .rewrite import identity
from .rewrite import permute
from .rewrite import scatter
from .rewrite import shift
from .rewrite import shift_mod
from .rewrite import swap
from .rewrite import write

from .selectors import SiteSelector
from .selectors import emitted
from .selectors import site
from .selectors import symbol

__all__ = [
    "DOperator",
    "Update",
    "affine",
    "identity",
    "permute",
    "scatter",
    "shift",
    "shift_mod",
    "swap",
    "write",
    "ExpressionContext",
    "site",
    "emitted",
    "symbol",
    "SiteSelector",
]

deprecate_symbolic_public_api(
    globals(),
)
