# Copyright (c) 2026 The neuraLQX and nkDSL Authors - All rights reserved.
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


"""Iterator clause subpackage for fluent DSL extension points."""

from .abstractions import AbstractIteratorClause
from .defaults import ensure_default_iterator_clause_registrations
from .dispatch import apply_iterator_clause
from .registry import available_iterator_clause_names
from .registry import register_iterator_clause
from .registry import resolve_iterator_clause
from .types import coerce_iterator_spec

__all__ = [
    "AbstractIteratorClause",
    "apply_iterator_clause",
    "available_iterator_clause_names",
    "coerce_iterator_spec",
    "ensure_default_iterator_clause_registrations",
    "register_iterator_clause",
    "resolve_iterator_clause",
]
