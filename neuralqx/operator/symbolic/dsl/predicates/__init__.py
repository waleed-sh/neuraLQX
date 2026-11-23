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


"""Predicate clause subpackage for fluent DSL extension points."""

from .abstractions import AbstractPredicateClause
from .defaults import ensure_default_predicate_clause_registrations
from .dispatch import apply_predicate_clause
from .registry import available_predicate_clause_names
from .registry import register_predicate_clause
from .registry import resolve_predicate_clause

__all__ = [
    "AbstractPredicateClause",
    "apply_predicate_clause",
    "available_predicate_clause_names",
    "ensure_default_predicate_clause_registrations",
    "register_predicate_clause",
    "resolve_predicate_clause",
]
