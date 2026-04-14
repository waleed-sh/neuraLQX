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


"""
Symbolic selector helpers for the declarative symbolic operator DSL.

Selectors are lightweight immutable facades used at build time to construct
IR symbol references pointing to site quantum numbers, site indices, and
free named parameters.
"""

from __future__ import annotations

from typing import Any

from neuralqx.experimental.operators.symbolic.ir.expressions import AmplitudeExpr


def _site_symbol(label: str, field: str, namespace: str = "site") -> AmplitudeExpr:
    """Returns a symbolic AmplitudeExpr for one site field."""
    return AmplitudeExpr.symbol(f"{namespace}:{label}:{field}")


class SiteSelector:
    """
    Symbolic selector for one Hilbert-space site iterator.

    SiteSelector is created by :func:`site` and used inside DSL predicates,
    amplitude rules, and update programs.  Attribute access on a selector
    returns symbolic :class:`~neuralqx.experimental.operators.symbolic.ir.AmplitudeExpr`
    nodes that are resolved by the compiler at lowering time.

    Args:
        label: Iterator label bound by ``for_each_site(label)``.
    """

    __slots__ = ("_label", "_namespace")

    def __init__(self, label: str, namespace: str = "site") -> None:
        normalized = str(label).strip()
        if not normalized:
            raise ValueError("SiteSelector label must be a non-empty string.")
        ns = str(namespace).strip()
        if not ns:
            raise ValueError("SiteSelector namespace must be a non-empty string.")
        self._label = normalized
        self._namespace = ns

    @property
    def label(self) -> str:
        """Returns selector label."""
        return self._label

    @property
    def namespace(self) -> str:
        """Returns selector namespace."""
        return self._namespace

    @property
    def value(self) -> AmplitudeExpr:
        """Returns symbolic expression for the quantum number at this site."""
        return _site_symbol(self._label, "value", self._namespace)

    @property
    def index(self) -> AmplitudeExpr:
        """Returns symbolic expression for the integer index of this site."""
        return _site_symbol(self._label, "index", self._namespace)

    def as_site_ref(self) -> AmplitudeExpr:
        """
        Returns the site-index expression used in update op site parameters.

        Important:
            Update-program site refs must always resolve against the source-site
            namespace, even if this selector is an emitted-state selector.
        """
        return _site_symbol(self._label, "index", "site")

    def attr(self, name: str) -> AmplitudeExpr:
        """
        Returns a symbolic expression for an arbitrary site attribute.

        Args:
            name: Attribute field name.

        Returns:
            Symbolic amplitude expression.
        """
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("SiteSelector attribute name must be non-empty.")
        return _site_symbol(self._label, normalized, self._namespace)

    def abs(self) -> AmplitudeExpr:
        """Returns ``|x[i]|`` as an amplitude expression."""
        return AmplitudeExpr.abs_(self.value)

    def __lt__(self, other: Any) -> Any:
        from neuralqx.experimental.operators.symbolic.ir.predicates import PredicateExpr

        return PredicateExpr.lt(self.value, other)

    def __le__(self, other: Any) -> Any:
        from neuralqx.experimental.operators.symbolic.ir.predicates import PredicateExpr

        return PredicateExpr.le(self.value, other)

    def __gt__(self, other: Any) -> Any:
        from neuralqx.experimental.operators.symbolic.ir.predicates import PredicateExpr

        return PredicateExpr.gt(self.value, other)

    def __ge__(self, other: Any) -> Any:
        from neuralqx.experimental.operators.symbolic.ir.predicates import PredicateExpr

        return PredicateExpr.ge(self.value, other)

    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        from neuralqx.experimental.operators.symbolic.ir.predicates import PredicateExpr

        return PredicateExpr.eq(self.value, other)

    def __ne__(self, other: Any) -> Any:  # type: ignore[override]
        from neuralqx.experimental.operators.symbolic.ir.predicates import PredicateExpr

        return PredicateExpr.ne(self.value, other)

    def __getattr__(self, name: str) -> AmplitudeExpr:
        """Delegates unknown attribute access to :meth:`attr`."""
        return self.attr(name)

    def __repr__(self) -> str:
        return f"SiteSelector(label={self._label!r}, namespace={self._namespace!r})"


#
#
#   Public factory functions


def site(label: str) -> SiteSelector:
    """
    Returns a symbolic site selector.

    Args:
        label: Iterator label bound by ``for_each_site(label)`` or
            ``for_each_pair(label_a, label_b)``.

    Returns:
        Site selector handle.

    Example:
        .. code-block:: python

            from neuralqx.experimental.operators.symbolic.dsl import site

            s = site("i")
            print(s.value)          # AmplitudeExpr, x[i]
            print(s.index)          # AmplitudeExpr, i
            print(s.value < 3)      # PredicateExpr, x[i] < 3
            print(s.value + 1)      # AmplitudeExpr, x[i] + 1
    """
    return SiteSelector(label, namespace="site")


def emitted(label: str) -> SiteSelector:
    """
    Returns a symbolic selector bound to the emitted or connected state ``x'``.

    Example:
        .. code-block:: python

            from neuralqx.experimental.operators.symbolic.dsl import emitted

            e = emitted("i")
            print(e.value)   # AmplitudeExpr, x'[i]
            print(e.index)   # AmplitudeExpr, i
    """
    return SiteSelector(label, namespace="emit")


def symbol(name: str) -> AmplitudeExpr:
    """
    Returns a free symbolic amplitude expression by name.

    Free symbols are not bound to any site iterator, they are resolved at
    operator-evaluation time from external parameter dictionaries.

    Args:
        name: Symbol name.

    Returns:
        Symbolic amplitude expression.
    """
    return AmplitudeExpr.symbol(name)


__all__ = [
    "SiteSelector",
    "site",
    "emitted",
    "symbol",
]
