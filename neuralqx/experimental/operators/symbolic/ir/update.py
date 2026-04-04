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


"""Typed site-update IR nodes for symbolic operators."""

from __future__ import annotations

import dataclasses
from typing import Any

_UPDATE_OP_KINDS: frozenset[str] = frozenset(
    {
        "write_site",  # x'[i] = v
        "shift_site",  # x'[i] = x[i] + delta
        "shift_mod_site",  # x'[i] = wrapped(x[i] + delta) using Hilbert local_states
        "swap_sites",  # x'[i], x'[j] = x[j], x[i]
        "invalidate_branch",  # mark this branch as zero amplitude
        "affine_site",  # x'[i] = scale * x[i] + bias
        "permute_sites",  # cyclic rotation
        "scatter",  # bulk write to static flat indices
        "cond_branch",  # conditional update
    }
)


@dataclasses.dataclass(frozen=True, repr=False)
class UpdateOp:
    """
    One primitive site-update operation.

    Attributes:
        kind: Update operation kind (see ``_UPDATE_OP_KINDS``).
        params: Deterministic parameter tuple ``((key, value), ...)``.
            Values are :class:`~neuralqx.experimental.operators.symbolic.ir.expressions.AmplitudeExpr`
            nodes, plain integers, or nested structures depending on ``kind``.
    """

    kind: str
    params: tuple = dataclasses.field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.kind not in _UPDATE_OP_KINDS:
            raise ValueError(
                f"Unsupported update operation kind: {self.kind!r}. "
                f"Allowed: {sorted(_UPDATE_OP_KINDS)}."
            )

    @classmethod
    def from_mapping(
        cls,
        *,
        kind: str,
        params: dict[str, Any] | None = None,
    ) -> "UpdateOp":
        """Builds an update op from mapping-form parameters."""
        if params is None:
            params_tuple: tuple = ()
        else:
            params_tuple = tuple(sorted(params.items()))
        return cls(kind=kind, params=params_tuple)

    def get(self, key: str, *, default: Any = None) -> Any:
        """Returns one parameter value by key."""
        for k, v in self.params:
            if k == key:
                return v
        return default

    def __str__(self) -> str:
        return _render_update_op(self)

    def __repr__(self) -> str:
        return f"UpdateOp(kind={self.kind!r}, params={self.params!r})"


@dataclasses.dataclass(frozen=True, repr=False)
class UpdateProgram:
    """
    Ordered immutable sequence of site-update operations.

    Attributes:
        ops: Ordered update-operation tuple.
    """

    ops: tuple = dataclasses.field(default_factory=tuple)

    def append(self, op: UpdateOp) -> "UpdateProgram":
        """Returns a new program with one appended operation."""
        return UpdateProgram(ops=self.ops + (op,))

    def extend(self, other: "UpdateProgram") -> "UpdateProgram":
        """Returns a new program with another program appended."""
        return UpdateProgram(ops=self.ops + other.ops)

    @property
    def op_count(self) -> int:
        """Returns number of update operations."""
        return len(self.ops)

    def has_invalidate(self) -> bool:
        """Returns whether this program contains any invalidate_branch op."""
        return any(op.kind == "invalidate_branch" for op in self.ops)

    def __str__(self) -> str:
        if not self.ops:
            return "identity"
        return "; ".join(str(op) for op in self.ops)

    def __repr__(self) -> str:
        return f"UpdateProgram(op_count={self.op_count})"


def _render_update_op(op: "UpdateOp") -> str:
    """Renders one UpdateOp as a pseudocode assignment string."""
    from .expressions import AmplitudeExpr, _render_amplitude
    from .predicates import PredicateExpr, _render_predicate

    def _amp(v: Any) -> str:
        if isinstance(v, AmplitudeExpr):
            return _render_amplitude(v)
        return repr(v)

    def _pred(v: Any) -> str:
        if isinstance(v, PredicateExpr):
            return _render_predicate(v)
        return repr(v)

    kind = op.kind
    get = lambda k, d=None: op.get(k, default=d)

    if kind == "write_site":
        return f"x'[{_amp(get('site'))}] = {_amp(get('value'))}"

    if kind == "shift_site":
        s = _amp(get("site"))
        d = _amp(get("delta"))
        return f"x'[{s}] = (x[{s}] + {d})"

    if kind == "shift_mod_site":
        s = _amp(get("site"))
        d = _amp(get("delta"))
        return f"x'[{s}] = wrap(x[{s}] + {d})"

    if kind == "swap_sites":
        a = _amp(get("site_a"))
        b = _amp(get("site_b"))
        return f"x'[{a}], x'[{b}] = x[{b}], x[{a}]"

    if kind == "affine_site":
        s = _amp(get("site"))
        sc = _amp(get("scale"))
        bi = _amp(get("bias"))
        return f"x'[{s}] = ({sc} * x[{s}] + {bi})"

    if kind == "permute_sites":
        sites = get("sites") or ()
        ss = [_amp(s) for s in sites]
        n = len(ss)
        lhs = ", ".join(f"x'[{s}]" for s in ss)
        rhs = ", ".join(f"x[{ss[(i + 1) % n]}]" for i in range(n))
        return f"{lhs} = {rhs}"

    if kind == "scatter":
        indices = get("flat_indices") or ()
        values = get("values") or ()
        parts = [f"x'[{idx}] = {_amp(v)}" for idx, v in zip(indices, values)]
        return "; ".join(parts) if parts else "scatter()"

    if kind == "invalidate_branch":
        reason = get("reason")
        return f"invalidate  ; {reason}" if reason else "invalidate"

    if kind == "cond_branch":
        pred = get("predicate")
        then_ops = get("then_ops") or ()
        else_ops = get("else_ops") or ()
        pred_str = _pred(pred)
        then_str = "; ".join(_render_update_op(o) for o in then_ops) or "identity"
        else_str = "; ".join(_render_update_op(o) for o in else_ops) or "identity"
        return f"cond({pred_str}) {{ {then_str} }} | {{ {else_str} }}"

    # Fallback
    return repr(op)


def _collect_free_symbols_from_ops(ops: tuple, result: "set[str]") -> None:
    """Recursively collects free symbol names from a sequence of UpdateOp instances."""
    from .expressions import AmplitudeExpr, _collect_free_symbols
    from .predicates import PredicateExpr, _collect_free_symbols_pred

    def _visit(val: Any) -> None:
        if isinstance(val, AmplitudeExpr):
            _collect_free_symbols(val, result)
        elif isinstance(val, PredicateExpr):
            _collect_free_symbols_pred(val, result)
        elif isinstance(val, UpdateOp):
            for _k, v in val.params:
                _visit(v)
        elif isinstance(val, tuple):
            for item in val:
                _visit(item)

    for op in ops:
        for _k, v in op.params:
            _visit(v)


__all__ = [
    "UpdateOp",
    "UpdateProgram",
    "_render_update_op",
    "_collect_free_symbols_from_ops",
]
