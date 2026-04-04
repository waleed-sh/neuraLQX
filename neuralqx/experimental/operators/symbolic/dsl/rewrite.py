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
Fluent immutable update-program builder for the symbolic operator DSL.

The :class:`Update` class is designed for ergonomic construction of
site-update programs: every method is chainable and returns a *new*
immutable :class:`Update` instance.

Module-level factory functions (``shift``, ``write``, ``swap``, ``permute``,
``affine``, ``scatter``, ``identity``) serve as zero-boilerplate entry points,
there is no need to first construct an empty ``Update()`` before chaining.

Examples::

    from neuralqx.experimental.operators.symbolic.dsl import shift, write, swap, permute, scatter

    # Single operation, direct factory call
    update = shift("i", +1)

    # Compound update, chain freely
    update = shift("i", -1).shift("j", +1)  # hopping
    update = swap("i", "j").write("k", 0)  # swap then zero
    update = permute("i", "j", "k")  # cyclic rotation
    update = affine("i", scale=2, bias=-1)  # x[i] = 2*x[i] - 1

    # Bulk scatter to static flat indices
    update = scatter([0, 5, 10], [val_0, val_5, val_10])

    # Identity (no-op), for diagonal operators
    update = identity()

    # Conditional update
    update = Update.cond(
        site("i") > 0,
        if_true=shift("i", -1),
        if_false=write("i", 0),
    )

Amplitude-side semantics
-------------------------
Amplitude expressions are evaluated after the emitted/connected state ``x'``
has been constructed for the current branch.

By default:
- ``site("i").value`` refers to the source configuration ``x[i]``.
- ``emitted("i").value`` refers to the connected/emitted configuration ``x'[i]``.

This allows matrix elements to depend on either the source state, the emitted
state, or both. In addition, ``wrap_mod(expr)`` applies the same Hilbert-aware
modulo-wrap semantics used by ``shift_mod(...)``.

Branch-multiset semantics
--------------------------
The padded connected-states representation returned by ``get_conn_padded`` is a
**branch multiset**, not a deduplicated adjacency row. If two terms (or two
emissions within one term) produce the same ``x'``, both appear as separate
entries with separate matrix elements.  Callers that accumulate matrix elements
must sum over duplicate ``x'`` entries explicitly.
"""

from __future__ import annotations

from typing import Any

from neuralqx.experimental.operators.symbolic.dsl.selectors import SiteSelector
from neuralqx.experimental.operators.symbolic.ir.expressions import (
    AmplitudeExpr,
    coerce_amplitude_expr,
)
from neuralqx.experimental.operators.symbolic.ir.predicates import (
    PredicateExpr,
    coerce_predicate_expr,
)
from neuralqx.experimental.operators.symbolic.ir.update import UpdateOp, UpdateProgram

#
#
#   Internal helpers


def _site_ref_to_expr(ref: Any) -> AmplitudeExpr:
    """
    Converts a site reference to an AmplitudeExpr for the site *index*.

    Accepts:
        - ``str`` label  -> ``symbol("site:<label>:index")``
        - :class:`SiteSelector` -> ``selector.index``
        - ``int``         -> ``constant(float(idx))``
        - :class:`AmplitudeExpr` -> used as-is
    """
    if isinstance(ref, str):
        return AmplitudeExpr.symbol(f"site:{ref}:index")
    if isinstance(ref, SiteSelector):
        return ref.as_site_ref()
    if isinstance(ref, int):
        return AmplitudeExpr.constant(float(ref))
    if isinstance(ref, AmplitudeExpr):
        return ref
    raise TypeError(
        f"site reference must be str, SiteSelector, int, or AmplitudeExpr; "
        f"got {type(ref).__name__!r}."
    )


#
#
#   Update class


class Update:
    """
    Immutable, chainable site-update program builder.

    Every instance method appends one operation and returns a **new**
    ``Update`` object, the original is never mutated. The canonical
    entry points are the module-level free functions (:func:`shift`,
    :func:`write`, :func:`swap`, :func:`permute`, :func:`affine`,
    :func:`scatter`, :func:`identity`) which avoid the ``Update()``
    boilerplate.

    Args:
        _program: Internal update program (do not pass manually).
    """

    __slots__ = ("_program",)

    def __init__(self, _program: UpdateProgram | None = None) -> None:
        self._program: UpdateProgram = UpdateProgram() if _program is None else _program

    def _append(self, op: UpdateOp) -> "Update":
        return Update(self._program.append(op))

    def to_program(self) -> UpdateProgram:
        """Returns the underlying immutable :class:`~neuralqx.experimental.operators.symbolic.ir.update.UpdateProgram`."""
        return self._program

    #
    #
    #   Primitive site mutations

    def shift(
        self,
        site_ref: str | SiteSelector | int | AmplitudeExpr,
        delta: Any,
    ) -> "Update":
        """
        Appends ``x'[i] = x[i] + delta``.

        Args:
            site_ref: Target site (label string, selector, or flat index).
            delta: Shift amount, numeric or amplitude expression.

        Returns:
            New ``Update`` with this operation appended.
        """
        return self._append(
            UpdateOp.from_mapping(
                kind="shift_site",
                params={
                    "site": _site_ref_to_expr(site_ref),
                    "delta": coerce_amplitude_expr(delta),
                },
            )
        )

    def shift_mod(
        self,
        site_ref: str | SiteSelector | int | AmplitudeExpr,
        delta: Any,
    ) -> "Update":
        """
        Appends a Hilbert-aware wrapped shift.

        Semantics are resolved from the enclosing operator's Hilbert space at
        build/compile time. For now this requires contiguous unit-spaced integer
        local_states such as [-m_max, ..., m_max].

        Resulting runtime semantics:

            x'[i] = ((x[i] + delta - state_min) % mod_span) + state_min
        """
        return self._append(
            UpdateOp.from_mapping(
                kind="shift_mod_site",
                params={
                    "site": _site_ref_to_expr(site_ref),
                    "delta": coerce_amplitude_expr(delta),
                },
            )
        )

    def write(
        self,
        site_ref: str | SiteSelector | int | AmplitudeExpr,
        value: Any,
    ) -> "Update":
        """
        Appends ``x'[i] = value``.

        Args:
            site_ref: Target site.
            value: New quantum number, numeric or amplitude expression.

        Returns:
            New ``Update`` with this operation appended.
        """
        return self._append(
            UpdateOp.from_mapping(
                kind="write_site",
                params={
                    "site": _site_ref_to_expr(site_ref),
                    "value": coerce_amplitude_expr(value),
                },
            )
        )

    def swap(
        self,
        site_a: str | SiteSelector | int | AmplitudeExpr,
        site_b: str | SiteSelector | int | AmplitudeExpr,
    ) -> "Update":
        """
        Appends ``x'[a], x'[b] = x[b], x[a]``.

        Args:
            site_a: First site.
            site_b: Second site.

        Returns:
            New ``Update`` with this operation appended.
        """
        return self._append(
            UpdateOp.from_mapping(
                kind="swap_sites",
                params={
                    "site_a": _site_ref_to_expr(site_a),
                    "site_b": _site_ref_to_expr(site_b),
                },
            )
        )

    def permute(
        self,
        *site_refs: str | SiteSelector | int | AmplitudeExpr,
    ) -> "Update":
        """
        Appends a **cyclic rotation** over K sites.

        After the operation::

            x'[s0] ← x[s1],   x'[s1] ← x[s2],   ...,   x'[sK-1] ← x[s0]

        All K source values are captured from the current ``x'`` state
        *before* any writes are applied, so the rotation is atomic.

        Args:
            *site_refs: Two or more site references in rotation order.

        Returns:
            New ``Update`` with this operation appended.

        Raises:
            ValueError: If fewer than 2 site references are provided.
        """
        if len(site_refs) < 2:
            raise ValueError("permute requires at least 2 site references.")
        exprs = tuple(_site_ref_to_expr(s) for s in site_refs)
        return self._append(
            UpdateOp.from_mapping(
                kind="permute_sites",
                params={"sites": exprs},
            )
        )

    def affine(
        self,
        site_ref: str | SiteSelector | int | AmplitudeExpr,
        *,
        scale: Any,
        bias: Any = 0,
    ) -> "Update":
        """
        Appends ``x'[i] = scale * x[i] + bias``.

        Args:
            site_ref: Target site.
            scale: Multiplicative scale, numeric or amplitude expression.
            bias: Additive bias, numeric or amplitude expression (default 0).

        Returns:
            New ``Update`` with this operation appended.
        """
        return self._append(
            UpdateOp.from_mapping(
                kind="affine_site",
                params={
                    "site": _site_ref_to_expr(site_ref),
                    "scale": coerce_amplitude_expr(scale),
                    "bias": coerce_amplitude_expr(bias),
                },
            )
        )

    def scatter(
        self,
        flat_indices: list[int] | tuple[int, ...],
        values: list[Any] | tuple[Any, ...],
    ) -> "Update":
        """
        Appends bulk writes to static flat site indices.

        For each ``(flat_index, value)`` pair::

            x'[flat_index] = value

        Indices must be compile-time-constant integers (baked into the IR).
        Values may be arbitrary amplitude expressions.

        Args:
            flat_indices: Sequence of static integer site indices.
            values: Sequence of amplitude expressions (or coercible values).

        Returns:
            New ``Update`` with this operation appended.

        Raises:
            ValueError: If *flat_indices* and *values* have different lengths.
        """
        flat_indices = tuple(int(i) for i in flat_indices)
        values = tuple(coerce_amplitude_expr(v) for v in values)
        if len(flat_indices) != len(values):
            raise ValueError(
                f"scatter: flat_indices and values must have the same length; "
                f"got {len(flat_indices)} indices and {len(values)} values."
            )
        return self._append(
            UpdateOp.from_mapping(
                kind="scatter",
                params={"flat_indices": flat_indices, "values": values},
            )
        )

    def invalidate(self, *, reason: str | None = None) -> "Update":
        """
        Marks this branch as **invalid** (zero matrix element).

        Useful for boundary conditions: emit a branch unconditionally and
        let the update program itself decide validity.

        Args:
            reason: Optional readable explanation.

        Returns:
            New ``Update`` with this operation appended.
        """
        params = {"reason": str(reason)} if reason is not None else None
        return self._append(
            UpdateOp.from_mapping(kind="invalidate_branch", params=params)
        )

    #
    #
    #   Conditional update

    @classmethod
    def cond(
        cls,
        predicate: Any,
        *,
        if_true: "Update",
        if_false: "Update | None" = None,
    ) -> "Update":
        """
        Returns a new ``Update`` wrapping a JAX-compatible conditional.

        At lowering time this becomes ``jax.lax.cond(predicate, ...)`` so both
        branches must produce the same output shape. The ``if_false`` branch
        defaults to the identity (no site changes) when not provided.

        Args:
            predicate: Branch predicate, :class:`~neuralqx.experimental.operators.symbolic.ir.predicates.PredicateExpr`
                or any coercible value (e.g. ``site("i").value > 0``).
            if_true: Update program to apply when *predicate* is true.
            if_false: Update program to apply when *predicate* is false.
                Defaults to identity (no writes).

        Returns:
            New ``Update`` wrapping the conditional.
        """
        pred_expr = coerce_predicate_expr(predicate)
        then_ops = if_true._program.ops
        else_ops = if_false._program.ops if if_false is not None else ()
        op = UpdateOp.from_mapping(
            kind="cond_branch",
            params={
                "predicate": pred_expr,
                "then_ops": then_ops,
                "else_ops": else_ops,
            },
        )
        return cls(UpdateProgram(ops=(op,)))

    #
    #
    #   Dunder

    def __repr__(self) -> str:
        kinds = [op.kind for op in self._program.ops]
        return f"Update({kinds!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Update):
            return NotImplemented
        return self._program == other._program

    def __hash__(self) -> int:
        return hash(self._program)


# Identity sentinel
_IDENTITY = Update()


#
#
#   Module-level factory functions


def shift(
    site_ref: str | SiteSelector | int | AmplitudeExpr,
    delta: Any,
) -> Update:
    """
    Returns an ``Update`` that shifts site *site_ref* by *delta*.

    Example::

        shift("i", +1)  # raise site i by 1
        shift(0, -1)  # lower flat site 0 by 1
        shift("j", site("i").value)  # shift j by x[i]
    """
    return _IDENTITY.shift(site_ref, delta)


def shift_mod(
    site_ref: str | SiteSelector | int | AmplitudeExpr,
    delta: Any,
) -> Update:
    """
    Returns an Update performing a Hilbert-aware wrapped modular shift.

    Example::

        shift_mod("i", +1)
        shift_mod(0, -2)
    """
    return _IDENTITY.shift_mod(site_ref, delta)


def write(
    site_ref: str | SiteSelector | int | AmplitudeExpr,
    value: Any,
) -> Update:
    """
    Returns an ``Update`` that writes *value* to site *site_ref*.

    Example::

        write("i", 0)  # zero site i
        write(5, site("j").value)  # copy x[j] into flat site 5
    """
    return _IDENTITY.write(site_ref, value)


def swap(
    site_a: str | SiteSelector | int | AmplitudeExpr,
    site_b: str | SiteSelector | int | AmplitudeExpr,
) -> Update:
    """
    Returns an ``Update`` that swaps sites *site_a* and *site_b*.

    Example::

        swap("i", "j")  # exchange x[i] and x[j]
        swap(0, 10)  # exchange flat sites 0 and 10
    """
    return _IDENTITY.swap(site_a, site_b)


def permute(
    *site_refs: str | SiteSelector | int | AmplitudeExpr,
) -> Update:
    """
    Returns an ``Update`` performing a cyclic rotation over K sites.

    Example::

        permute("i", "j", "k")  # x'[i]←x[j], x'[j]←x[k], x'[k]←x[i]
        permute(0, 5, 10)  # same with flat indices
    """
    return _IDENTITY.permute(*site_refs)


def affine(
    site_ref: str | SiteSelector | int | AmplitudeExpr,
    *,
    scale: Any,
    bias: Any = 0,
) -> Update:
    """
    Returns an ``Update`` computing ``x'[i] = scale * x[i] + bias``.

    Example::

        affine("i", scale=2, bias=-1)  # x'[i] = 2*x[i] - 1
        affine(0, scale=-1, bias=0)  # negate flat site 0
    """
    return _IDENTITY.affine(site_ref, scale=scale, bias=bias)


def scatter(
    flat_indices: list[int] | tuple[int, ...],
    values: list[Any] | tuple[Any, ...],
) -> Update:
    """
    Returns an ``Update`` performing bulk writes to static flat indices.

    Example::

        scatter([0, 10, 20], [1, -1, 0])  # write constant values
        scatter([0, 10], [site("i").value, 0])  # mixed expr / constant
    """
    return _IDENTITY.scatter(flat_indices, values)


def identity() -> Update:
    """
    Returns the identity (no-op) ``Update``.

    Use for diagonal operators where ``x' = x``::

        DOperator(hi, "diagonal").globally().emit(identity(), amplitude=my_expr)
    """
    return _IDENTITY


__all__ = [
    "Update",
    "affine",
    "identity",
    "permute",
    "scatter",
    "shift",
    "shift_mod",
    "swap",
    "write",
]
