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

r"""
Concrete implementation of the Abelian gauge group :math:`U(1)^N`.

This module defines :class:`~neuralqx.gauge_groups.u1.U1GaugeGroup`, a gauge group
implementation compatible with :class:`~neuralqx.hilbert.u1.HilbertU1`. It provides an
explicit Gauss constraint operator and supports multiple operator backends

- :class:`netket.operator.LocalOperator`
- :class:`neuralqx.operators.types.ComputationalOperator`
- :class:`neuralqx.operators.types.ComputationalJaxOperator`

The Gauss constraint is implemented in the squared form

.. math::

   \hat C \equiv \sum_{v \in V} \hat G_v^2

where :math:`\hat G_v` is the discrete divergence of the electric flux at vertex :math:`v`
with sign determined by edge orientation.
"""

from typing import Optional, Union

import jax.numpy as jnp
from netket.operator import LocalOperator

from neuralqx.gauge_groups import AbstractGaugeGroup
from neuralqx.hilbert.u1 import HilbertU1
from neuralqx.operators import get_quantum_number
from neuralqx.operators.types import ComputationalJaxOperator
from neuralqx.operators.types import ComputationalOperator
from neuralqx.operators.computational.Euclidean4d import GaussConstraintOperator
from neuralqx.utils.errors import IncompatibleHilbertSpaceError
from neuralqx.utils.errors import IncompatibleModdedOperatorWarning


class U1GaugeGroup(AbstractGaugeGroup):
    r"""
    Implements the Abelian gauge group :math:`U(1)^N` acting on a :class:`~neuralqx.hilbert.u1.HilbertU1`.

    Gauss constraint
    The Gauss law at each vertex :math:`v` is enforced by a generator :math:`\hat G_v`.
    On an oriented graph it takes the form

    .. math::

       \hat G_v = \sum_{\ell \in \mathrm{inc}(v)} \hat E_\ell \;-\; \sum_{\ell \in \mathrm{out}(v)} \hat E_\ell

    and the constraint used for optimisation is the positive operator

    .. math::

       \hat C = \sum_{v \in V} \hat G_v^2

    For :math:`U(1)^N`, the full constraint is the sum of :math:`N` independent copies.
    In the local operator backend this is realised by constructing one copy and shifting its
    support by a fixed site offset per copy.

    Operator backends
    Depending on configuration, the constraint is instantiated as one of

    - :class:`netket.operator.LocalOperator`
    - :class:`neuralqx.operators.types.ComputationalOperator`
    - :class:`neuralqx.operators.types.ComputationalJaxOperator`

    Group composition
    This implementation supports a simple notion of composition in terms of the number of
    independent :math:`U(1)` factors

    .. math::

       U(1)^m \times U(1)^n \mapsto U(1)^{m+n},
       \qquad
       \bigl(U(1)^m\bigr)^k \mapsto U(1)^{mk}

    :param H: A :math:`U(1)` compatible Hilbert space, must be an instance of
      :class:`~neuralqx.hilbert.u1.HilbertU1`.
    :param lazy: If True and a local operator backend is requested, delay building expensive
      local operator structures.
    :param computational: If True, prefer computational operator backends.
    :param jax: If True and `computational` is True, request a JAX compatible computational backend.
    :raises IncompatibleHilbertSpaceError: If `H` is not a :class:`~neuralqx.hilbert.u1.HilbertU1`.
    """

    def __init__(
        self,
        H: HilbertU1,
        *,
        lazy: Optional[bool] = True,
        computational: Optional[bool] = True,
        jax: Optional[bool] = True,
    ):
        """
        Initialise a :math:`U(1)^N` gauge group instance.

        This validates that the provided Hilbert space is compatible, stores the effective
        constraint construction flags, and delegates to the base gauge group initialisation
        which constructs and caches the constraint operator.

        :param H: Hilbert space instance compatible with :math:`U(1)` gauge degrees of freedom.
        :param lazy: If True and a local operator backend is requested, build lazily.
        :param computational: If True, prefer computational operator backends.
        :param jax: If True and `computational` is True, request a JAX compatible backend.
        :return: None.
        :raises IncompatibleHilbertSpaceError: If `H` is not a :class:`~neuralqx.hilbert.u1.HilbertU1`.
        """

        if not isinstance(H, HilbertU1):
            raise IncompatibleHilbertSpaceError(H, HilbertU1)

        # save a local copy of how this interface was initialised
        self._constraint_flags = {
            "lazy": bool(lazy),
            "computational": bool(computational),
            "jax": bool(jax),
        }

        super().__init__(
            H,
            lazy=lazy,
            computational=computational,
            jax=jax,
        )

    @property
    def name(self) -> str:
        """
        Return a descriptive name for this gauge group instance.

        :return: Name string of the form ``"U(1)^N Gauge Group"``.
        """

        return f"U(1)^{self.dimensions} Gauge Group"

    @property
    def is_abelian(self) -> bool:
        """
        Report whether the gauge group is Abelian.

        :return: Always True for :math:`U(1)^N`.
        """

        return True

    def _one_copy_constraint(self) -> LocalOperator:
        r"""
        Construct the squared Gauss constraint for a single :math:`U(1)` copy as a local operator.

        This builds

        .. math::

           \hat C^{(1)} = \sum_{v \in V} \hat G_v^2

        with

        .. math::

           \hat G_v = \sum_{\ell \in \mathrm{inc}(v)} \hat E_\ell - \sum_{\ell \in \mathrm{out}(v)} \hat E_\ell

        using the graph orientation to determine incoming and outgoing contributions.

        :return: A :class:`netket.operator.LocalOperator` representing :math:`\hat C^{(1)}`.
        """

        H = self.hilbert
        G = self.graph
        lop_type = LocalOperator

        # start with a zero operator on the full space
        total = lop_type(H.hilbert, dtype=jnp.float64)

        # build a temporary accumulator for G_v (per vertex), then square and add to total
        Gv = lop_type(H.hilbert, dtype=jnp.float64)

        for _node, attributes in G.handler.list_of_node_connectivity.items():
            # add incoming
            for edge in attributes["incoming"]:
                Gv += get_quantum_number(H, G.edge_to_index(edge), False, False)
            # subtract outgoing
            for edge in attributes["outgoing"]:
                Gv -= get_quantum_number(H, G.edge_to_index(edge), False, False)

            # Add G_v^2 to the total
            total += Gv @ Gv

            # reset for next vertex
            Gv = lop_type(H.hilbert, dtype=jnp.float64)

        return total

    def _shift_copy(self, base: LocalOperator, copy_idx: int) -> LocalOperator:
        """
        Create a shifted copy of a local operator to act on a different :math:`U(1)` factor.

        The shift is implemented by offsetting each site index in the operator support by

        ``copy_idx * num_dual_vertices``

        where ``num_dual_vertices`` is the number of dual graph vertices used by the Hilbert
        space layout for gauge copies.

        :param base: Local operator representing a single copy constraint.
        :param copy_idx: Which copy to shift to, with 0 meaning no shift.
        :return: Shifted :class:`netket.operator.LocalOperator` acting on the selected copy.
        """

        H = self.hilbert
        G = self.graph
        num_dual = len(list(G.dual_nx_graph.nodes))

        if copy_idx == 0:
            # no shift needed
            return base

        shifted_acting_on = [
            tuple(site + copy_idx * num_dual for site in sites)
            for sites in base.acting_on
        ]

        return LocalOperator(
            H.hilbert,
            operators=base.operators,
            acting_on=shifted_acting_on,
            dtype=base.dtype,
        )

    def init_constraint(
        self,
        *,
        computational: Optional[bool] = True,
        jax: Optional[bool] = False,
        lazy: Optional[bool] = True,
        modded: Optional[bool] = False,
    ) -> Union[LocalOperator, ComputationalOperator, ComputationalJaxOperator]:
        r"""
        Construct and return the Gauss constraint operator for :math:`U(1)^N`.

        The constraint used in optimisation is

        .. math::

           \hat C = \sum_{v \in V} \hat G_v^2

        and for :math:`U(1)^N` it is the sum of :math:`N` independent shifted copies of the
        single copy constraint.

        Backend selection
        If `computational` is True, returns a :class:`~neuralqx.operators.computational.Euclidean4d.GaussConstraintOperator`
        optionally in a JAX compatible form.
        If `computational` is False, returns a :class:`netket.operator.LocalOperator` built by
        explicitly summing shifted copies of the one copy local operator constraint.

        The `modded` option is supported only for computational backends.
        If `modded` is True while `computational` is False, a warning is emitted and the local
        operator is still constructed without modded arithmetic.

        :param computational: If True, return a computational operator backend.
        :param jax: If True and `computational` is True, return a JAX compatible computational backend.
        :param lazy: If True and a local operator backend is requested, delay building expensive structures.
        :param modded: If True, use modded addition in the computational implementation.
        :return: Constraint operator as a LocalOperator, ComputationalOperator, or ComputationalJaxOperator.
        """

        if computational:
            # return it as a computational operator, the gauge dimensions will be deduced from
            # the Hilbert object
            return GaussConstraintOperator(
                self.hilbert,
                gauge_dimensions=self.dimensions,
                modded=modded,
                jax=jax,
            )

        if not computational and modded:
            IncompatibleModdedOperatorWarning()

        # LocalOperator path: explicitly sum N shifted copies
        base = self._one_copy_constraint()
        if self.dimensions == 1:
            return base

        H = self.hilbert
        total = LocalOperator(H.hilbert, dtype=jnp.float64)

        # copy 0 (unshifted)
        total += base

        # copies 1...N-1 (shifted)
        for k in range(1, int(self.dimensions)):
            total += self._shift_copy(base, k)

        return total

    #
    #
    #   group composition
    #
    #   Semantics:
    #   - U(1)^m  *  U(1)^n  -> U(1)^(m+n)
    #   - (U(1)^m) ** k      -> U(1)^(m*k)

    def _current_flags(self) -> dict:
        """
        Determine the currently effective constraint construction flags.

        The method prefers the flags recorded at initialisation, and falls back to type checks
        on the cached constraint when necessary.

        :return: Dictionary with boolean keys ``"lazy"``, ``"computational"``, and ``"jax"``.
        """

        lazy = self._constraint_flags.get("lazy", True)
        comp = self._constraint_flags.get(
            "computational",
            isinstance(
                self.constraint, (ComputationalOperator, ComputationalJaxOperator)
            ),
        )
        jax = self._constraint_flags.get(
            "jax", isinstance(self.constraint, ComputationalJaxOperator)
        )

        return {"lazy": bool(lazy), "computational": bool(comp), "jax": bool(jax)}

    def _clone_with_dimensions(self, new_dim: int) -> "U1GaugeGroup":
        r"""
        Clone this gauge group with a new number of :math:`U(1)` factors.

        The clone preserves the effective constraint construction flags, updates the stored
        dimension, and forces reinitialisation of the cached constraint.

        :param new_dim: New gauge group dimension, interpreted as the exponent in :math:`U(1)^{new\_dim}`.
        :return: New :class:`U1GaugeGroup` instance with the requested dimension.
        """

        # get the current flags
        flags = self._current_flags()

        # create a fresh instance, honouring the current flags
        new = self.__class__(
            self.hilbert,
            lazy=flags["lazy"],
            computational=flags["computational"],
            jax=flags["jax"],
        )

        # reset dimensions and set the constraint flags
        new._dimensions = int(new_dim)
        new._constraint_flags = flags.copy()

        # re-init the constraint
        new.constraint = {
            "lazy": flags["lazy"],
            "computational": flags["computational"],
            "jax": flags["jax"],
            "reinit": True,
        }

        return new

    def __mul__(self, other) -> "U1GaugeGroup":
        r"""
        Compose two :math:`U(1)` gauge groups by adding their number of factors.

        The semantic operation is

        .. math::

           U(1)^m \times U(1)^n \mapsto U(1)^{m+n}

        Both operands must be defined on the same Hilbert space instance.

        :param other: Another :class:`U1GaugeGroup` defined on the same Hilbert space.
        :return: New :class:`U1GaugeGroup` with dimension ``self.dimensions + other.dimensions``.
        :raises ValueError: If the two gauge groups are defined on different Hilbert spaces.
        """

        if not isinstance(other, U1GaugeGroup):
            return NotImplemented
        # sanity: same Hilbert
        if other.hilbert is not self.hilbert:
            raise ValueError(
                "Cannot multiply gauge groups defined on different Hilbert spaces."
            )
        return self._clone_with_dimensions(self.dimensions + other.dimensions)

    def __pow__(self, power: int, modulo=None) -> "U1GaugeGroup":
        r"""
        Repeat composition of :math:`U(1)^N` by an integer power.

        The semantic operation is

        .. math::

           \bigl(U(1)^m\bigr)^k \mapsto U(1)^{mk}

        :param power: Positive integer exponent.
        :param modulo: Unused, present for Python protocol compatibility.
        :return: New :class:`U1GaugeGroup` with dimension ``self.dimensions * power``.
        :raises ValueError: If `power` is not a positive integer.
        """

        if not isinstance(power, int) or power < 1:
            raise ValueError("Power must be a positive integer.")

        return self._clone_with_dimensions(self.dimensions * power)

    def __imul__(self, other) -> "U1GaugeGroup":
        """
        In place composition of two :math:`U(1)` gauge groups.

        This updates the current instance dimension and reinitialises the cached constraint to
        match the result of `self * other`.

        :param other: Another :class:`U1GaugeGroup` defined on the same Hilbert space.
        :return: This instance after in place update.
        :raises ValueError: If the two gauge groups are defined on different Hilbert spaces.
        """

        out = self * other
        self._dimensions = out._dimensions
        self._constraint_flags = out._constraint_flags
        self.constraint = {
            "lazy": self._constraint_flags["lazy"],
            "computational": self._constraint_flags["computational"],
            "jax": self._constraint_flags["jax"],
            "reinit": True,
        }

        return self

    def __repr__(self) -> str:
        return (
            f"U1GaugeGroup("
            f"dimensions={self.dimensions}, "
            f"hilbert={self.hilbert.core.hilbert}, "
            f"is_abelian={self.is_abelian}, "
            f"is_computational={self.is_computational}"
            f")"
        )
