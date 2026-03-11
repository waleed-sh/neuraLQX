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
Abstract interface for gauge groups in neuraLQX.

This module defines the `AbstractGaugeGroup` base class which standardizes how gauge
groups expose their Gauss constraint operator, compatibility with a Hilbert space and
graph, and key group properties such as name, dimension, and whether the group is
abelian.

Concrete gauge group implementations must provide the group name, an abelian flag, and
a method that constructs the Gauss constraint in one of several supported operator
representations.
"""

import abc
import logging
from typing import Optional
from typing import Union

from netket.operator import LocalOperator

from neuralqx.debug import event
from neuralqx.graph.core import AbstractGraph
from neuralqx.hilbert import AbstractHilbertInterface
from neuralqx.operators.types import ComputationalOperator
from neuralqx.operators.types import ComputationalJaxOperator


class AbstractGaugeGroup(abc.ABC):
    """
    Abstract base class for gauge group implementations.

    A gauge group implementation is responsible for
    - declaring compatibility with a given Hilbert space and its graph
    - exposing the group dimension in the representation used by the Hilbert space
    - constructing and caching the Gauss constraint operator

    The Gauss constraint can be implemented as a NetKet `LocalOperator`, a neuraLQX
    `ComputationalOperator`, or a `ComputationalJaxOperator`. The class also supports
    switching operator backends by reinitialising the cached constraint.

    :param H: Hilbert space instance that defines graph, local degrees of freedom, and
        gauge group dimensionality.
    :param lazy: If True and a LocalOperator backend is requested, delay building the
        operator structure until needed.
    :param computational: If True, prefer neuraLQX computational operator backends.
    :param jax: If True and `computational` is True, request a JAX friendly computational
        operator backend.
    :return: None.
    """

    _hilbert: AbstractHilbertInterface
    """The Hilbert space this gauge group will be compatible with."""

    _constraint: Union[
        LocalOperator, ComputationalOperator, ComputationalJaxOperator
    ] = None
    """The constraint of the gauge group as an operator."""

    _dimensions: int
    """The dimensions of the gauge group."""

    def __init__(
        self,
        H: AbstractHilbertInterface,
        *,
        lazy: Optional[bool] = True,
        computational: Optional[bool] = True,
        jax: Optional[bool] = False,
    ):
        """
        Initialize the gauge group interface and construct the Gauss constraint.

        This stores references to the Hilbert space and its graph, reads the gauge dimension
        from the Hilbert space, and initializes the `constraint` property using the provided
        backend selection flags.

        :param H: Hilbert space instance the gauge group is compatible with.
        :param lazy: If True and a LocalOperator backend is requested, delay building the
            operator structure until needed.
        :param computational: If True, prefer neuraLQX computational operator backends.
        :param jax: If True and `computational` is True, request a `ComputationalJaxOperator`.
        :return: None.
        """

        event(
            msg="INITIALIZING GAUGE GROUP INTERFACE",
            tag="GAUGE_GROUP:INIT",
            level=logging.INFO,
            lazy=lazy,
            computational=computational,
            jax=jax,
        )

        # local copies
        self._hilbert = H
        self._graph = H.graph
        self._dimensions = H.gauge_dimensions

        # create the constraint object
        self.constraint = dict(
            lazy=lazy,
            computational=computational,
            jax=jax,
            reinit=False,
        )

        # local flags
        self._computational = computational
        self._jax = jax

    @property
    def is_computational(self) -> bool:
        r"""
        Report whether this instance is configured to use computational operator backends.

        If True, the constraint is intended to be built as a neuraLQX computational operator
        type rather than a NetKet `LocalOperator`.

        :return: True if computational operators are selected, otherwise False.
        """

        return self._computational

    @property
    def is_jax(self) -> bool:
        r"""
        Report whether this instance is configured to use the JAX computational backend.

        If True, and if computational operators are enabled, the constraint is intended to be
        built as a `ComputationalJaxOperator`.

        :return: True if JAX computational operators are selected, otherwise False.
        """

        return self._jax

    @property
    def graph(self) -> AbstractGraph:
        """
        Return the graph associated with the Hilbert space used by this gauge group.

        :return: Graph instance that defines the combinatorial structure of the model.
        """

        return self._graph

    @property
    def hilbert(self) -> AbstractHilbertInterface:
        """
        Return the Hilbert space this gauge group is compatible with.

        :return: Hilbert space interface instance used to configure this gauge group.
        """

        return self._hilbert

    @property
    def dimensions(self) -> int:
        """
        Return the dimension of the gauge group in the representation used by the Hilbert space.

        :return: Gauge group dimension.
        """

        return self._dimensions

    @property
    def constraint(
        self,
    ) -> Union[LocalOperator, ComputationalOperator, ComputationalJaxOperator]:
        """
        Return the cached Gauss constraint operator for this gauge group.

        The returned operator type depends on how the constraint was initialised and can be a
        NetKet `LocalOperator`, a neuraLQX `ComputationalOperator`, or a `ComputationalJaxOperator`.

        :return: Constraint operator instance.
        """

        return self._constraint

    @constraint.setter
    def constraint(
        self,
        params: dict,
    ) -> None:
        """
        Initialise or reinitialise the cached Gauss constraint operator.

        This setter accepts a dictionary of backend selection options and calls
        `init_constraint` when the constraint has not been created yet or when reinitialisation
        is requested.

        :param params: Dictionary of options controlling constraint construction. Supported keys
            include
            - "computational" to request computational operator backends
            - "jax" to request the JAX computational backend when computational is True
            - "lazy" to delay building a LocalOperator backend
            - "reinit" to force rebuilding the cached constraint even if it already exists
        """

        lazy = params.get("lazy", True)
        computational = params.get("computational", True)
        jax = params.get("jax", True)
        reinit = params.get("reinit", False)

        # force re-initialisation if requested as users may want to switch types during runtime
        if self._constraint is None or reinit:
            self._constraint = self.init_constraint(
                lazy=lazy,
                computational=computational,
                jax=jax,
            )

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name in string format of this gauge group."""

    @property
    @abc.abstractmethod
    def is_abelian(self) -> bool:
        """Returns True if this gauge group is abelian, otherwise False."""

    @abc.abstractmethod
    def init_constraint(
        self,
        *,
        computational: Optional[bool] = True,
        jax: Optional[bool] = False,
        lazy: Optional[bool] = True,
    ) -> Union[LocalOperator, ComputationalOperator, ComputationalJaxOperator]:
        """
        Construct and return the Gauss constraint operator for this gauge group.

        Implementations should create the constraint in one of the supported operator
        representations. By default, computational backends are preferred, and LocalOperator
        construction can optionally be deferred using the `lazy` flag.

        :param computational: If True, return a neuraLQX computational operator backend.
        :param jax: If True and `computational` is True, return a `ComputationalJaxOperator`.
        :param lazy: If True and a LocalOperator backend is requested, delay building the
            operator structure until needed.
        :return: Gauss constraint operator instance.
        """

    def __repr__(self) -> str:
        return f"AbstractGaugeGroup(hilbert={self.hilbert.hilbert_netket})"

    def __mul__(self, other):
        return NotImplemented

    def __rmul__(self, other):
        return NotImplemented

    def __imul__(self, other):
        return NotImplemented

    def __add__(self, other):
        return NotImplemented

    def __radd__(self, other):
        return NotImplemented

    def __iadd__(self, other):
        return NotImplemented

    def __pow__(self, power, modulo=None):
        return NotImplemented
