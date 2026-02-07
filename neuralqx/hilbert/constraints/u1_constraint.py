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
NetKet discrete constraint for U(1) gauge-fixing checks.

This module provides :class:`U1Constraint`, a :class:`netket.hilbert.constraint.DiscreteHilbertConstraint`
implementation used by constrained (gauge-fixed) U(1) Hilbert-space cores.

NetKet requires constrained discrete Hilbert spaces to supply a constraint object that can be called
on *batches* of states. This constraint:

- stores a gauge-fixing specification that has been preprocessed into a JAX-friendly format,
- evaluates gauge-fixing satisfaction for each state in a batch using a JIT-compatible routine,
- implements stable hashing/equality so NetKet can cache/compare constraint instances.

The actual per-state gauge-fixing check is delegated to :func:`check_one_state_jax` from
``..utils``; hashing uses :func:`make_hashable` to transform nested Python/JAX structures into a
comparable, hashable representation.
"""

from typing import Union

import netket as nk
import jax
import jax.numpy as jnp
from netket.utils import struct

from ..utils import check_one_state_jax
from ..utils import make_hashable


class U1Constraint(nk.hilbert.constraint.DiscreteHilbertConstraint):
    """
    Discrete Hilbert-space constraint enforcing a U(1) gauge fixing.

    This class implements NetKet's :class:`netket.hilbert.constraint.DiscreteHilbertConstraint` API for
    the constrained U(1) Hilbert-space backend. It is constructed with:

    - ``gauge_fixing``: a gauge-fixing specification already preprocessed into the internal JAX-friendly
      representation expected by :func:`check_one_state_jax`,
    - ``q_min``, ``q_max``, ``q_step``: range metadata used to perform modular arithmetic consistent
      with the Hilbert space's allowed local basis values,
    - ``cutoff``: stored primarily for identity/hashing and for traceability/debugging.

    Batch semantics
        NetKet calls the constraint with an array ``x`` of shape ``(B, N)`` (batch size ``B``, Hilbert
        size ``N``). The constraint must return a boolean array of shape ``(B,)`` indicating whether
        each state satisfies the gauge fixing.

    JAX / JIT friendliness
        The implementation must be compatible with JAX transformations and JIT compilation. The
        per-state check is performed by :func:`check_one_state_jax` and vectorized over the batch using
        :func:`jax.vmap`.

    PyTree notes
        NetKet expects constraint objects to be PyTrees. All stored fields are marked with
        ``struct.field(pytree_node=False)`` so they are treated as static (non-PyTree) metadata.

    :param gauge_fixing: Gauge-fixing specification in the internal JAX-friendly format consumed by
        :func:`check_one_state_jax`.
    :param cutoff: Cutoff associated with the Hilbert space using this constraint.
    :param q_min: Minimum allowed local quantum number/value.
    :param q_max: Maximum allowed local quantum number/value.
    :param q_step: Step between consecutive allowed local quantum numbers/values.
    :return: None.
    :raises None: This class does not intentionally raise exceptions during normal construction.
    """

    gauge_fixing: Union[jnp.ndarray, list] = struct.field(pytree_node=False)
    """The gauge fixing, pre-processed to fit jax"""

    cutoff: Union[float, int] = struct.field(pytree_node=False)
    """The cutoff of the space calling this constraint class"""

    q_min: Union[float, int] = struct.field(pytree_node=False)
    """The minimum possible quantum number for a given DoF"""

    q_max: Union[float, int] = struct.field(pytree_node=False)
    """The maximum possible quantum number for a given DoF"""

    q_step: Union[float, int] = struct.field(pytree_node=False)
    """The step between two consecutive allowed quantum numbers"""

    def __init__(
        self,
        gauge_fixing: Union[jnp.ndarray, list],
        cutoff: Union[float, int],
        q_min: Union[float, int],
        q_max: Union[float, int],
        q_step: Union[float, int],
    ):
        """
        Initialise the constraint with a processed gauge fixing and range metadata.

        :param gauge_fixing: Gauge-fixing specification in the internal processed format expected by
            :func:`check_one_state_jax`.
        :param cutoff: Cutoff associated with the Hilbert space using this constraint.
        :param q_min: Minimum allowed local quantum number/value.
        :param q_max: Maximum allowed local quantum number/value.
        :param q_step: Step between consecutive allowed local quantum numbers/values.
        :return: None.
        :raises None: This method does not intentionally raise exceptions.
        """

        # set class attributes
        self.gauge_fixing = gauge_fixing
        self.cutoff = cutoff
        self.q_min = q_min
        self.q_max = q_max
        self.q_step = q_step

    def __call__(self, x):
        """
        Evaluate gauge-fixing satisfaction for a batch of states.

        NetKet calls this method with a batch ``x`` of states. The method vectorizes a per-state check using
        :func:`jax.vmap` and returns a boolean array indicating whether each state satisfies the gauge
        fixing.

        The per-state check delegates to :func:`check_one_state_jax` and must remain compatible with JAX JIT.

        :param x: Batch of states of shape ``(B, N)`` where ``B`` is the batch size and ``N`` is the Hilbert
            size.
        :return: Boolean array of shape ``(B,)``; entry ``i`` is ``True`` iff ``x[i]`` satisfies the gauge
            fixing.
        :raises None: This method does not intentionally raise exceptions. Any errors typically indicate an
            invalid gauge-fixing structure or incompatible input shapes.
        """

        def check_wrapped(col):
            return check_one_state_jax(
                col,
                self.gauge_fixing,
                q_min=self.q_min,
                q_max=self.q_max,
                q_step=self.q_step,
            )

        batched_check = jax.vmap(check_wrapped, in_axes=0, out_axes=0)
        return batched_check(x)

    def __hash__(self):
        """
        Compute a stable hash for this constraint instance.

        The gauge fixing may be a nested structure containing lists, tuples, dicts, and JAX/NumPy arrays.
        To ensure a stable hash, the gauge fixing is converted into a hashable representation using
        :func:`make_hashable`.

        :param None: This method takes no arguments beyond ``self``.
        :return: Integer hash value suitable for using the constraint as a dictionary key and for NetKet's
            internal caching.
        :raises TypeError: If the stored gauge fixing contains elements that cannot be converted into a
            hashable form by :func:`make_hashable`.
        """

        # convert the gauge fixing into a hashable object first
        hashable_gauge_fixing = make_hashable(self.gauge_fixing)

        return hash(("U1Constraint", hashable_gauge_fixing, self.cutoff))

    def __eq__(self, other):
        """
        Test equality with another object.

        Two :class:`U1Constraint` instances are considered equal if:

        - their processed gauge-fixing specifications compare equal after conversion with
          :func:`make_hashable`, and
        - their ``cutoff`` values are equal.

        Objects of other types are considered not equal.

        :param other: Object to compare against.
        :return: ``True`` if ``other`` is a :class:`U1Constraint` with equivalent gauge fixing and cutoff,
            otherwise ``False``.
        :raises TypeError: If either gauge-fixing specification cannot be converted into a comparable
            hashable form by :func:`make_hashable`.
        """

        if isinstance(other, U1Constraint):
            # we compare gauge_fixing by converting both sides to a hashable
            # (and thus comparable) representation and then comparing
            return (
                make_hashable(self.gauge_fixing) == make_hashable(other.gauge_fixing)
                and self.cutoff == other.cutoff
                and make_hashable(self.q_min) == make_hashable(other.q_min)
                and make_hashable(self.q_max) == make_hashable(other.q_max)
                and make_hashable(self.q_step) == make_hashable(other.q_step)
            )
        return False
