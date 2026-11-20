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

"""Operator restrictions and residuals of zero-extended wavefunctions."""

from collections.abc import Sequence

import jax.numpy as jnp

from neuralqx.hilbert.embedding import BasisEmbedding
from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError
from neuralqx.utils.struct import field

from .abstract import AbstractOperator
from .computational import ComputationalOperator
from .discrete import DiscreteOperator
from .wrappers.squared import Squared
from .wrappers.sum_of_squares import SumOfSquares


def _source(wrapper):
    return wrapper.embedding.source


def _action(operator):
    return operator.operator if isinstance(operator, Squared) else operator


def _validate(operator, embedding):
    if not isinstance(embedding, BasisEmbedding):
        raise NeuralqxTypeError("embedding must be a BasisEmbedding.")
    if not isinstance(operator, DiscreteOperator):
        raise NeuralqxTypeError(
            "The action must be a discrete operator with matrix-row connectivity."
        )
    if operator.hilbert != embedding.target:
        raise NeuralqxValueError(
            "The action must be defined on the embedding's target Hilbert space."
        )


class EmbeddedOperator(ComputationalOperator):
    r"""Restrict a larger linear operator to $E^\dagger A E$.

    Input configurations and returned connections belong to the source
    Hilbert space. Matrix rows are evaluated in the target space and columns
    outside the embedding are removed. The original operator supplies all
    physics, including its cutoff and boundary convention.

    Squaring this restriction gives $E^\dagger A^\dagger P A E$, with
    $P=EE^\dagger$. It excludes outgoing action. Use ``FullResidual(A,
    embedding)`` for $E^\dagger A^\dagger A E$, or ``embed_operator`` to
    pull back an existing ``Squared`` objective before restricting it.
    """

    hilbert: object = field(static=True, init=False, derived=_source)
    """Source Hilbert space on which the restricted operator acts."""

    operator: DiscreteOperator
    """Linear operator evaluated in the target space."""

    embedding: BasisEmbedding
    """Identification of source configurations inside the target basis."""

    def __post_init__(self):
        """Validate the operator's domain without enumerating either basis."""
        _validate(self.operator, self.embedding)
        super().__post_init__()

    @property
    def dtype(self):
        """Matrix-element dtype of the target operator."""
        return self.operator.dtype

    @property
    def max_conn_size(self):
        """The target operator's row capacity, including masked columns."""
        return self.operator.max_conn_size

    @property
    def is_hermitian(self):
        """Restriction preserves known Hermiticity."""
        return self.operator.is_hermitian

    @property
    def is_diagonal(self):
        """Restriction preserves a known diagonal action."""
        return self.operator.is_diagonal

    @property
    def adjoint(self):
        r"""Return $E^\dagger A^\dagger E$."""
        return EmbeddedOperator(self.operator.adjoint, self.embedding)

    def _get_conn_padded_kernel(self, state):
        """Return one restricted matrix row."""
        x, m = self._get_conn_padded_batch_kernel(state[None])
        return x[0], m[0]

    def _get_conn_padded_batch_kernel(self, states):
        """Map rows to the target, then mask columns outside the source image."""
        targets, mels = self.operator.get_conn_padded(
            self.embedding.embed(states), use_streaming=False
        )
        projected, valid = self.embedding.project(targets)
        mels = jnp.where(valid, mels, 0)
        return jnp.where((mels != 0)[..., None], projected, states[:, None]), mels

    def get_conn_compact(self, states):
        """Retain a parent's compact connection path when one is available."""
        compact = getattr(self.operator, "get_conn_compact", None)
        packed = None if compact is None else compact(self.embedding.embed(states))
        if packed is None:
            return None
        targets, mels, owners = packed
        projected, valid = self.embedding.project(targets)
        mels = jnp.where(valid, mels, 0)
        return jnp.where((mels != 0)[:, None], projected, states[owners]), mels, owners


class FullResidual(AbstractOperator):
    r"""Measure the complete action of a larger operator on an embedded state.

    For an action $A$ and isometry $E$, this objective is
    $E^\dagger A^\dagger A E$. Passing ``Squared(A)`` has the same meaning
    as passing ``A``. In particular, ``Squared(H.adjoint)`` keeps the
    convention $H H^\dagger$ and is never squared a second time.

    With ``part='outgoing'`` the objective is
    $E^\dagger A^\dagger(I-EE^\dagger)AE$. With ``part='retained'`` the
    complementary projector $EE^\dagger$ is used. These two expectations
    sum to the full expectation, including when the larger operator changes
    matrix elements inside the smaller space.

    Estimators draw ``n_probes`` distinct-output proposals with replacement
    per sampled source configuration. They evaluate complete rows at those
    outputs and include an exact proposal correction. Interference is retained
    before taking an absolute square. This adds Monte Carlo variance but
    avoids enumerating every path through $A^\dagger A$. ``n_probes=None``
    sums all reachable outputs, removing this additional sampling noise at
    the cost of two-step connectivity. The estimator requires actual adjoint
    rows and fixed operator coefficients. Gradients differentiate the full
    estimator, including its normalization, rather than using the ordinary
    Hermitian covariance shortcut.

    Both the source and target spaces may be constrained and non-indexable.
    The target operator must include the outgoing configurations of interest.
    No wrapper can recover connections that its parent has already truncated.
    """

    hilbert: object = field(static=True, init=False, derived=_source)
    """Source space in which the variational state and optimizer live."""

    operator: DiscreteOperator = field(converter=_action)
    """Target-space action, with one outer Squared marker removed if present."""

    embedding: BasisEmbedding
    """Isometric basis injection used to extend the state by zero."""

    part: str = field(static=True, default="all")
    """Output region: 'all', 'retained', or 'outgoing'."""

    n_probes: int | None = field(static=True, default=1)
    """Output draws per source sample, or None to sum all reachable outputs."""

    connection_chunk_size: int | None = field(static=True, default=None)
    """Source rows per preparation block, or a device-aware automatic bound.

    Automatic compact preparation uses larger CPU blocks when the declared
    connection width permits them. Accelerator and traced preparation retain
    128 source rows. This controls operator work, independently of the model's
    evaluation chunk size.
    """

    def __post_init__(self):
        """Require a compatible action, a valid region, and positive capacities."""
        _validate(self.operator, self.embedding)
        if self.part not in ("all", "retained", "outgoing"):
            raise NeuralqxValueError("part must be 'all', 'retained', or 'outgoing'.")
        if self.n_probes is not None and (
            type(self.n_probes) is not int or self.n_probes < 1
        ):
            raise NeuralqxValueError("n_probes must be a positive integer or None.")
        if self.connection_chunk_size is not None and (
            type(self.connection_chunk_size) is not int
            or self.connection_chunk_size < 1
        ):
            raise NeuralqxValueError(
                "connection_chunk_size must be a positive integer or None."
            )
        _validate(self.operator.adjoint, self.embedding)
        super().__post_init__()

    @property
    def dtype(self):
        """Real dtype of this nonnegative quadratic objective."""
        return jnp.empty((), self.operator.dtype).real.dtype

    @property
    def is_hermitian(self):
        """Every projected action norm is a Hermitian quadratic form."""
        return True

    @property
    def adjoint(self):
        """Return this positive, self-adjoint objective."""
        return self


def embed_operator(operator, embedding, **residual_options):
    r"""Pull an operator or quadratic objective back through a basis embedding.

    Pass the target-space objective itself. For ``Q = Squared(A)``, the
    result represents $E^\dagger Q E=E^\dagger A^\dagger A E$, including
    the action that leaves the source space. The recorded factor $A$ permits
    an efficient norm estimator without expanding $Q$ or squaring it again.
    A ``SumOfSquares`` or explicit sequence is mapped term by term, preserving
    separate local squares. In particular, a target objective built from
    ``Squared(H.adjoint)`` retains the ordering $H H^\dagger$.

    Ordinary linear operators are restricted as $E^\dagger A E$. Options
    such as ``n_probes`` apply to quadratic markers. The returned objective
    already includes both retained and outgoing contributions and can be
    passed directly to ``state.expect`` or the VMC driver.

    Example:
        ``full = embed_operator(large_quadratic, embedding, n_probes=2)``
    """
    if "part" in residual_options:
        raise NeuralqxTypeError(
            "embed_operator includes the full objective. Use outgoing_residual for leakage."
        )
    if isinstance(operator, SumOfSquares) or isinstance(operator, (tuple, list)):
        return tuple(
            embed_operator(op, embedding, **residual_options) for op in operator
        )
    if isinstance(operator, Squared):
        if (
            isinstance(operator.operator, DiscreteOperator)
            and operator.operator.is_diagonal
        ):
            # Diagonal action has no outgoing component or nodal contribution.
            # Preserve the ordinary one-row estimator for this exact case.
            if residual_options:
                FullResidual(operator, embedding, **residual_options)
            return Squared(EmbeddedOperator(operator.operator, embedding))
        return FullResidual(operator, embedding, **residual_options)
    if residual_options:
        raise NeuralqxTypeError(
            "Residual options require a Squared objective or a sum of squares."
        )
    return EmbeddedOperator(operator, embedding)


def outgoing_residual(operator, embedding, **options):
    r"""Extract the outgoing contribution of a target-space quadratic objective.

    For ``Q = Squared(A)``, this represents
    $E^\dagger A^\dagger(I-EE^\dagger)AE$. Its expectation is the part of
    $\langle E\psi|Q|E\psi\rangle/\|\psi\|^2$ contributed by outputs of
    $A$ outside the source image. For ``Squared(H.adjoint)``, those outputs
    belong to $H^\dagger E\psi$. The squared objective is not treated as a
    new linear action to be squared again.

    ``SumOfSquares`` and explicit sequences retain separate residual terms.
    Every term must retain its ``Squared`` marker. An expanded quadratic
    matrix has lost its factorization and is rejected. For a bare linear
    action use ``FullResidual(A, embedding, part='outgoing')`` explicitly.

    This contribution may be measured independently or combined with a
    retained objective. Adding it to ``embed_operator(Q, embedding)`` would
    count outgoing action a second time.

    Example:
        ``leakage = outgoing_residual(large_quadratic, embedding, n_probes=2)``
    """
    if isinstance(operator, SumOfSquares) or isinstance(operator, (tuple, list)):
        return tuple(outgoing_residual(op, embedding, **options) for op in operator)
    if not isinstance(operator, Squared):
        raise NeuralqxTypeError(
            "outgoing_residual requires a Squared objective or a sum of squared objectives. "
            "Keep the original factorization. For a linear action A use "
            "FullResidual(A, embedding, part='outgoing')."
        )
    return FullResidual(operator, embedding, part="outgoing", **options)


def contains_residual(operator):
    """Recognize auxiliary-output estimators inside supported objective trees."""
    from .wrappers.cost import PenaltyCost

    if isinstance(operator, FullResidual):
        return True
    if isinstance(operator, PenaltyCost):
        return contains_residual(operator.cost_operator)
    if isinstance(operator, SumOfSquares) or (
        isinstance(operator, Sequence)
        and not isinstance(operator, (str, bytes, AbstractOperator))
    ):
        return any(contains_residual(op) for op in operator)
    return False


__all__ = ["EmbeddedOperator", "FullResidual", "embed_operator", "outgoing_residual"]
