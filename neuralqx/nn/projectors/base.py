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


"""Pure JAX projections acting on amplitudes of logarithmic wavefunctions."""

from abc import abstractmethod

import jax.numpy as jnp

from neuralqx.utils.errors import NeuralqxValueError
from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import field

from ..utils.connections import map_connections
from ..utils.connections import validate_connections
from .registry import ProjectorMeta
from .utils import coherent_logsum
from .utils import log_values
from .utils import mask_log_values


class Projector(Struct, metaclass=ProjectorMeta):
    r"""A fixed linear projection of a neural wavefunction.

    Implement :meth:`project` to return :math:`\log(P\psi)(x)` from a callable
    supplying :math:`\log\psi(x)`. A projector acts on amplitudes, not their
    logarithms. It must satisfy :math:`P^2=P`. An orthogonal projector also
    satisfies :math:`P^\dagger=P`. These mathematical properties are the
    implementer's responsibility and cannot be inferred from arbitrary code.

    Concrete subclasses register automatically. For example,
    ``class ChargeZero(DiagonalProjector, name="charge_zero")`` exposes
    ``@nn.projectors.charge_zero``. Annotated fields define the constructor
    and are immutable pytree data by default. Declare discrete options with
    ``field(static=True, default=...)``. Array fields belong to model state,
    never trainable parameters. Use ``register=False`` for a private class.

    Calling an instance wraps an initialized NNX model, a Linen module, an
    explicit apply function, or a model class. A decorated class becomes a
    constructor returning the corresponding wrapped model. No parameters
    are copied or reinitialized when wrapping an existing instance.

    Projection does not normalize the wavefunction. Sampling must use the
    projected amplitude. Direct autoregressive sampling from the original
    model is not inherited. The evaluator must be deterministic and give
    the same amplitude independently of batch composition and chunking.
    Noncommuting stacked projections need not themselves form a projector.

    A variational state recognizes the native wrappers. For non-diagonal
    ``Squared`` objectives it uses the output-sampling residual estimator,
    so action into excluded configurations is retained. By default there is
    one output draw per source. To control that additional sampling variance,
    pass a ``FullResidual`` with an identity ``BasisEmbedding`` and a chosen
    ``n_probes``, or ``None`` to sum all reachable outputs. Existing embedded
    residuals retain their own settings. Ordinary network connection reuse
    is unaffected by this choice of squared-action estimator.
    """

    @abstractmethod
    def project(self, log_value, states):
        """Return one projected log amplitude per input configuration.

        ``log_value(states)`` evaluates the underlying wavefunction. Both
        functions accept arbitrary leading batch axes and one final feature
        axis. Use JAX operations and preserve these batch axes. A zero
        amplitude is represented by a real part of negative infinity.
        """

    def project_connected(
        self, log_value, connected, sources, targets, owners, *, chunk_size=None
    ):
        """Evaluate projected targets, with a full-evaluation default.

        Override this when the projection can use ``connected``, whose
        signature matches a model's ``connected_log_value``. ``owners``
        indexes the unprojected source configurations. Reuse is local to
        this call and remains differentiable with respect to model weights.
        """
        validate_connections(sources, targets, owners, chunk_size)
        return map_connections(
            lambda x, _: self.project(log_value, x), targets, owners, chunk_size
        )

    def __call__(self, model):
        """Wrap a supported model or decorate its constructor with this projection."""
        from .wrappers import wrap_model

        return wrap_model(model, self)


class DiagonalProjector(Projector):
    r"""An orthogonal projection onto a subset of basis configurations.

    Implement :meth:`keep` to define a boolean indicator :math:`m(x)`.
    Then :math:`(P\psi)(x)=m(x)\psi(x)`. Because the mask takes only zero
    and one, the projection is Hermitian and idempotent. The implementation
    preserves the wrapped model's connected-state evaluator, adding only
    the mask on its output. Excluded states have exact zero amplitude and
    zero parameter derivative.

    The mask restricts the variational state, not the operator. In particular,
    a squared or embedded constraint must still include action into excluded
    configurations. Do not remove those operator output connections.
    """

    @abstractmethod
    def keep(self, states):
        """Return a boolean array of shape ``states.shape[:-1]``."""

    def project(self, log_value, states):
        """Evaluate the base network once and set excluded amplitudes to zero."""
        states = jnp.asarray(states)
        return mask_log_values(log_values(log_value(states), states), self.keep(states))

    def project_connected(
        self, log_value, connected, sources, targets, owners, *, chunk_size=None
    ):
        """Mask the base model's exact connected evaluation without extra forwards."""
        validate_connections(sources, targets, owners, chunk_size)
        values = connected(sources, targets, owners, chunk_size=chunk_size)
        return mask_log_values(log_values(values, targets), self.keep(targets))


class RowProjector(Projector):
    r"""A projection specified by a finite expansion of each matrix row.

    Implement :meth:`connections` so that

    .. math::

        (P\psi)(x) = \sum_{a=1}^{K} c_a(x)\psi(y_a(x)),\qquad
        c_a(x)=\langle x|P|y_a(x)\rangle.

    This covers finite symmetry averages and sparse basis projections.
    Coefficients are matrix rows, with no implicit conjugation. Negative
    and complex coefficients are summed coherently in a stable logarithmic
    reduction. The result can be complex even when the base model is real.

    Connected evaluation forwards expanded configurations to the base
    model's exact evaluator. Its usual cost policy and overflow fallback
    remain active. ``evaluation_chunk_size`` bounds ordinary network calls.
    Connection arrays have static size and are local to the sample shard.
    No host deduplication, all-gather, or persistent activation cache is used.

    Exact nodes return ``-inf+0j`` with a zero log-derivative convention.
    A logarithmic derivative is undefined at a parameter-dependent node.
    As with other logarithmic ansatzes, evaluating a derivative of an
    amplitude exactly at such a node requires an amplitude-level formulation.
    """

    evaluation_chunk_size: int | None = field(static=True, default=None)
    """Maximum base-network batch for ordinary projection, or no extra bound."""

    def __post_init__(self):
        """Validate the static evaluation bound before tracing the model."""
        if self.evaluation_chunk_size is not None and (
            type(self.evaluation_chunk_size) is not int
            or self.evaluation_chunk_size < 1
        ):
            raise NeuralqxValueError(
                "evaluation_chunk_size must be a positive integer or None."
            )

    @abstractmethod
    def connections(self, states):
        """Return ``(configurations, coefficients)`` with a fixed row width.

        Shapes are ``(*batch, K, features)`` and ``(*batch, K)`` for inputs
        of shape ``(*batch, features)``. ``K`` must be positive. Zero
        coefficients pad unused entries. Padded configurations are replaced
        by the input row before network evaluation, so their labels need
        not be valid. Nonzero entries must belong to the model's domain.
        """

    def _rows(self, states):
        points, coefficients = self.connections(states)
        points, coefficients = jnp.asarray(points), jnp.asarray(coefficients)
        if (
            points.ndim != states.ndim + 1
            or points.shape[:-2] != states.shape[:-1]
            or points.shape[-1] != states.shape[-1]
            or points.shape[-2] == 0
            or coefficients.shape != points.shape[:-1]
        ):
            raise NeuralqxValueError(
                "Projector connections must have shapes (*batch, K, features) and (*batch, K)."
            )
        points = jnp.where((coefficients != 0)[..., None], points, states[..., None, :])
        return points, coefficients

    def project(self, log_value, states):
        """Evaluate and coherently sum the amplitude expansion of each row."""
        from neuralqx.jax.streaming import apply_batched

        states = jnp.asarray(states)
        points, coefficients = self._rows(states)
        flat = points.reshape((-1, points.shape[-1]))
        values = apply_batched(
            lambda _, x: log_values(log_value(x), x),
            None,
            flat,
            chunk_size=self.evaluation_chunk_size,
        )
        return coherent_logsum(values.reshape(coefficients.shape), coefficients)

    def project_connected(
        self, log_value, connected, sources, targets, owners, *, chunk_size=None
    ):
        """Reuse source features while evaluating the row expansion of each target."""
        validate_connections(sources, targets, owners, chunk_size)
        points, coefficients = self._rows(targets)
        limits = [n for n in (chunk_size, self.evaluation_chunk_size) if n is not None]
        size = min(limits) if limits else None
        flat = points.reshape((-1, points.shape[-1]))
        expanded_owners = jnp.repeat(owners, points.shape[-2])
        values = connected(sources, flat, expanded_owners, chunk_size=size)
        return coherent_logsum(
            log_values(values, flat).reshape(coefficients.shape), coefficients
        )
