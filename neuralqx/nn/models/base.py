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


"""NNX wavefunctions and the optional connected-state capability."""

from flax import nnx

import jax

from ..utils.connections import ConnectionPolicy
from ..utils.connections import map_connections
from ..utils.connections import sparse_changes
from ..utils.connections import validate_connections


class LogWavefunction(nnx.Module):
    """Base for models returning one scalar log amplitude per configuration.

    Inputs have a final configuration axis. Leading batch axes are preserved
    by ``__call__``. Models must be deterministic and independent of other batch
    entries for their output to define a wavefunction. Subclasses can supply
    exact connected-state reuse through ``prepare_connections`` and
    ``apply_connections``. Ordinary evaluation is always the fallback.

    The cache belongs to one evaluation at one set of parameters. It is built
    inside autodiff and is never retained between optimizer steps. This is
    essential when differentiating a non-Hermitian or squared local estimator.
    """

    connection_policy: ConnectionPolicy = ConnectionPolicy(mode="off")
    """Static policy selecting exact reuse or ordinary target evaluation."""

    def __call__(self, states):
        """Return log amplitudes with the final configuration axis removed."""
        raise NotImplementedError(f"{type(self).__name__} must implement __call__.")

    def prepare_connections(self, sources):
        """Build differentiable source features for one connected-state call."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement source reuse."
        )

    def apply_connections(self, cache, owners, sites, differences):
        """Evaluate exact target log amplitudes from a source cache and sparse changes."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement sparse updates."
        )

    def use_connection_reuse(self, source_shape, target_shape):
        """Select reuse from static batch shapes and the model's policy.

        Subclasses can add architecture-specific cost checks here. The decision
        is made while tracing, so a rejected fast path adds no device operations.
        """
        return self.connection_policy.enabled(
            source_shape[-1], source_shape[0], target_shape[0]
        )

    def connected_log_value(self, sources, targets, owners, *, chunk_size=None):
        """Evaluate connected configurations, reusing source work when appropriate.

        Parameters
        ----------
        sources : array, shape (batch, features)
            Configurations on which the operator acts.
        targets : array, shape (connections, features)
            Full connected configurations.
        owners : integer array, shape (connections,)
            Source row corresponding to each target.
        chunk_size : int, optional
            Bound on target evaluations in each rematerialized chunk.

        The result has shape ``(connections,)``. All differences are discovered
        on device. Overflow selects full evaluation for that chunk without
        dropping connections or approximating the wavefunction. Gradients flow
        through both the source cache and its updates.
        """
        chunk_size = validate_connections(sources, targets, owners, chunk_size)
        policy = self.connection_policy
        if not self.use_connection_reuse(sources.shape, targets.shape):
            return map_connections(lambda x, _: self(x), targets, owners, chunk_size)
        cache = self.prepare_connections(sources)

        def evaluate(x, rows):
            sites, delta, overflow = sparse_changes(
                sources, x, rows, policy.max_changes, dtype=getattr(self, "dtype", None)
            )
            return jax.lax.cond(
                overflow,
                lambda: self(x),
                lambda: self.apply_connections(cache, rows, sites, delta),
            )

        return map_connections(evaluate, targets, owners, chunk_size)
