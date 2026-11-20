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


"""Normalized autoregressive amplitudes with exact reusable prefixes."""

import jax.numpy as jnp
import numpy as np
from flax import nnx

import jax
from neuralqx.utils.errors import NeuralqxValueError

from ..layers import Embed
from ..layers import Linear
from ..utils import ConnectionPolicy
from ..utils import as_rngs
from ..utils.connections import map_connections
from ..utils.connections import validate_connections
from ..utils.initialization import compute_dtype
from ..utils.initialization import input_array
from ..utils.initialization import positive_int
from .base import LogWavefunction


class AutoregressiveRNN(LogWavefunction):
    r"""Categorical recurrent wavefunction with normalized conditional amplitudes.

    The model represents
    :math:`\log\psi(x)=\tfrac12\sum_i\log p(x_i|x_{<i})
    +i\sum_i\phi(x_i|x_{<i})`. Every conditional distribution is normalized
    with ``log_softmax``. Consequently the Born distribution is normalized on
    the full Cartesian product of ``local_states``. Restricting it to a gauge
    sector requires the usual constrained sampler and a new normalization.

    A changed coordinate leaves its causal prefix unchanged. Connected
    evaluation can reuse a source prefix and recompute the entire remaining
    suffix. Static suffix buckets keep shapes stable as the changed coordinates
    vary. Early changes fall back to the complete recurrence.
    """

    in_features: int
    """Number of ordered configuration coordinates."""

    hidden_features: int
    """Size of the recurrent hidden state."""

    local_states: nnx.Variable
    """Sorted allowed labels, stored as nontrainable model state."""

    embedding: Embed
    """Embedding added to the recurrence after observing each coordinate."""

    recurrent: Linear
    """Affine transformation of the previous hidden state."""

    probability: Linear
    """Conditional categorical logits from the current causal hidden state."""

    phase: Linear | None
    """Conditional phase contributions when complex output is requested."""

    dtype: object
    """Floating computation dtype for hidden states and logits."""

    def __init__(
        self,
        in_features,
        hidden_features=32,
        *,
        local_states=(0, 1),
        complex_output=True,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=None,
        connection_policy=None,
        rngs,
    ):
        """Create a shared categorical recurrence for the given coordinate order."""
        self.in_features = positive_int(in_features, "in_features")
        self.hidden_features = positive_int(hidden_features, "hidden_features")
        labels = np.asarray(local_states)
        if (
            labels.ndim != 1
            or len(labels) < 2
            or not np.issubdtype(labels.dtype, np.number)
        ):
            raise NeuralqxValueError(
                "local_states must be a one-dimensional numeric alphabet of size at least two."
            )
        if (
            np.iscomplexobj(labels)
            or not np.all(np.isfinite(labels))
            or len(np.unique(labels)) != len(labels)
        ):
            raise NeuralqxValueError(
                "local_states must contain distinct finite real labels."
            )
        if jnp.issubdtype(
            compute_dtype(dtype, param_dtype), jnp.complexfloating
        ) or jnp.issubdtype(jnp.dtype(param_dtype), jnp.complexfloating):
            raise NeuralqxValueError(
                "Autoregressive logits and parameters must use real dtypes."
            )
        self.local_states = nnx.Variable(jnp.asarray(np.sort(labels)))
        dtype = compute_dtype(dtype, param_dtype)
        self.dtype = dtype
        self.connection_policy = connection_policy or ConnectionPolicy(min_features=64)
        rngs = as_rngs(rngs)
        options = dict(
            rngs=rngs, dtype=dtype, param_dtype=param_dtype, precision=precision
        )
        self.embedding = Embed(
            len(labels),
            hidden_features,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=nnx.initializers.normal(0.1),
        )
        self.recurrent = Linear(hidden_features, hidden_features, **options)
        self.probability = Linear(hidden_features, len(labels), **options)
        self.phase = (
            Linear(hidden_features, len(labels), **options) if complex_output else None
        )

    def _labels(self, states):
        states = input_array(states, self.in_features, None)
        labels = self.local_states[...]
        indices = jnp.clip(jnp.searchsorted(labels, states), 0, labels.shape[0] - 1)
        valid = jnp.all(labels[indices] == states, axis=-1)
        return indices, valid

    def _step(self, carry, label):
        hidden, value = carry
        logits = jax.nn.log_softmax(self.probability(hidden), axis=-1)
        increment = 0.5 * jnp.take_along_axis(logits, label[..., None], axis=-1)[..., 0]
        if self.phase is not None:
            phase = jnp.take_along_axis(self.phase(hidden), label[..., None], axis=-1)[
                ..., 0
            ]
            increment = increment + 1j * phase
        hidden = jnp.tanh(self.recurrent(hidden) + self.embedding(label))
        return hidden, value + increment

    def _initial(self, batch):
        hidden = jnp.zeros((batch, self.hidden_features), self.dtype)
        value = jnp.zeros(
            batch,
            jnp.result_type(self.dtype, 1j) if self.phase is not None else self.dtype,
        )
        return hidden, value

    def _run(self, labels, initial, *, keep_prefix=False):
        # Take the first transition outside scan so the carry acquires the
        # same manual-mesh variation as the labels and model parameters.
        # Constant zero carries otherwise fail shard_map's scan type check.
        first = self._step(initial, labels[:, 0])

        def step(carry, label):
            updated = self._step(carry, label)
            return updated, updated if keep_prefix else None

        final, history = jax.lax.scan(step, first, labels[:, 1:].T)
        if keep_prefix:
            history = jax.tree.map(
                lambda start, rest: jnp.concatenate((start[None], rest), axis=0),
                first,
                history,
            )
        return final, history

    def __call__(self, states):
        """Return normalized log amplitudes, or negative infinity for invalid labels."""
        labels, valid = self._labels(states)
        leading = labels.shape[:-1]
        labels = labels.reshape((-1, self.in_features))
        (_, value), _ = self._run(labels, self._initial(labels.shape[0]))
        return jnp.where(valid, value.reshape(leading), -jnp.inf)

    def prepare_connections(self, sources):
        """Store the causal hidden state and cumulative log amplitude at every boundary."""
        labels, valid = self._labels(sources)
        initial = self._initial(labels.shape[0])
        _, (hidden, values) = self._run(labels, initial, keep_prefix=True)
        hidden = jnp.concatenate((initial[0][:, None], hidden.swapaxes(0, 1)), axis=1)
        values = jnp.concatenate((initial[1][:, None], values.T), axis=1)
        return hidden, values, valid

    def connected_log_value(self, sources, targets, owners, *, chunk_size=None):
        """Reuse a causal prefix and recompute each target's complete suffix.

        The earliest change in a chunk selects a prefix at zero, half, or
        three quarters of the sequence. All choices are exact. No host readback
        or variable-length allocation is used to choose the prefix.
        """
        chunk_size = validate_connections(sources, targets, owners, chunk_size)
        if not self.use_connection_reuse(sources.shape, targets.shape):
            return map_connections(lambda x, _: self(x), targets, owners, chunk_size)
        boundaries = tuple(
            sorted({0, self.in_features // 2, 3 * self.in_features // 4})
        )
        if len(boundaries) == 1:
            return map_connections(lambda x, _: self(x), targets, owners, chunk_size)

        # Avoid building an entire recurrent history when no chunk can reuse
        # even half of it. This guard uses only configuration labels and stays
        # on device, including when different data shards choose differently.
        eligible = jnp.all(
            targets[:, : boundaries[1]] == sources[owners, : boundaries[1]], axis=-1
        )
        size = (
            targets.shape[0]
            if chunk_size is None
            else min(chunk_size, targets.shape[0])
        )
        eligible = jnp.pad(eligible, (0, (-len(eligible)) % size), constant_values=True)
        use_prefix = jnp.any(jnp.all(eligible.reshape((-1, size)), axis=-1))

        def reuse():
            hidden, values, source_valid = self.prepare_connections(sources)

            def evaluate(x, rows):
                labels, valid = self._labels(x)
                dirty = x != sources[rows]
                earliest = jnp.min(
                    jnp.where(dirty, jnp.arange(self.in_features), self.in_features)
                )
                branch = sum(
                    (earliest >= start).astype(jnp.int32) for start in boundaries[1:]
                )
                branch = jnp.where(jnp.all(source_valid[rows]), branch, 0)

                def suffix(start):
                    def run():
                        initial = (hidden[rows, start], values[rows, start])
                        (_, value), _ = self._run(labels[:, start:], initial)
                        return jnp.where(valid, value, -jnp.inf)

                    return run

                return jax.lax.switch(
                    branch, tuple(suffix(start) for start in boundaries)
                )

            return map_connections(evaluate, targets, owners, chunk_size)

        return jax.lax.cond(
            use_prefix,
            reuse,
            lambda: map_connections(lambda x, _: self(x), targets, owners, chunk_size),
        )

    def sample(self, key, n_samples):
        """Draw independent Born samples on the full, unconstrained product space.

        This method does not enforce graph or gauge constraints. The conditional
        probabilities correspond to ``machine_pow=2``. Returned labels have
        shape ``(n_samples, in_features)`` and the dtype of ``local_states``.
        """
        n_samples = positive_int(n_samples, "n_samples")

        def step(hidden, key):
            label = jax.random.categorical(key, self.probability(hidden), axis=-1)
            hidden = jnp.tanh(self.recurrent(hidden) + self.embedding(label))
            return hidden, self.local_states[...][label]

        keys = jax.random.split(key, self.in_features)
        hidden, first = step(self._initial(n_samples)[0], keys[0])
        _, states = jax.lax.scan(step, hidden, keys[1:])
        return jnp.concatenate((first[None], states), axis=0).T
