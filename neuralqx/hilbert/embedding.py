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

"""Isometric inclusion of discrete configurations without basis enumeration."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError
from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import field

from .space import DiscreteHilbertSpace


class BasisEmbedding(Struct):
    r"""Identify source basis vectors with distinct target basis vectors.

    This represents the isometry $E|x\rangle=|f(x)\rangle$. A source
    wavefunction is extended by zero outside the image of $f$, preserving
    its norm. Neither space needs to be enumerable.

    By default labels are unchanged. This is appropriate for nested cutoffs
    with the same configuration layout and basis conventions. Supplying
    ``encode`` and ``decode`` also permits permutations, additional fixed
    degrees of freedom, and different configuration widths. Both functions
    must accept arbitrary leading batch dimensions and use JAX operations.
    ``decode`` returns source labels with the same leading dimensions.
    Projection checks both source validity and the round trip through
    ``encode``, so labels outside the image are rejected.

    The caller must ensure that ``encode`` maps every valid source state to
    a valid target state and that ``decode(encode(x)) == x``. Merely sharing
    a configuration width does not establish a physical identification.
    This class describes basis injections, not changes of basis that map a
    configuration into a superposition.
    """

    source: DiscreteHilbertSpace = field(static=True)
    """Space in which the variational wavefunction is represented."""

    target: DiscreteHilbertSpace = field(static=True)
    """Space containing the embedded wavefunction and the larger operator."""

    encode: Callable | None = field(static=True, default=None)
    """Map source labels to target labels, or leave labels unchanged."""

    decode: Callable | None = field(static=True, default=None)
    """Map target labels to candidate source labels, or leave labels unchanged."""

    def __post_init__(self):
        """Check the spaces and require a paired forward and reverse mapping."""
        if not isinstance(self.source, DiscreteHilbertSpace) or not isinstance(
            self.target, DiscreteHilbertSpace
        ):
            raise NeuralqxTypeError(
                "BasisEmbedding requires two discrete Hilbert spaces."
            )
        if (self.encode is None) != (self.decode is None):
            raise NeuralqxValueError("Supply both encode and decode, or neither.")
        if self.encode is None and self.source.size != self.target.size:
            raise NeuralqxValueError(
                "Different configuration widths require encode and decode."
            )
        if self.encode is not None and not (
            callable(self.encode) and callable(self.decode)
        ):
            raise NeuralqxTypeError("encode and decode must be callable.")

    def embed(self, states: Any):
        """Map configurations with final axis ``source.size`` into the target."""
        states = jnp.asarray(states)
        if states.shape[-1] != self.source.size:
            raise NeuralqxValueError(
                "Source configuration width does not match the embedding."
            )
        result = states if self.encode is None else jnp.asarray(self.encode(states))
        if result.shape != (*states.shape[:-1], self.target.size):
            raise NeuralqxValueError(
                "encode must preserve batch dimensions and return target labels."
            )
        return result

    def project(self, states: Any):
        """Return candidate source labels and a mask for membership in the image.

        Labels where the mask is false must not be evaluated by a source
        wavefunction. Estimator kernels substitute valid source configurations
        before evaluating padded or rejected connections.
        """
        states = jnp.asarray(states)
        if states.shape[-1] != self.target.size:
            raise NeuralqxValueError(
                "Target configuration width does not match the embedding."
            )
        result = states if self.decode is None else jnp.asarray(self.decode(states))
        if result.shape != (*states.shape[:-1], self.source.size):
            raise NeuralqxValueError(
                "decode must preserve batch dimensions and return source labels."
            )
        valid = self.source.is_valid(result)
        return result, valid & jnp.all(self.embed(result) == states, axis=-1)


__all__ = ["BasisEmbedding"]
