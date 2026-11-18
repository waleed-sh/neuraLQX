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


"""Exclusion of one distinguished basis configuration."""

import jax.numpy as jnp

from .base import DiagonalProjector


class NonVacuum(DiagonalProjector, name="nonvacuum"):
    r"""Remove exactly the vacuum basis amplitude from a wavefunction.

    For the normalized basis configuration :math:`|\Omega\rangle`,

    .. math::

        P_{\ne\Omega}=I-|\Omega\rangle\langle\Omega|.

    By default ``vacuum=0`` excludes the configuration whose every coordinate
    is zero. This is the spin and intertwiner vacuum in the SU(2) encoding
    and the zero-charge configuration in U(1) encodings. Configurations with
    only some zero labels remain allowed. Supply a one-dimensional array to
    exclude another basis configuration, or a scalar for a repeated label.

    The model is not renormalized. Variational sampling supplies the usual
    normalization on the remaining support. For example,
    ``model = nn.projectors.nonvacuum(nn.MLP(hilbert.size, rngs=0))``.
    This removes the vacuum component exactly without an overlap penalty.
    It does not assert that a non-vacuum ground state exists.
    """

    vacuum: object = 0
    """Scalar vacuum label or vector of labels, stored as non-trainable state."""

    def keep(self, states):
        """Keep a configuration if any coordinate differs from the vacuum."""
        states = jnp.asarray(states)
        vacuum = jnp.asarray(self.vacuum)
        if vacuum.ndim > 1 or (vacuum.ndim == 1 and vacuum.shape != states.shape[-1:]):
            raise ValueError(
                "vacuum must be a scalar or a vector matching the configuration size."
            )
        return jnp.any(states != vacuum, axis=-1)
