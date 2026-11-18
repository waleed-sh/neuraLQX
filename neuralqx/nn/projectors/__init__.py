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


"""Registered projections of neural wavefunction amplitudes.

Wrap an instance with ``nn.projectors.nonvacuum(model)`` or decorate a model
class with ``@nn.projectors.nonvacuum``. Parameterized projectors use the same
syntax, for example ``@nn.projectors.nonvacuum(vacuum=(0, 0, 0))``.

Custom subclasses register automatically::

    class ChargeZero(DiagonalProjector, name="charge_zero"):
        def keep(self, states):
            return jnp.sum(states, axis=-1) == 0


    model = nn.projectors.charge_zero(nn.MLP(12, rngs=0))

DiagonalProjector defines basis support, RowProjector defines a coherent row
expansion, and Projector accepts any pure JAX implementation of a projection.
Registration is a construction-time lookup and adds no work to a JIT kernel.
"""

from neuralqx.utils.struct import field

from .base import DiagonalProjector
from .base import Projector
from .base import RowProjector
from .registry import available_projectors
from .registry import get_projector
from .registry import projector_decorator as _projector_decorator
from .registry import register_projector
from .vacuum import NonVacuum
from .wrappers import ProjectedCallable
from .wrappers import ProjectedLinen
from .wrappers import ProjectedNNX


def __getattr__(name):
    try:
        return _projector_decorator(name)
    except KeyError:
        raise AttributeError(f"No projector named {name!r} is registered.") from None


def __dir__():
    return sorted(set(globals()) | set(available_projectors()))


__all__ = [
    "DiagonalProjector",
    "NonVacuum",
    "ProjectedCallable",
    "ProjectedLinen",
    "ProjectedNNX",
    "Projector",
    "RowProjector",
    "available_projectors",
    "field",
    "get_projector",
    "nonvacuum",
    "register_projector",
]
