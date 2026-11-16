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


"""Linen access to NNX implementations without duplicating model kernels."""

import inspect

from flax.core import FrozenDict
from flax.nnx import bridge


class LinenModel(bridge.ToLinen):
    """Linen facade retaining the connected-state capability of an NNX model.

    Standard Linen ``init`` and ``apply`` manage variables. The official Flax
    bridge carries parameter collections and partition metadata across the
    boundary. Both backends execute the same model operations, including the
    exact connected-state evaluator.
    """

    def connected_log_value(self, sources, targets, owners, *, chunk_size=None):
        """Call the wrapped model's exact connected-state evaluator through Linen."""
        return self(
            sources,
            targets,
            owners,
            chunk_size=chunk_size,
            nnx_method="connected_log_value",
        )


def to_linen(module_class, *args, name=None, skip_rng=None, **kwargs):
    """Construct a Linen module from a neuraLQX or native NNX module class.

    Pass constructor arguments without ``rngs``. Linen supplies initialization
    keys through ``module.init(key, inputs)``. For example,
    ``to_linen(MLP, 32, hidden_features=(64, 64))`` returns a Linen wavefunction.
    The conversion also works for individual layers inside a custom Linen model.
    Wrapped neuraLQX wavefunctions retain their connected-state capability.
    RNG injection is skipped when the constructor has no named ``rngs``
    parameter. Override ``skip_rng`` for a custom constructor that accepts RNG streams
    indirectly through keyword arguments.
    """
    from ..models.base import LogWavefunction

    if "rngs" in kwargs:
        raise ValueError("Pass RNGs to Linen init(), not to to_linen().")
    if not isinstance(module_class, type):
        raise TypeError(
            "to_linen expects an NNX module class, not an initialized instance."
        )
    if skip_rng is None:
        skip_rng = "rngs" not in inspect.signature(module_class.__init__).parameters
    if issubclass(module_class, LogWavefunction):
        return LinenModel(
            module_class,
            args=args,
            kwargs=FrozenDict(kwargs),
            name=name,
            skip_rng=skip_rng,
        )
    return bridge.to_linen(module_class, *args, name=name, skip_rng=skip_rng, **kwargs)
