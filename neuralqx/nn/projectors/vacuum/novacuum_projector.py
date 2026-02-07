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

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

Array = jnp.ndarray


class NoVacuumProjector(nn.Module):
    """
    Wrapper that enforces zero amplitude on the vacuum (all-zero) basis state.

    Assumes:
      - input sigma has shape (B, N)
      - base network returns shape (B,)

    Behaviour:
      - For any row sigma[b, :] that is all zeros, the output is replaced by `vacuum_value`.
      - Default vacuum_value=0.
    """

    base: nn.Module
    """The base Flax module to be wrapped."""

    vacuum_value: Any = -1e30  # supports float/complex/Array scalars
    """The value the all-zero basis element amplitude will be replaced with"""

    enabled: bool = True
    """If False, no replacement of the all-zero basis element amplitude occurs."""

    @nn.compact
    def __call__(self, sigma: Array, **kwargs) -> Array:
        out = self.base(sigma, **kwargs)

        # optional sanity check (happens at trace-time)
        if out.ndim != 1:
            raise ValueError(
                f"{type(self).__name__} expects base output shape (B,), got {out.shape}."
            )

        if not self.enabled:
            return out

        is_vacuum = jnp.all(sigma == 0, axis=-1)

        # build replacement with both real and imag = vacuum_value if out is complex
        if jnp.issubdtype(out.dtype, jnp.complexfloating):
            vv = self.vacuum_value + 1j * self.vacuum_value
        else:
            vv = self.vacuum_value

        vv = jnp.asarray(vv, dtype=out.dtype)

        # jnp.where broadcasts scalar vv across masked entries
        return jnp.where(is_vacuum, vv, out)


def wrap_model(
    base_model: nn.Module,
    *,
    vacuum_value: Any = -1e30,
    enabled: bool = True,
) -> NoVacuumProjector:
    """Convenience wrapper to wrap any Flax module."""
    return NoVacuumProjector(
        base=base_model, vacuum_value=vacuum_value, enabled=enabled
    )


def projector(
    *,
    vacuum_value: Any = -1e30,
    enabled: bool = True,
):
    """
    Decorator factory to wrap a Module class into a NoVacuumProjector.

    Usage:
        @projector()
        class MyAnsatz(nn.Module):
            ...

        model = MyAnsatz(...)
    """

    def _decorator(BaseModuleClass: type[nn.Module]):
        def _factory(*args, **kwargs) -> NoVacuumProjector:
            base_instance = BaseModuleClass(*args, **kwargs)
            return NoVacuumProjector(
                base=base_instance,
                vacuum_value=vacuum_value,
                enabled=enabled,
            )

        _factory.__name__ = BaseModuleClass.__name__
        _factory.__doc__ = BaseModuleClass.__doc__
        return _factory

    return _decorator
