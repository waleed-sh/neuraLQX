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


"""Monte Carlo statistics containers."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from neuralqx.utils.struct import Struct

from .formatting import scalar_item
from .formatting import stats_repr


class Stats(Struct):
    """Summary statistics for scalar Monte Carlo estimators."""

    mean: Any
    variance: Any
    error_of_mean: Any
    n_samples: Any
    r_hat: Any = jnp.asarray(jnp.nan)

    @property
    def error(self) -> Any:
        """Alias for the Monte Carlo standard error of the mean."""
        return self.error_of_mean

    @property
    def R_hat(self) -> Any:
        """Alias using the common diagnostic name."""
        return self.r_hat

    @property
    def Sigma(self) -> Any:
        """NetKet-compatible alias for ``error_of_mean``."""
        return self.error_of_mean

    def to_dict(self) -> dict[str, Any]:
        """Return a logging-friendly dictionary."""
        return {
            "Mean": scalar_item(self.mean),
            "Variance": scalar_item(self.variance),
            "Sigma": scalar_item(self.error_of_mean),
            "R_hat": scalar_item(self.r_hat),
            "N": scalar_item(self.n_samples),
        }

    def to_compound(self) -> tuple[str, dict[str, Any]]:
        """Return a compound logging payload."""
        return "Mean", self.to_dict()

    def real(self) -> Stats:
        """Return stats with a real-valued mean."""
        return self.replace(mean=jnp.real(self.mean))

    def imag(self) -> Stats:
        """Return stats with an imaginary-valued mean."""
        return self.replace(mean=jnp.imag(self.mean))

    def __repr__(self) -> str:
        return stats_repr(self.mean, self.error_of_mean, self.variance, self.r_hat)


__all__ = ["Stats"]
