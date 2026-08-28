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


"""Semantic squared-operator marker."""

from __future__ import annotations

import jax.numpy as jnp

from ._utils import result_dtype
from .base import WrappedOperator


class Squared(WrappedOperator):
    """Lazy marker representing the positive observable ``A.adjoint @ A``.

    This wrapper intentionally does not implement connected-component kernels.
    It exists so the VQS/local-estimator layer can dispatch on the semantic
    object and choose the appropriate estimator for ``A.adjoint @ A`` without
    forcing operator connectivity materialisation here.
    """

    @property
    def dtype(self) -> jnp.dtype:
        return result_dtype(self.operator)

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def is_diagonal(self) -> bool:
        return bool(getattr(self.operator, "is_diagonal", False))

    @property
    def adjoint(self) -> Squared:
        return self


__all__ = ["Squared"]
