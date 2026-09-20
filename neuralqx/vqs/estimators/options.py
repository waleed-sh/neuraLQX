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


"""Typed option objects for VQS estimator execution."""

from __future__ import annotations

from typing import Any

from neuralqx.utils.struct import Struct
from neuralqx.utils.struct import field


class LocalEstimatorOptions(Struct):
    """Execution policy for local-estimator evaluation.

    The object is intentionally small and static so it can be passed through
    Plum-dispatched estimator kernels without growing a long positional
    signature. ``None`` means "use the subsystem default"; for model chunking
    this is resolved from the variational state before dispatch.
    """

    chunk_size: int | None = field(static=True, default=None)
    operator_sharding: bool | None = field(static=True, default=None)
    operator_streaming: bool | None = field(static=True, default=None)
    operator_streaming_chunk_size: int | None = field(static=True, default=None)

    def __post_init__(self) -> None:
        if self.chunk_size is not None and int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive when provided.")
        if (
            self.operator_streaming_chunk_size is not None
            and int(self.operator_streaming_chunk_size) <= 0
        ):
            raise ValueError(
                "operator_streaming_chunk_size must be positive when provided."
            )

    @classmethod
    def resolve(
        cls,
        vstate: Any,
        *,
        options: LocalEstimatorOptions | None = None,
        chunk_size: int | None = None,
        operator_sharding: bool | None = None,
        operator_streaming: bool | None = None,
        operator_streaming_chunk_size: int | None = None,
        operator_chunk_size: int | None = None,
    ) -> LocalEstimatorOptions:
        """Resolve call-time options against a variational state."""
        if (
            operator_streaming_chunk_size is not None
            and operator_chunk_size is not None
        ):
            raise ValueError(
                "Pass either operator_streaming_chunk_size or operator_chunk_size, not both."
            )
        if operator_streaming_chunk_size is None:
            operator_streaming_chunk_size = operator_chunk_size

        if options is not None:
            if (
                chunk_size is not None
                or operator_sharding is not None
                or operator_streaming is not None
                or operator_streaming_chunk_size is not None
            ):
                raise ValueError(
                    "Pass either options or individual local-estimator options, not both."
                )
            return options.replace(
                chunk_size=(
                    vstate.chunk_size
                    if options.chunk_size is None
                    else options.chunk_size
                )
            )

        return cls(
            chunk_size=vstate.chunk_size if chunk_size is None else chunk_size,
            operator_sharding=operator_sharding,
            operator_streaming=operator_streaming,
            operator_streaming_chunk_size=operator_streaming_chunk_size,
        )


__all__ = ["LocalEstimatorOptions"]
