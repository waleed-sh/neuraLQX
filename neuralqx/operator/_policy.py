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


"""Runtime policy resolution for operator execution."""

from __future__ import annotations

from neuralqx.config import config


def resolve_streaming_policy(
    *,
    use_streaming: bool | None,
    chunk_size: int | None,
) -> tuple[bool, int]:
    """Resolve public ``get_conn_padded`` policy arguments."""
    stream = (
        bool(config.get("operator_streaming_enabled"))
        if use_streaming is None
        else bool(use_streaming)
    )
    resolved_chunk = (
        int(config.get("operator_streaming_chunk_size"))
        if chunk_size is None
        else int(chunk_size)
    )
    if resolved_chunk <= 0:
        raise ValueError("operator streaming chunk_size must be positive.")
    return stream, resolved_chunk


__all__ = ["resolve_streaming_policy"]
