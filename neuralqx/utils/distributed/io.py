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

from .core import bcast


def _tree_map_host(fn, value: Any) -> Any:
    """Apply ``fn`` over a generic pytree with a no-JAX fallback."""

    try:
        import jax  # type: ignore

        return jax.tree_util.tree_map(fn, value)
    except Exception:
        if isinstance(value, dict):
            return {k: _tree_map_host(fn, v) for k, v in value.items()}
        if isinstance(value, list):
            return [_tree_map_host(fn, v) for v in value]
        if isinstance(value, tuple):
            return tuple(_tree_map_host(fn, v) for v in value)
        if isinstance(value, set):
            return {_tree_map_host(fn, v) for v in value}
        return fn(value)


def block_until_ready_tree(value: Any) -> Any:
    """
    Block on device work for all leaves exposing ``block_until_ready``.
    """

    def _block(x):
        fn = getattr(x, "block_until_ready", None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                return x
        return x

    return _tree_map_host(_block, value)


def device_get_tree(value: Any) -> Any:
    """
    Convert all JAX device-backed leaves to host-backed values.
    """

    try:
        import jax  # type: ignore

        return jax.device_get(value)
    except Exception:
        return value


def safe_replicate_for_io(
    value: Any,
    *,
    replicate_to_all_processes: bool = False,
    root: int = 0,
    block_until_ready: bool = True,
) -> Any:
    """
    Safely prepare an arbitrary pytree for host-side I/O.

    Steps:
    - Optionally block device execution.
    - ``device_get`` to host memory.
    - Optionally broadcast host payload to all processes.
    """

    prepared = value
    if block_until_ready:
        prepared = block_until_ready_tree(prepared)

    prepared = device_get_tree(prepared)

    if replicate_to_all_processes:
        prepared = bcast(prepared, root=root)

    return prepared


__all__ = [
    "block_until_ready_tree",
    "device_get_tree",
    "safe_replicate_for_io",
]
