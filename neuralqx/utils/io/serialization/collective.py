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


"""Small control messages shared by participants in collective checkpoint I/O."""

import hashlib
import json

import numpy as np

import jax

from .errors import CheckpointError


def canonical_json(value):
    """Encode deterministic UTF-8 JSON, rejecting non-finite JSON numbers."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def collective_call(function, phase):
    """Run a local preparation step and propagate Python failures to every process.

    All processes must enter the same phases in the same order. This covers
    recoverable Python errors, not a lost process or a failed JAX collective.
    """
    if jax.process_count() == 1:
        return function()
    from jax.experimental import multihost_utils

    result = None
    message = b""
    try:
        result = function()
    except Exception as error:
        message = f"{type(error).__name__}: {error}".encode()[:8191]
    buffer = np.zeros(8192, dtype=np.uint8)
    buffer[: len(message)] = np.frombuffer(message, dtype=np.uint8)
    errors = np.asarray(multihost_utils.process_allgather(buffer, tiled=False))
    failures = [
        f"process {rank}: {bytes(row).rstrip(bytes([0])).decode('utf-8', errors='replace')}"
        for rank, row in enumerate(errors)
        if np.any(row)
    ]
    if failures:
        raise CheckpointError(f"{phase} failed. " + " | ".join(failures))
    return result


def agree(value, phase):
    """Require identical control metadata without gathering any tensor payloads."""
    if jax.process_count() == 1:
        return
    from jax.experimental import multihost_utils

    digest = np.frombuffer(
        hashlib.sha256(canonical_json(value)).digest(), dtype=np.uint8
    )
    hashes = np.asarray(multihost_utils.process_allgather(digest, tiled=False))
    if not np.all(hashes == hashes[0]):
        raise CheckpointError(f"Processes disagree on {phase}.")


def primary_call(function, phase):
    """Execute a JSON-valued filesystem operation once and broadcast its result."""
    result = collective_call(
        lambda: function() if jax.process_index() == 0 else None, phase
    )
    if jax.process_count() == 1:
        return result
    from jax.experimental import multihost_utils

    data = canonical_json(result) if jax.process_index() == 0 else b""
    size = np.asarray(len(data), dtype=np.int64)
    size = int(multihost_utils.broadcast_one_to_all(size))
    buffer = np.zeros(size, dtype=np.uint8)
    if jax.process_index() == 0:
        buffer[:] = np.frombuffer(data, dtype=np.uint8)
    buffer = multihost_utils.broadcast_one_to_all(buffer)
    return json.loads(bytes(buffer))
