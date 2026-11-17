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


"""Versioned .nqx checkpoints for JAX pytrees and sampled variational states.

Each checkpoint is one atomically published ZIP64 file containing a JSON
manifest and Orbax tensor shards. Temporary staging beside the file is removed
after saving or loading. Earlier directory checkpoints remain readable.
The default state snapshot includes chain endpoints and RNGs, while cached
sample batches are optional. Distributed saves and loads are collective calls
on a writable shared filesystem after JAX distributed initialization.
"""

from .errors import CheckpointError
from .errors import ReconstructionError
from .metadata import FORMAT_VERSION
from .metadata import inspect_checkpoint
from .registry import ComponentCodec
from .registry import register_codec
from .storage import load_pytree
from .storage import load_subtree
from .storage import save_pytree

__all__ = [
    "FORMAT_VERSION",
    "CheckpointError",
    "ComponentCodec",
    "ReconstructionError",
    "inspect_checkpoint",
    "load_pytree",
    "load_subtree",
    "register_codec",
    "save_pytree",
]
