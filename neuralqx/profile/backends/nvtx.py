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


"""Process ranges that remain valid when asyncio tasks overlap."""

from importlib import import_module

from ..errors import ProfileError


class NvtxEmitter:
    """Emit process ranges that remain valid across overlapping coroutine lifetimes."""

    def __init__(self):
        """Load the optional NVTX module and select the neuralqx annotation domain."""
        try:
            nvtx = import_module("nvtx")
        except ImportError as error:
            raise ProfileError(
                "NVTX was requested. Install neuralqx[profile-nvidia] or nvtx."
            ) from error
        self.domain = nvtx.get_domain("neuralqx")
        """NVTX domain isolating package annotations from other libraries."""

    def begin(self, name: str, category: str):
        """Start a named process range and return its opaque closing handle.

        Args:
            name: Stable human-readable region name.
            category: Grouping label understood by NVIDIA tools.

        Returns:
            Handle accepted by end, including when other ranges overlap.
        """
        return self.domain.start_range(message=name, category=category)

    def end(self, handle) -> None:
        """Close exactly the process range identified by handle.

        Args:
            handle: Value returned by begin.
        """
        self.domain.end_range(handle)
