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

from __future__ import annotations

from typing import Literal

from neuralqx.utils.struct import Struct

class AbstractProposalMove(Struct): ...

class UniformSiteUpdate(AbstractProposalMove):
    n_sites: int
    avoid_current: bool
    def __init__(self, n_sites: int = 1, avoid_current: bool = True) -> None: ...

class AdjacentSiteUpdate(AbstractProposalMove):
    n_sites: int
    boundary: Literal["wrap", "clamp"]
    def __init__(
        self,
        n_sites: int = 1,
        boundary: Literal["wrap", "clamp"] = "wrap",
    ) -> None: ...
