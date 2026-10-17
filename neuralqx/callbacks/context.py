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


"""Runtime callback context objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import MutableMapping


@dataclass(slots=True)
class CallbackContext:
    """Mutable view of the currently executing driver event."""

    driver: Any
    step: int
    log_data: MutableMapping[str, Any]
    update: Any = None
    accepted: bool = False
    error: BaseException | None = None

    @property
    def state(self) -> Any:
        return self.driver.state

    @property
    def loss(self) -> Any:
        return self.driver.loss_stats

    @property
    def loss_name(self) -> str:
        return self.driver.loss_name

    def reset_step(
        self,
        *,
        step: int,
        log_data: MutableMapping[str, Any],
    ) -> None:
        self.step = int(step)
        self.log_data = log_data
        self.update = None
        self.accepted = False
        self.error = None


__all__ = ["CallbackContext"]
