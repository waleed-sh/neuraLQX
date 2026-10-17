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


"""Control signals returned by callback hooks."""

from __future__ import annotations

from dataclasses import dataclass

from neuralqx.utils.typing import ControlLike


@dataclass(frozen=True, slots=True)
class CallbackControl:
    """A small explicit control message emitted by callbacks.

    Callbacks can return one of these objects to request that the current
    parameter update is skipped or that the run stops gracefully. Returning
    ``None`` means "continue".
    """

    stop: bool = False
    skip_update: bool = False
    reason: str | None = None

    @classmethod
    def continue_(cls) -> CallbackControl:
        return cls()

    @classmethod
    def stop_run(cls, reason: str | None = None) -> CallbackControl:
        return cls(stop=True, reason=reason)

    @classmethod
    def skip_parameter_update(cls, reason: str | None = None) -> CallbackControl:
        return cls(skip_update=True, reason=reason)

    def merge(self, other: CallbackControl) -> CallbackControl:
        reason = self.reason if self.reason is not None else other.reason
        return CallbackControl(
            stop=self.stop or other.stop,
            skip_update=self.skip_update or other.skip_update,
            reason=reason,
        )


def as_control(value: ControlLike) -> CallbackControl:
    """Normalize hook return values into a :class:`CallbackControl`."""
    if value is None or value is True:
        return CallbackControl.continue_()
    if value is False:
        return CallbackControl.stop_run()
    if isinstance(value, CallbackControl):
        return value
    raise TypeError(f"Invalid callback control value {value!r}.")


__all__ = ["CallbackControl", "ControlLike", "as_control"]
