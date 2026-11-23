# Copyright (c) 2026 The neuraLQX and nkDSL Authors - All rights reserved.
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


"""Warnings for recoverable configuration and registration changes."""


class SymbolicWarning(UserWarning):
    """Base class for recoverable symbolic operator warnings."""


class LoweringWarning(SymbolicWarning):
    """Emitted when a script assigns its default lowering again."""


class RegistrationWarning(SymbolicWarning):
    """Emitted when a duplicate registration keeps the existing implementation."""


__all__ = ["LoweringWarning", "RegistrationWarning", "SymbolicWarning"]
