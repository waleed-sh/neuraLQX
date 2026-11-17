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


"""Errors raised before a checkpoint can change a live object."""

from neuralqx.utils.errors import SerializationError


class CheckpointError(SerializationError):
    """An incomplete, incompatible, or malformed .nqx checkpoint was encountered."""

    message: str
    """Plain diagnostic text suitable for checkpoint metadata or collective errors."""

    def __init__(self, message):
        """Keep the plain reason alongside the standard package error formatting."""
        self.message = message
        super().__init__(message)


class ReconstructionError(CheckpointError):
    """A saved component needs an explicitly supplied template or registered codec."""
