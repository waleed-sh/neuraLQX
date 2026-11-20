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


"""SU(2) spin network Hilbert space errors.

Canonical definitions live in neuralqx.utils.errors. These imports preserve
existing public exception paths.
"""

from neuralqx.utils.errors import InvalidSU2CutoffError
from neuralqx.utils.errors import InvalidSU2StateLayoutError
from neuralqx.utils.errors import SU2ChannelConstructionFailureError
from neuralqx.utils.errors import SU2ExactEnumerationTooLargeError
from neuralqx.utils.errors import SU2HilbertError
from neuralqx.utils.errors import SU2UnconstrainedNotImplementedError
from neuralqx.utils.errors import UnsupportedRecouplingSchemeError

__all__ = [
    "InvalidSU2CutoffError",
    "InvalidSU2StateLayoutError",
    "SU2ChannelConstructionFailureError",
    "SU2ExactEnumerationTooLargeError",
    "SU2HilbertError",
    "SU2UnconstrainedNotImplementedError",
    "UnsupportedRecouplingSchemeError",
]
