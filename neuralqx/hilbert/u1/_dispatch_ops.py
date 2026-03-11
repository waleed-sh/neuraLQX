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

"""
Backward-compatibility stub.

The dispatch overloads have moved to:

* :mod:`neuralqx.hilbert.u1.operations.random` for ``random_state``
* :mod:`neuralqx.hilbert.u1.operations.flip` for ``flip_state``

This file re-exports both overloads so that existing side-effect imports of
this module continue to install the overloads correctly::

    # old path (still works via u1/__init__.py):
    from neuralqx.hilbert.u1 import _dispatch_ops

    # new canonical path:
    from neuralqx.hilbert.u1 import operations
"""

from .operations.random import random_state  # noqa: F401
from .operations.flip import flip_state  # noqa: F401
