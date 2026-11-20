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


"""Explicit application and native profiling with distributed trace collection.

Importing this module does not start collection. Enter Profiler to open a
session, and use span, annotate, event, and counter to contribute application
events. Package profile_* options control new sessions. Existing numerical
execution paths are not automatically instrumented.
"""

from .annotations import annotate
from .annotations import compiled_scope
from .annotations import counter
from .annotations import event
from .annotations import span
from .errors import ProfileError
from .errors import ProfileStateError
from .errors import ProfileTimeoutError
from .options import ProfileOptions
from .schedule import Action
from .schedule import Schedule
from .session import Profiler
from .session import ProfileResult
from .trace.merge import merge_traces

__all__ = [
    "Action",
    "ProfileError",
    "ProfileOptions",
    "ProfileResult",
    "ProfileStateError",
    "ProfileTimeoutError",
    "Profiler",
    "Schedule",
    "annotate",
    "compiled_scope",
    "counter",
    "event",
    "merge_traces",
    "span",
]
