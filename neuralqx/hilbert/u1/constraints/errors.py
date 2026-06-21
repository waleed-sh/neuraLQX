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


"""Errors raised by U(1) gauge-fixing code.

The exceptions in this module distinguish invalid user-facing gauge-fixing
specifications from dependency cycles discovered while ordering constructive
relations. They are public so callers can report configuration errors without
matching error-message strings.
"""

# TODO: maybe move to main errors?

from __future__ import annotations


class U1GaugeFixingError(ValueError):
    """Base error for malformed U(1) gauge-fixing input.

    This exception covers invalid edge references, duplicate or inconsistent
    relation data, and topology metadata that cannot define a constructive
    gauge-fixing slice. It subclasses :class:`ValueError` because these
    failures indicate invalid user configuration rather than numerical
    execution failure. Catch this base class when all U(1) gauge-fixing
    configuration problems should be handled in the same way.
    """


class U1CyclicGaugeFixingError(U1GaugeFixingError):
    """Gauge-fixing relations cannot be ordered constructively.

    A constructive U(1) gauge fixing requires every slave edge to be
    reconstructible from free edges or from earlier slave edges. This exception
    is raised when the dependency graph contains a cycle, meaning no
    topological reconstruction order exists. The caller must change at least
    one relation to break the cycle.
    """
