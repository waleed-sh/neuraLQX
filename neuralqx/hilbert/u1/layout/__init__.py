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
U(1) Hilbert-space layout subpackage.

Concrete layout descriptor for U(1) gauge-copy Hilbert spaces.

Public API
----------
:class:`StridedGaugeCopyLayout`
    Strided block layout: ``σ[g*E + e]`` with gauge copy ``g`` and edge ``e``.

:class:`GaugeCoord`
    Structured coordinate ``(gauge_copy, edge_index)`` for the strided layout.
"""

from .strided import StridedGaugeCopyLayout
from .strided import GaugeCoord
from .strided import GaugeLayout  # deprecated alias

__all__ = [
    "StridedGaugeCopyLayout",
    "GaugeCoord",
    "GaugeLayout",  # deprecated
]
