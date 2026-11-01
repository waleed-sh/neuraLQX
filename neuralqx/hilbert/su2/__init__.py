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

# TODO: this is just a mess right now, needs to be restructured or have a better abstraction...

"""Gauge-invariant SU(2) spin-network Hilbert spaces."""

from .constraints import GaugeInvariantSU2Constraint
from .constraints import SU2GaugeConstraint
from .errors import InvalidSU2CutoffError
from .errors import InvalidSU2StateLayoutError
from .errors import SU2ChannelConstructionFailureError
from .errors import SU2ExactEnumerationTooLargeError
from .errors import SU2HilbertError
from .errors import SU2UnconstrainedNotImplementedError
from .errors import UnsupportedRecouplingSchemeError
from .layout import SpinNetworkCoord
from .layout import SpinNetworkLayout
from .layout import SpinNetworkView
from .layout import SU2SiteInfo
from .layout import channel_max_label
from .metadata import SU2GraphMetadata
from .metadata import compile_su2_graph_metadata
from .peter_weyl import SU2PeterWeylHilbert
from .proposals import AbstractSU2Move
from .proposals import EdgeSpinHeatbathMove
from .proposals import LoopFundamentalMove
from .proposals import MixedSU2Move
from .proposals import VertexRecouplingMove
from .recoupling import all_left_comb_channels
from .recoupling import allowed_couplings
from .recoupling import count_left_comb_channels
from .recoupling import coupling_ok
from .recoupling import coupling_ok_py
from .recoupling import first_left_comb_channels
from .recoupling import left_comb_valid
from .recoupling import left_comb_valid_py
from .spaces import GaugeInvariantSU2Hilbert
from .spaces import SpinNetworkHilbert
from .spaces import SU2GaugeInvariantHilbert
from .spaces import SU2Hilbert

__all__ = [
    "AbstractSU2Move",
    "EdgeSpinHeatbathMove",
    "GaugeInvariantSU2Constraint",
    "GaugeInvariantSU2Hilbert",
    "InvalidSU2CutoffError",
    "InvalidSU2StateLayoutError",
    "LoopFundamentalMove",
    "MixedSU2Move",
    "SU2ChannelConstructionFailureError",
    "SU2ExactEnumerationTooLargeError",
    "SU2GaugeConstraint",
    "SU2GaugeInvariantHilbert",
    "SU2GraphMetadata",
    "SU2Hilbert",
    "SU2HilbertError",
    "SU2PeterWeylHilbert",
    "SU2SiteInfo",
    "SU2UnconstrainedNotImplementedError",
    "SpinNetworkCoord",
    "SpinNetworkHilbert",
    "SpinNetworkLayout",
    "SpinNetworkView",
    "UnsupportedRecouplingSchemeError",
    "VertexRecouplingMove",
    "all_left_comb_channels",
    "allowed_couplings",
    "channel_max_label",
    "compile_su2_graph_metadata",
    "count_left_comb_channels",
    "coupling_ok",
    "coupling_ok_py",
    "first_left_comb_channels",
    "left_comb_valid",
    "left_comb_valid_py",
]
