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


#
#
#   Version and environment/cfg variables init

from ._cite import cite
from ._version import __version__ as _raw_version

from .config import config
from .config import get_static

#
#
#   Imports

from . import graph
from . import hilbert
from . import operator
from . import jax
from . import utils
from . import optimizer
from . import vqs

# Keep top-level version str-compatible for ecosystem tooling while exposing
# strict semantic comparison behaviour through the new version subsystem.
from .utils.module.version import NeuralqxVersion
from .utils.module.version import Version

__version__: NeuralqxVersion = NeuralqxVersion(_raw_version)
version: NeuralqxVersion = __version__
version_info: Version = __version__.version
del _raw_version

#
#
#   Public API

__all__ = [
    "__version__",
    "cite",
    "config",
    "get_static",
    "utils",
    "version",
    "version_info",
    "graph",
    "hilbert",
    "operator",
    "jax",
    "optimizer",
    "vqs",
]
