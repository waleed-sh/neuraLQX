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

import sys

#
#
#   Version and environment/cfg variables init

from ._version import __version__ as _raw_version

from .configs import cfg
from .configs import _should_init_jax_distributed

if _should_init_jax_distributed():
    import jax

    if not jax.distributed.is_initialized():
        jax.distributed.initialize()
    del jax

#
#
#   Package wide logging

from .debug import initialise as _init_debug

_init_debug()


#
#
#   Imports

from . import utils
from . import graph
from . import hilbert
from . import lqx
from . import operators
from . import gauge_groups
from . import solver
from . import nn
from . import samplers
from . import callbacks
from . import driver
from . import vqs
from . import profile
from ._cite import cite

# Keep top-level version str-compatible for ecosystem tooling while exposing
# strict semantic comparison behaviour through the new version subsystem.
from .utils.module.version import NeuralqxVersion
from .utils.module.version import Version

__version__: NeuralqxVersion = NeuralqxVersion(_raw_version)
version: NeuralqxVersion = __version__
version_info: Version = __version__.version


def __getattr__(name):
    if name == "experimental":
        if not cfg.get("EXPERIMENTAL"):
            raise utils.errors.DeniedExperimentalModuleImportError
        else:
            import importlib

            return importlib.import_module(".experimental", __name__)
    raise AttributeError(f"Module neuralqx has no attribute `{name}`.")


__author__ = "Waleed Sherif (waleed.sherif@gravity.fau.de)"

#
#   Package modules information
#
__all__ = [
    "utils",
    "graph",
    "hilbert",
    "lqx",
    "gauge_groups",
    "solver",
    "nn",
    "operators",
    "samplers",
    "cfg",
    "callbacks",
    "vqs",
    "driver",
    "__version__",
    "__author__",
    "cite",
    "profile",
]

if cfg.get("EXPERIMENTAL"):
    __all__ += [
        "experimental",
    ]
