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


import sys as _sys
from importlib import import_module as _import_module

from .core import AbstractVariationalState
from .core import VariationalState
from .estimators import LocalEstimatorOptions
from .estimators import LocalEstimators
from .estimators import OnlineStats
from .estimators import Stats
from .estimators import local_estimator_kernel
from .estimators import local_estimators
from .estimators import online_statistics
from .estimators import register_local_estimator
from .estimators import statistics
from .expectation import expect
from .expectation import expect_and_forces
from .expectation import expect_and_grad
from .expectation import force_to_grad
from .expectation import penalty_expect
from .expectation import penalty_expect_and_forces
from .expectation import penalty_expect_and_grad
from .expectation import register_penalty_expect
from .expectation import register_penalty_expect_and_forces
from .expectation import register_penalty_expect_and_grad

_FLAT_MODULE_ALIASES = {
    "base": _import_module(".core.base", __name__),
    "expect": _import_module(".expectation.values", __name__),
    "forces": _import_module(".expectation.forces", __name__),
    "gradients": _import_module(".expectation.gradients", __name__),
    "kernels": _import_module(".estimators.kernels", __name__),
    "local_estimators": _import_module(".estimators.local", __name__),
    "options": _import_module(".estimators.options", __name__),
    "penalties": _import_module(".estimators.penalties", __name__),
    "state": _import_module(".core.state", __name__),
    "statistics": _import_module(".estimators.statistics", __name__),
}
for _name, _module in _FLAT_MODULE_ALIASES.items():
    _sys.modules.setdefault(f"{__name__}.{_name}", _module)

del _FLAT_MODULE_ALIASES, _import_module, _module, _name, _sys

__all__ = [
    "AbstractVariationalState",
    "LocalEstimatorOptions",
    "LocalEstimators",
    "OnlineStats",
    "Stats",
    "VariationalState",
    "expect",
    "expect_and_forces",
    "expect_and_grad",
    "force_to_grad",
    "local_estimator_kernel",
    "local_estimators",
    "online_statistics",
    "penalty_expect",
    "penalty_expect_and_forces",
    "penalty_expect_and_grad",
    "register_penalty_expect",
    "register_penalty_expect_and_forces",
    "register_penalty_expect_and_grad",
    "register_local_estimator",
    "statistics",
]
