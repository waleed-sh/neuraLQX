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


"""Runtime logging utilities for neuraLQX drivers."""

from .adapters import CallableLogger
from .adapters import LoggerList
from .adapters import as_logger
from .base import AbstractLogger
from .base import Logger
from .history import HistorySeries
from .history import append_to_history
from .history import flatten_history
from .plotting import plot_history
from .runtime import RuntimeLogger

RuntimeLog = RuntimeLogger

__all__ = [
    "AbstractLogger",
    "CallableLogger",
    "HistorySeries",
    "Logger",
    "LoggerList",
    "RuntimeLog",
    "RuntimeLogger",
    "append_to_history",
    "as_logger",
    "flatten_history",
    "plot_history",
]
