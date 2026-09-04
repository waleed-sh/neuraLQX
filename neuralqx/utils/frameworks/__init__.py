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


"""Framework adapters for variational neural models."""

from .base import AbstractModelFramework
from .base import FrameworkApply
from .base import FrameworkMerge
from .base import Variables
from .base import static_training_kwargs
from .base import variables_dict
from .callable import CallableFramework
from .factory import as_framework
from .flax import FlaxLinenFramework
from .flax import FlaxNNXFramework

__all__ = [
    "AbstractModelFramework",
    "CallableFramework",
    "FlaxLinenFramework",
    "FlaxNNXFramework",
    "FrameworkApply",
    "FrameworkMerge",
    "Variables",
    "as_framework",
    "static_training_kwargs",
    "variables_dict",
]
