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

from __future__ import annotations

import logging

from typing import Optional
from typing import Dict
from typing import Any
from typing import Tuple
from typing import List
from typing import Union

from .types import OPTIMIZER_REGISTRY
from .types import SCHEDULE_REGISTRY
from .types import LearningRateSpec
from .builds import build_optimizer
from .export import export_info
from ..debug import event


class Optimizer:

    def __init__(
        self,
        optimizer_name: str,
        learning_rate: Optional[float] = None,
        scheduler_type: Optional[str] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        **optimizer_kwargs,
    ):
        event(
            msg="OPTIMIZER_REGISTRY",
            tag="OPTIMIZER:INIT",
            level=logging.INFO,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            scheduler_type=scheduler_type,
        )

        self.optimizer_name = optimizer_name.lower()
        if self.optimizer_name not in OPTIMIZER_REGISTRY:
            raise ValueError(
                f"Unknown optimizer `{optimizer_name}`. "
                f"Available: {list(OPTIMIZER_REGISTRY.keys())}"
            )
        # Build optimizer config object
        OptCfg = OPTIMIZER_REGISTRY[self.optimizer_name]
        self.optimizer_cfg = OptCfg(**optimizer_kwargs)

        # Build LR spec (float or schedule config)
        if scheduler_type is None:
            self.lr_spec: Union[float, LearningRateSpec] = (
                learning_rate if learning_rate is not None else 1e-3
            )
        else:
            key = scheduler_type.lower()
            if key not in SCHEDULE_REGISTRY:
                raise ValueError(
                    f"Unknown scheduler `{scheduler_type}`. "
                    f"Available: {list(SCHEDULE_REGISTRY.keys())}"
                )
            SchedCfg = SCHEDULE_REGISTRY[key]
            self.lr_spec = SchedCfg(**(scheduler_kwargs or {}))

        self._optimizer = None

    def build(self):
        self._optimizer = build_optimizer(self.optimizer_cfg, self.lr_spec)
        return self._optimizer

    @property
    def optimizer(self):
        return self._optimizer if self._optimizer is not None else self.build()

    def export_info(self) -> Tuple[List[str], List[str]]:
        return export_info(self.optimizer_cfg, self.lr_spec)
