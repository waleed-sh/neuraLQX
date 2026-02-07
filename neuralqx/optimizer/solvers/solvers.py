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
from typing import Tuple
from typing import List

from .types import SOLVERS_REGISTRY
from .builds import build_solver
from .export import export_info
from ...debug import event


class Solvers:

    def __init__(
        self,
        solver_name: str,
        **solver_kwargs,
    ):
        event(
            msg="LR_SOLVER_REGISTRY",
            tag="SOLVER:INIT",
            level=logging.INFO,
            solver_name=solver_name,
            **solver_kwargs,
        )

        self.solver_name = solver_name.lower()
        if self.solver_name not in SOLVERS_REGISTRY:
            raise ValueError(
                f"Unknown solver `{solver_name}`. "
                f"Available: {list(SOLVERS_REGISTRY.keys())}"
            )
        # Build optimizer config object
        SolverCfg = SOLVERS_REGISTRY[self.solver_name]
        self.solver_cfg = SolverCfg(**solver_kwargs)

        self._solver = None

    def build(self):
        self._solver = build_solver(self.solver_cfg)
        return self._solver

    @property
    def solver(self):
        return self._solver if self._solver is not None else self.build()

    def export_info(self) -> Tuple[List[str], List[str]]:
        return export_info(self.solver_cfg)
