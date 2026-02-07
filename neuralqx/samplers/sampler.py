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

from typing import Any
from typing import Dict
from typing import Tuple
from typing import List

from .types import SAMPLER_REGISTRY

from .types import MetropolisLocal
from .types import MuLocal
from .types import MetropolisKLocal
from .types import MetropolisHamiltonian
from .types import MetropolisExchange
from .types import Exact
from .types import U1Gauge
from .types import RandomU1Gauge
from .types import U1Plaquette
from .types import Weighted
from .types import PTLocal
from .types import PTExchange
from .types import PTU1Gauge
from .types import PTRandomU1Gauge
from .types import PTU1Plaquette
from .types import PTWeighted

from .builds import build_sampler
from .export import export_info
from ..debug import event


class Sampler:

    def __init__(
        self,
        sampler_type: str,
        number_of_chains: int = 70,
        number_of_sweeps: int = 10,
        machine_pow: int = 2,
        reset_chains: bool = True,
        number_of_samples: int = 250,
        **sampler_kwargs,
    ):
        event(
            msg="SAMPLER_REGISTRY",
            tag="SAMPLER:INIT",
            level=logging.INFO,
            sampler_type=sampler_type,
            number_of_chains=number_of_chains,
            number_of_sweeps=number_of_sweeps,
            machine_pow=machine_pow,
            reset_chains=reset_chains,
            number_of_samples=number_of_samples,
            **sampler_kwargs,
        )

        # keep original string surface
        sampler_key = sampler_type.strip().lower()
        if sampler_key not in SAMPLER_REGISTRY:
            avail = "\n\t\t\t- " + "\n\t\t\t- ".join(sorted(SAMPLER_REGISTRY.keys()))
            raise NotImplementedError(f"""
                The chosen sampler_type `{sampler_type}` is not implemented.
                Available samplers: {avail}
                """)

        Cfg = SAMPLER_REGISTRY[sampler_key]

        # build cfg dataclass (we separate runtime objects like hilbert/graph/hamiltonian)
        if Cfg is Exact:
            # no params
            self.cfg = Cfg()
        else:
            base = dict(
                n_chains_per_rank=number_of_chains,
                sweep_size=number_of_sweeps,
                machine_pow=machine_pow,
                reset_chains=reset_chains,
            )
            self.cfg = Cfg(**(base | sampler_kwargs))

        self.sampler_type = sampler_type
        self.number_of_samples = number_of_samples

        # built artifacts
        self._sampler = None
        self._sampler_g = None
        self._sampler_args: Dict[str, Any] = {}

        # keep these for export_info
        self._number_of_chains = number_of_chains
        self._number_of_sweeps = number_of_sweeps
        self._machine_pow = machine_pow
        self._reset_chains = reset_chains

    def build(
        self,
        hilbert,
        hamiltonian=None,
        graph=None,
    ) -> Tuple[Any, Any, Dict[str, Any]]:

        # helpful identity check (catches autoreload/dual-import problems early)
        cfg_type = type(self.cfg)

        # choose one representative type to test (they all come from the same module)
        from . import types as _types_mod

        if cfg_type.__module__ != _types_mod.__name__:
            raise RuntimeError(
                f"Config object {cfg_type.__name__} comes from module {cfg_type.__module__}, "
                f"but builds.py overloads were registered for {_types_mod.__name__}. "
                "This usually happens if the samplers.types module was imported along two paths "
                "(e.g. with/without package prefix or via an autoreloader). "
                "Ensure all imports use the same package path and restart the interpreter."
            )

        # dispatch purely on config type, runtime objects are kwargs
        s, s2, skwargs = build_sampler(
            self.cfg, hilbert, hamiltonian=hamiltonian, graph=graph
        )
        self._sampler, self._sampler_g, self._sampler_args = s, s2, skwargs
        return s, s2, skwargs

    @property
    def sampler(self):
        return self._sampler

    @property
    def sampler_g(self):
        return self._sampler_g

    def export_info(self) -> Tuple[List[str], List[str]]:
        return export_info(
            sampler_type=self.sampler_type,
            number_of_samples=self.number_of_samples,
            number_of_chains=self._number_of_chains,
            number_of_sweeps=self._number_of_sweeps,
            machine_pow=self._machine_pow,
            reset_chains=self._reset_chains,
        )
