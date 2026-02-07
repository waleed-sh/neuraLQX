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

from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Sequence

from netket.utils import struct

Betas = Union[str, Sequence[float]]


#
#
#   base hyperparameters shared by most samplers


@dataclass(frozen=True)
class Common:
    n_chains_per_rank: int
    sweep_size: int
    machine_pow: int
    reset_chains: bool


@dataclass(frozen=True)
class PTCommon(Common):
    n_replicas: int
    betas: Betas


#
#
#   Metropolis family


@dataclass(frozen=True)
class MetropolisLocal(Common):
    pass


@dataclass(frozen=True)
class MuLocal(Common):
    pass


@dataclass(frozen=True)
class MetropolisKLocal(Common):
    n_flips: int


@dataclass(frozen=True)
class MetropolisHamiltonian(Common):
    # hamiltonian is runtime-only, not stored here
    pass


@dataclass(frozen=True)
class MetropolisMultiHamiltonian(Common):
    Hamiltonian: tuple = struct.field(pytree_node=False)
    choose_per_chain: bool = struct.field(pytree_node=False, default=False)
    p_ops: object | None = struct.field(pytree_node=False, default=None)


@dataclass(frozen=True)
class MetropolisExchange(Common):
    d_max: int
    # graph is runtime-only, not stored here


@dataclass(frozen=True)
class Exact:
    pass


#
#
#   gauge-invariant Metropolis (rules supplied internally)


@dataclass(frozen=True)
class U1Gauge(Common):
    pass


@dataclass(frozen=True)
class U1GaugeNonzero(Common):
    pass


@dataclass(frozen=True)
class RandomU1Gauge(Common):
    pass


@dataclass(frozen=True)
class U1Plaquette(Common):
    pass


#
#
#   weighted (MultipleRules) Metropolis


@dataclass(frozen=True)
class Weighted(Common):
    probabilities: List[float]
    # nk.sampler.rules.Rule-like (e.g., U1GaugeSampler(), ...)
    rules: List[Any]


#
#
#   parallel Tempering (PT) variants


@dataclass(frozen=True)
class PTLocal(PTCommon):
    pass


@dataclass(frozen=True)
class PTExchange(PTCommon):
    d_max: int


@dataclass(frozen=True)
class PTU1Gauge(PTCommon):
    pass


@dataclass(frozen=True)
class PTRandomU1Gauge(PTCommon):
    pass


@dataclass(frozen=True)
class PTU1Plaquette(PTCommon):
    pass


@dataclass(frozen=True)
class PTWeighted(PTCommon):
    probabilities: List[float]
    rules: List[Any]


@dataclass(frozen=True)
class PTHamiltonianList(PTCommon):
    Hamiltonian: tuple = struct.field(pytree_node=False)
    choose_per_chain: bool = struct.field(pytree_node=False, default=False)
    p_ops: object | None = struct.field(pytree_node=False, default=None)


#
#
#   registry (string -> config class) with aliases
#   we keep both the previous "PT ..." and human names used in Solver.required_kwargs


SAMPLER_REGISTRY: Dict[str, Any] = {
    # Metropolis family
    "metropolis local": MetropolisLocal,
    "mu sampler": MuLocal,
    "metropolis k local": MetropolisKLocal,
    "metropolis hamiltonian": MetropolisHamiltonian,
    "metropolis multi hamiltonian": MetropolisMultiHamiltonian,
    "metropolis exchange": MetropolisExchange,
    "exact sampler": Exact,
    "u1 gauge sampler": U1Gauge,
    "nonzero u1 gauge sampler": U1GaugeNonzero,
    "random u1 gauge sampler": RandomU1Gauge,
    "u1 plaquette sampler": U1Plaquette,
    "weighted sampler": Weighted,
    # PT aliases (old)
    "pt metropolis local": PTLocal,
    "pt metropolis exchange": PTExchange,
    "pt u1 gauge sampler": PTU1Gauge,
    "pt random u1 gauge sampler": PTRandomU1Gauge,
    "pt u1 plaquette sampler": PTU1Plaquette,
    "pt weighted sampler": PTWeighted,
    "pt hamiltonian list": PTHamiltonianList,
    # PT aliases (new / humanised like in Solver.required_kwargs)
    "parallel tempering local": PTLocal,
    "parallel tempering exchange": PTExchange,
    "parallel tempering u1 gauge sampler": PTU1Gauge,
    "parallel tempering random u1 gauge sampler": PTRandomU1Gauge,
    "parallel tempering u1 plaquette sampler": PTU1Plaquette,
    "parallel tempering weighted sampler": PTWeighted,
    "parallel tempering hamiltonian list": PTHamiltonianList,
}
