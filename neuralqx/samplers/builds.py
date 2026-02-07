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

from typing import Any
from typing import Dict
from typing import Tuple

from plum import dispatch

import netket as nk

from .types import MetropolisLocal
from .types import MuLocal
from .types import MetropolisKLocal
from .types import MetropolisHamiltonian
from .types import MetropolisExchange
from .types import Exact
from .types import U1Gauge
from .types import RandomU1Gauge
from .types import U1GaugeNonzero
from .types import U1Plaquette
from .types import Weighted
from .types import PTLocal
from .types import PTExchange
from .types import PTU1Gauge
from .types import PTRandomU1Gauge
from .types import PTU1Plaquette
from .types import PTWeighted
from .types import MetropolisMultiHamiltonian
from .types import PTHamiltonianList

# convenience alias
WeightedSamplerRule = nk.sampler.rules.MultipleRules


#
#
#   helpers


def _mk_kwargs_base(cfg, **rt) -> Dict[str, Any]:
    # common fields for (Parallel) Metropolis
    d = dict(
        hilbert=rt["hilbert"],
        n_chains_per_rank=cfg.n_chains_per_rank,
        sweep_size=cfg.sweep_size,
        machine_pow=cfg.machine_pow,
        reset_chains=cfg.reset_chains,
    )
    return d


def _mk_kwargs_pt(cfg, **rt) -> Dict[str, Any]:
    d = _mk_kwargs_base(cfg, **rt)
    d.update(n_replicas=cfg.n_replicas, betas=cfg.betas)
    return d


def _ret(sampler, kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
    # return two identical samplers + kwargs (for logging/backwards compatibility)
    return sampler, sampler, kwargs


#
#
#   Metropolis family


@dispatch
def build_sampler(cfg: MetropolisLocal, hilbert, **rt):
    rule = nk.sampler.rules.LocalRule()
    kw = _mk_kwargs_base(cfg, hilbert=hilbert, **rt) | dict(rule=rule)
    s = nk.sampler.MetropolisSampler(**kw)
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: MuLocal, hilbert, **rt):
    from ..samplers.rules import MuSampler

    rule = MuSampler()
    kw = _mk_kwargs_base(cfg, hilbert=hilbert, **rt) | dict(rule=rule)
    s = nk.sampler.MetropolisSampler(**kw)
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: MetropolisKLocal, hilbert, **rt):
    from ..samplers.rules import MetropolisKLocalRule

    rule = MetropolisKLocalRule(n_flips=cfg.n_flips)
    kw = _mk_kwargs_base(cfg, hilbert=hilbert, **rt) | dict(rule=rule)
    s = nk.sampler.MetropolisSampler(**kw)
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: MetropolisHamiltonian, hilbert, **rt):
    # expects rt["hamiltonian"]
    kw = dict(
        hilbert=hilbert,
        hamiltonian=rt["hamiltonian"],
        n_chains_per_rank=cfg.n_chains_per_rank,
        sweep_size=cfg.sweep_size,
        machine_pow=cfg.machine_pow,
        reset_chains=cfg.reset_chains,
    )
    s = nk.sampler.MetropolisHamiltonian(**kw)
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: MetropolisExchange, hilbert, **rt):
    # expects rt["graph"]
    kw = dict(
        hilbert=hilbert,
        graph=rt["graph"],
        d_max=cfg.d_max,
        n_chains_per_rank=cfg.n_chains_per_rank,
        sweep_size=cfg.sweep_size,
        machine_pow=cfg.machine_pow,
        reset_chains=cfg.reset_chains,
    )
    s = nk.sampler.MetropolisExchange(**kw)
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: Exact, hilbert, **rt):
    kw = dict(hilbert=hilbert)
    s = nk.sampler.ExactSampler(**kw)
    return _ret(s, kw)


#
#
#   Gauge-invariant Metropolis


@dispatch
def build_sampler(cfg: U1Gauge, hilbert, **rt):
    from ..samplers.rules import U1GaugeSampler as Rule

    rule = Rule()
    kw = _mk_kwargs_base(cfg, hilbert=hilbert, **rt) | dict(rule=rule)
    s = nk.sampler.MetropolisSampler(**kw)
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: U1GaugeNonzero, hilbert, **rt):
    from ..samplers.rules import U1GaugeSamplerNonzero as Rule

    rule = Rule()
    kw = _mk_kwargs_base(cfg, hilbert=hilbert, **rt) | dict(rule=rule)
    s = nk.sampler.MetropolisSampler(**kw)
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: RandomU1Gauge, hilbert, **rt):
    from ..samplers.rules import RandomU1GaugeSampler as Rule

    rule = Rule()
    kw = _mk_kwargs_base(cfg, hilbert=hilbert, **rt) | dict(rule=rule)
    s = nk.sampler.MetropolisSampler(**kw)
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: U1Plaquette, hilbert, **rt):
    from ..samplers.rules import U1InvariantPlaquetteSampler as Rule

    rule = Rule()
    kw = _mk_kwargs_base(cfg, hilbert=hilbert, **rt) | dict(rule=rule)
    s = nk.sampler.MetropolisSampler(**kw)
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: MetropolisMultiHamiltonian, hilbert, **rt):
    from ..samplers.rules import MultiHamiltonianRule

    # expects rt["hamiltonian"], and maybe rt['choose_per_chain'] and rt['p_ops']

    operators = rt.get("hamiltonian", cfg.Hamiltonian)
    if operators is None:
        raise ValueError(
            "Metropolis Multi Hamiltonian requires a Hamiltonian (list of operators)."
        )

    choose_per_chain = rt.get(
        "choose_per_chain",
        cfg.choose_per_chain if hasattr(cfg, "choose_per_chain") else False,
    )

    p_ops = rt.get(
        "p_ops",
        cfg.p_ops if hasattr(cfg, "p_ops") else None,
    )

    rule = MultiHamiltonianRule(
        operators=operators,
        choose_per_chain=choose_per_chain,
        p_ops=p_ops,
    )

    rt_clean = {
        k: v
        for k, v in rt.items()
        if k not in ("hamiltonian", "choose_per_chain", "p_ops")
    }

    kw = _mk_kwargs_base(cfg, hilbert=hilbert, **rt_clean) | dict(rule=rule)
    s = nk.sampler.MetropolisSampler(**kw)
    return _ret(s, kw)


#
#
# Weighted (MultipleRules) Metropolis


@dispatch
def build_sampler(cfg: Weighted, hilbert, **rt):
    rule = nk.sampler.rules.MultipleRules(
        probabilities=cfg.probabilities, rules=cfg.rules
    )
    kw = _mk_kwargs_base(cfg, hilbert=hilbert, **rt) | dict(
        probabilities=cfg.probabilities, rules=cfg.rules
    )
    s = nk.sampler.MetropolisSampler(
        hilbert=hilbert,
        n_chains_per_rank=cfg.n_chains_per_rank,
        sweep_size=cfg.sweep_size,
        machine_pow=cfg.machine_pow,
        reset_chains=cfg.reset_chains,
        rule=rule,
    )
    return _ret(s, kw)


#
#
#   Parallel Tempering (PT)


@dispatch
def build_sampler(cfg: PTLocal, hilbert, **rt):
    rule = nk.sampler.rules.LocalRule()
    kw = _mk_kwargs_pt(cfg, hilbert=hilbert, **rt) | dict(rule=rule)
    s = nk.sampler.ParallelTemperingSampler(**kw)
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: PTExchange, hilbert, **rt):
    # expects rt["graph"]
    rule = nk.sampler.rules.ExchangeRule(graph=rt["graph"], d_max=cfg.d_max)
    kw = _mk_kwargs_pt(cfg, hilbert=hilbert, **rt) | dict(rule=rule)
    s = nk.sampler.ParallelTemperingSampler(**kw)
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: PTU1Gauge, hilbert, **rt):
    from ..samplers.rules import U1GaugeSampler as Rule

    rule = Rule()
    kw = _mk_kwargs_pt(cfg, hilbert=hilbert, **rt) | dict(rule=rule)
    s = nk.sampler.ParallelTemperingSampler(**kw)
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: PTRandomU1Gauge, hilbert, **rt):
    from ..samplers.rules import RandomU1GaugeSampler as Rule

    rule = Rule()
    kw = _mk_kwargs_pt(cfg, hilbert=hilbert, **rt) | dict(rule=rule)
    s = nk.sampler.ParallelTemperingSampler(**kw)
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: PTU1Plaquette, hilbert, **rt):
    from ..samplers.rules import U1InvariantPlaquetteSampler as Rule

    rule = Rule()
    kw = _mk_kwargs_pt(cfg, hilbert=hilbert, **rt) | dict(rule=rule)
    s = nk.sampler.ParallelTemperingSampler(**kw)
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: PTWeighted, hilbert, **rt):
    rule = nk.sampler.rules.MultipleRules(
        probabilities=cfg.probabilities, rules=cfg.rules
    )
    kw = _mk_kwargs_pt(cfg, hilbert=hilbert, **rt) | dict(
        probabilities=cfg.probabilities, rules=cfg.rules
    )
    s = nk.sampler.ParallelTemperingSampler(
        hilbert=hilbert,
        rule=rule,
        n_replicas=cfg.n_replicas,
        betas=cfg.betas,
        n_chains_per_rank=cfg.n_chains_per_rank,
        sweep_size=cfg.sweep_size,
        machine_pow=cfg.machine_pow,
        reset_chains=cfg.reset_chains,
    )
    return _ret(s, kw)


@dispatch
def build_sampler(cfg: PTHamiltonianList, hilbert, **rt):
    from ..samplers.rules import MultiHamiltonianRule as Rule

    rule = Rule(
        operators=cfg.Hamiltonian,
        p_ops=cfg.p_ops,
        choose_per_chain=cfg.choose_per_chain,
    )

    kw = _mk_kwargs_pt(cfg, hilbert=hilbert, **rt) | dict(rule=rule)
    s = nk.sampler.ParallelTemperingSampler(**kw)
    return _ret(s, kw)
