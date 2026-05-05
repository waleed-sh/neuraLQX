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

import pytest


def _mk_common(tmod, cls, **extra):
    base = dict(n_chains_per_rank=4, sweep_size=2, machine_pow=3, reset_chains=True)
    return cls(**(base | extra))


def _mk_pt_common(tmod, cls, **extra):
    base = dict(
        n_chains_per_rank=4,
        sweep_size=2,
        machine_pow=3,
        reset_chains=True,
        n_replicas=2,
        betas=[1.0, 0.5],
    )
    return cls(**(base | extra))


def test_build_metropolis_local_constructs_metropolis_sampler_with_localrule(
    sampler_modules, patch_builds_netket
):
    tmod, bmod, _, _ = sampler_modules
    _, rec = patch_builds_netket

    cfg = _mk_common(tmod, tmod.MetropolisLocal)
    hilbert = object()

    s, s2, kw = bmod.build_sampler(cfg, hilbert)

    assert s is s2
    assert rec.calls[0][0] == "LocalRule"
    assert rec.calls[1][0] == "MetropolisSampler"
    assert kw["hilbert"] is hilbert
    assert kw["n_chains_per_rank"] == 4
    assert kw["sweep_size"] == 2
    assert kw["machine_pow"] == 3
    assert kw["reset_chains"] is True
    assert "rule" in kw


def test_build_mu_local_uses_custom_musampler_rule(
    sampler_modules, patch_builds_netket
):
    tmod, bmod, _, _ = sampler_modules
    _, rec = patch_builds_netket

    cfg = _mk_common(tmod, tmod.MuLocal)
    s, s2, kw = bmod.build_sampler(cfg, object())

    assert s is s2
    assert rec.calls[-1][0] == "MetropolisSampler"
    assert kw["rule"].__class__.__name__ == "MuSampler"


def test_build_metropolis_klocal_passes_n_flips_to_rule(
    sampler_modules, patch_builds_netket
):
    tmod, bmod, _, _ = sampler_modules
    _, rec = patch_builds_netket

    cfg = _mk_common(tmod, tmod.MetropolisKLocal, n_flips=6)
    s, _, kw = bmod.build_sampler(cfg, object())

    assert rec.calls[-1][0] == "MetropolisSampler"
    assert kw["rule"].kwargs["n_flips"] == 6


def test_build_metropolis_hamiltonian_uses_runtime_hamiltonian(
    sampler_modules, patch_builds_netket
):
    tmod, bmod, _, _ = sampler_modules
    _, rec = patch_builds_netket

    cfg = _mk_common(tmod, tmod.MetropolisHamiltonian)
    H = object()
    s, _, kw = bmod.build_sampler(cfg, object(), hamiltonian=H)

    assert rec.calls[-1][0] == "MetropolisHamiltonian"
    assert kw["hamiltonian"] is H


def test_build_metropolis_exchange_requires_graph_and_dmax(
    sampler_modules, patch_builds_netket
):
    tmod, bmod, _, _ = sampler_modules
    _, rec = patch_builds_netket

    cfg = _mk_common(tmod, tmod.MetropolisExchange, d_max=9)
    G = object()
    s, _, kw = bmod.build_sampler(cfg, object(), graph=G)

    assert rec.calls[-1][0] == "MetropolisExchange"
    assert kw["graph"] is G
    assert kw["d_max"] == 9


def test_build_exact_sampler_only_passes_hilbert(sampler_modules, patch_builds_netket):
    tmod, bmod, _, _ = sampler_modules
    _, rec = patch_builds_netket

    cfg = tmod.Exact()
    hilbert = object()
    s, _, kw = bmod.build_sampler(cfg, hilbert)

    assert rec.calls[-1][0] == "ExactSampler"
    assert kw == {"hilbert": hilbert}


def test_build_autoregressive_direct_sampler_uses_netket_ardirect(
    sampler_modules, patch_builds_netket
):
    tmod, bmod, _, _ = sampler_modules
    _, rec = patch_builds_netket

    cfg = _mk_common(tmod, tmod.AutoregressiveDirect)
    hilbert = object()
    s, _, kw = bmod.build_sampler(cfg, hilbert)

    assert rec.calls[-1][0] == "ARDirectSampler"
    assert kw == {"hilbert": hilbert}


@pytest.mark.parametrize(
    "cfg_cls, expected_rule_name",
    [
        ("U1Gauge", "U1GaugeSampler"),
        ("U1GaugeNonzero", "U1GaugeSamplerNonzero"),
        ("RandomU1Gauge", "RandomU1GaugeSampler"),
        ("U1Plaquette", "U1InvariantPlaquetteSampler"),
    ],
)
def test_build_u1_family_metropolis_uses_custom_rules(
    sampler_modules, patch_builds_netket, cfg_cls, expected_rule_name
):
    tmod, bmod, _, _ = sampler_modules
    _, rec = patch_builds_netket

    cfg = _mk_common(tmod, getattr(tmod, cfg_cls))
    s, _, kw = bmod.build_sampler(cfg, object())

    assert rec.calls[-1][0] == "MetropolisSampler"
    assert kw["rule"].__class__.__name__ == expected_rule_name


def test_build_metropolis_multi_hamiltonian_overrides_runtime_and_strips_rt_keys(
    sampler_modules, patch_builds_netket
):
    tmod, bmod, _, _ = sampler_modules
    _, rec = patch_builds_netket

    cfg = _mk_common(
        tmod,
        tmod.MetropolisMultiHamiltonian,
        Hamiltonian=("Hcfg",),
        choose_per_chain=False,
        p_ops=None,
    )

    Hrt = ("Hrt", "Hrt2")
    extra = object()

    s, _, kw = bmod.build_sampler(
        cfg,
        object(),
        hamiltonian=Hrt,
        choose_per_chain=True,
        p_ops="pops",
        extra_runtime=extra,
    )

    assert rec.calls[-1][0] == "MetropolisSampler"
    rule = kw["rule"]
    assert rule.__class__.__name__ == "MultiHamiltonianRule"
    assert rule.kwargs["operators"] == Hrt
    assert rule.kwargs["choose_per_chain"] is True
    assert rule.kwargs["p_ops"] == "pops"

    assert "hamiltonian" not in kw
    assert "choose_per_chain" not in kw
    assert "p_ops" not in kw
    assert "extra_runtime" not in kw


def test_build_metropolis_multi_hamiltonian_raises_if_none_provided(
    sampler_modules, patch_builds_netket
):
    tmod, bmod, _, _ = sampler_modules

    cfg = _mk_common(
        tmod,
        tmod.MetropolisMultiHamiltonian,
        Hamiltonian=None,
        choose_per_chain=False,
        p_ops=None,
    )

    with pytest.raises(ValueError) as exc:
        bmod.build_sampler(cfg, object(), hamiltonian=None)

    assert "requires a hamiltonian" in str(exc.value).lower()


def test_build_weighted_builds_multiple_rules_and_passes_probabilities_rules(
    sampler_modules, patch_builds_netket
):
    tmod, bmod, _, _ = sampler_modules
    _, rec = patch_builds_netket

    probs = [0.2, 0.8]
    rules = [object(), object()]
    cfg = _mk_common(tmod, tmod.Weighted, probabilities=probs, rules=rules)

    s, s2, kw = bmod.build_sampler(cfg, object())

    assert s is s2
    assert any(name == "MultipleRules" for name, _ in rec.calls)
    assert rec.calls[-1][0] == "MetropolisSampler"
    assert kw["probabilities"] == probs
    assert kw["rules"] == rules


def test_build_pt_local_constructs_parallel_tempering_sampler(
    sampler_modules, patch_builds_netket
):
    tmod, bmod, _, _ = sampler_modules
    _, rec = patch_builds_netket

    cfg = _mk_pt_common(tmod, tmod.PTLocal)
    s, _, kw = bmod.build_sampler(cfg, object())

    assert rec.calls[0][0] == "LocalRule"
    assert rec.calls[1][0] == "ParallelTemperingSampler"
    assert kw["n_replicas"] == 2
    assert kw["betas"] == [1.0, 0.5]
    assert "rule" in kw


def test_build_pt_exchange_uses_exchange_rule_and_graph(
    sampler_modules, patch_builds_netket
):
    tmod, bmod, _, _ = sampler_modules
    _, rec = patch_builds_netket

    cfg = _mk_pt_common(tmod, tmod.PTExchange, d_max=3)
    G = object()
    s, _, kw = bmod.build_sampler(cfg, object(), graph=G)

    assert any(
        name == "ExchangeRule" and call["graph"] is G and call["d_max"] == 3
        for name, call in rec.calls
    )
    assert rec.calls[-1][0] == "ParallelTemperingSampler"
    assert kw["rule"].__class__.__name__ == "ExchangeRule"
    assert kw["rule"].kwargs["graph"] is G
    assert kw["rule"].kwargs["d_max"] == 3


def test_build_pt_weighted_uses_multiple_rules_and_parallel_tempering_sampler(
    sampler_modules, patch_builds_netket
):
    tmod, bmod, _, _ = sampler_modules
    _, rec = patch_builds_netket

    cfg = _mk_pt_common(
        tmod, tmod.PTWeighted, probabilities=[0.3, 0.7], rules=[object(), object()]
    )
    s, _, kw = bmod.build_sampler(cfg, object())

    assert any(name == "MultipleRules" for name, _ in rec.calls)
    assert rec.calls[-1][0] == "ParallelTemperingSampler"
    assert kw["probabilities"] == [0.3, 0.7]
    assert len(kw["rules"]) == 2


def test_build_pt_hamiltonian_list_uses_multihamiltonian_rule_from_cfg(
    sampler_modules, patch_builds_netket
):
    tmod, bmod, _, _ = sampler_modules
    _, rec = patch_builds_netket

    cfg = _mk_pt_common(
        tmod,
        tmod.PTHamiltonianList,
        Hamiltonian=("H1", "H2"),
        choose_per_chain=True,
        p_ops="pops",
    )

    s, _, kw = bmod.build_sampler(cfg, object())

    assert rec.calls[-1][0] == "ParallelTemperingSampler"
    rule = kw["rule"]
    assert rule.__class__.__name__ == "MultiHamiltonianRule"
    assert rule.kwargs["operators"] == ("H1", "H2")
    assert rule.kwargs["choose_per_chain"] is True
    assert rule.kwargs["p_ops"] == "pops"
