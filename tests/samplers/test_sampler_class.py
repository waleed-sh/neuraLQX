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


def test_sampler_init_unknown_type_raises_not_implemented(sampler_modules):
    _, _, smod, _ = sampler_modules
    with pytest.raises(NotImplementedError) as exc:
        smod.Sampler("does-not-exist")
    msg = str(exc.value).lower()
    assert "not implemented" in msg
    assert "available samplers" in msg


def test_sampler_init_strips_and_lowercases_key(sampler_modules):
    tmod, _, smod, _ = sampler_modules
    s = smod.Sampler(
        "  metropolis local  ",
        number_of_chains=11,
        number_of_sweeps=3,
        machine_pow=7,
        reset_chains=False,
    )
    assert isinstance(s.cfg, tmod.MetropolisLocal)
    assert s.cfg.n_chains_per_rank == 11
    assert s.cfg.sweep_size == 3
    assert s.cfg.machine_pow == 7
    assert s.cfg.reset_chains is False


def test_sampler_init_exact_has_no_common_fields(sampler_modules):
    tmod, _, smod, _ = sampler_modules
    s = smod.Sampler("Exact Sampler")
    assert isinstance(s.cfg, tmod.Exact)
    assert not hasattr(s.cfg, "n_chains_per_rank")


def test_sampler_init_accepts_sampler_specific_kwargs(sampler_modules):
    tmod, _, smod, _ = sampler_modules
    s = smod.Sampler("metropolis k local", n_flips=9, number_of_chains=4)
    assert isinstance(s.cfg, tmod.MetropolisKLocal)
    assert s.cfg.n_flips == 9
    assert s.cfg.n_chains_per_rank == 4


def test_sampler_build_caches_results_and_returns_triplet(monkeypatch, sampler_modules):
    _, _, smod, _ = sampler_modules

    sentinel_s = object()
    sentinel_s2 = object()
    sentinel_kwargs = {"x": 123}

    h = object()
    H = object()
    G = object()

    def fake_build_sampler(cfg, hilbert, **rt):
        assert hilbert is h
        assert rt["hamiltonian"] is H
        assert rt["graph"] is G
        return sentinel_s, sentinel_s2, sentinel_kwargs

    monkeypatch.setattr(smod, "build_sampler", fake_build_sampler, raising=True)

    sampler = smod.Sampler("metropolis local")
    out = sampler.build(h, hamiltonian=H, graph=G)

    assert out == (sentinel_s, sentinel_s2, sentinel_kwargs)
    assert sampler.sampler is sentinel_s
    assert sampler.sampler_g is sentinel_s2
    assert sampler._sampler_args == sentinel_kwargs


def test_sampler_build_identity_check_raises_on_foreign_cfg_module(sampler_modules):
    _, _, smod, _ = sampler_modules

    class ForeignCfg:
        __module__ = "some.other.module"

    sampler = smod.Sampler("metropolis local")

    sampler.cfg = ForeignCfg()

    with pytest.raises(RuntimeError) as exc:
        sampler.build(object())

    msg = str(exc.value).lower()
    assert (
        "imported along two paths" in msg or "autoreload" in msg or "dual-import" in msg
    )
