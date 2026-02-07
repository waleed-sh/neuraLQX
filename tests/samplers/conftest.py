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

import importlib
import sys
import types
import pytest


def _import_sampler_module(mod_name: str):
    try:
        return importlib.import_module(f"neuralqx.samplers.{mod_name}")
    except ModuleNotFoundError:
        return importlib.import_module(f"neuralqx.sampler.{mod_name}")


class _Recorder:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def record(self, name: str, **kwargs):
        self.calls.append((name, kwargs))


class _FakeSamplerBase:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeRuleBase:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_rules_module(monkeypatch):

    mod = types.ModuleType("neuralqx.samplers.rules")

    class MuSampler(_FakeRuleBase):
        def __init__(self):
            super().__init__()

    class MetropolisKLocalRule(_FakeRuleBase):
        def __init__(self, n_flips):
            super().__init__(n_flips=n_flips)

    class U1GaugeSampler(_FakeRuleBase):
        def __init__(self):
            super().__init__()

    class U1GaugeSamplerNonzero(_FakeRuleBase):
        def __init__(self):
            super().__init__()

    class RandomU1GaugeSampler(_FakeRuleBase):
        def __init__(self):
            super().__init__()

    class U1InvariantPlaquetteSampler(_FakeRuleBase):
        def __init__(self):
            super().__init__()

    class MultiHamiltonianRule(_FakeRuleBase):
        def __init__(self, operators, choose_per_chain=False, p_ops=None):
            super().__init__(
                operators=operators, choose_per_chain=choose_per_chain, p_ops=p_ops
            )

    mod.MuSampler = MuSampler
    mod.MetropolisKLocalRule = MetropolisKLocalRule
    mod.U1GaugeSampler = U1GaugeSampler
    mod.U1GaugeSamplerNonzero = U1GaugeSamplerNonzero
    mod.RandomU1GaugeSampler = RandomU1GaugeSampler
    mod.U1InvariantPlaquetteSampler = U1InvariantPlaquetteSampler
    mod.MultiHamiltonianRule = MultiHamiltonianRule

    monkeypatch.setitem(sys.modules, "neuralqx.samplers.rules", mod)
    return mod


@pytest.fixture
def sampler_modules(fake_rules_module):
    """Import modules under test."""
    tmod = _import_sampler_module("types")
    bmod = _import_sampler_module("builds")
    smod = _import_sampler_module("sampler")
    emod = _import_sampler_module("export")
    return tmod, bmod, smod, emod


@pytest.fixture
def fake_netket():
    rec = _Recorder()

    class LocalRule(_FakeRuleBase):
        def __init__(self):
            super().__init__()
            rec.record("LocalRule")

    class ExchangeRule(_FakeRuleBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            rec.record("ExchangeRule", **kwargs)

    class MultipleRules(_FakeRuleBase):
        def __init__(self, probabilities, rules):
            super().__init__(probabilities=probabilities, rules=rules)
            rec.record("MultipleRules", probabilities=probabilities, rules=rules)

    class MetropolisSampler(_FakeSamplerBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            rec.record("MetropolisSampler", **kwargs)

    class ParallelTemperingSampler(_FakeSamplerBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            rec.record("ParallelTemperingSampler", **kwargs)

    class MetropolisHamiltonian(_FakeSamplerBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            rec.record("MetropolisHamiltonian", **kwargs)

    class MetropolisExchange(_FakeSamplerBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            rec.record("MetropolisExchange", **kwargs)

    class ExactSampler(_FakeSamplerBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            rec.record("ExactSampler", **kwargs)

    nk = types.SimpleNamespace(
        sampler=types.SimpleNamespace(
            MetropolisSampler=MetropolisSampler,
            ParallelTemperingSampler=ParallelTemperingSampler,
            MetropolisHamiltonian=MetropolisHamiltonian,
            MetropolisExchange=MetropolisExchange,
            ExactSampler=ExactSampler,
            rules=types.SimpleNamespace(
                LocalRule=LocalRule,
                ExchangeRule=ExchangeRule,
                MultipleRules=MultipleRules,
            ),
        )
    )
    return nk, rec


@pytest.fixture
def patch_builds_netket(monkeypatch, sampler_modules, fake_netket):

    _tmod, bmod, _smod, _emod = sampler_modules
    nk, rec = fake_netket
    monkeypatch.setattr(bmod, "nk", nk, raising=True)
    return nk, rec
