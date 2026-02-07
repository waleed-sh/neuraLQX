#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from __future__ import annotations

import importlib
import types
from dataclasses import dataclass
import pytest


def _import_optimizer_module(mod_name: str):
    try:
        return importlib.import_module(f"neuralqx.optimizer.{mod_name}")
    except ModuleNotFoundError:
        return importlib.import_module(f"neuralqx.optimizers.{mod_name}")


class _Recorder:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def record(self, name: str, **kwargs):
        self.calls.append((name, kwargs))


@dataclass
class _FakeGT:

    name: str
    kwargs: dict


@pytest.fixture
def optimizer_modules():

    tmod = _import_optimizer_module("types")
    smod = _import_optimizer_module("schedule")
    bmod = _import_optimizer_module("builds")
    emod = _import_optimizer_module("export")
    omod = _import_optimizer_module("optimizer")
    return tmod, smod, bmod, emod, omod


@pytest.fixture
def fake_optax():

    rec = _Recorder()

    def _mk(name: str):
        def _f(**kwargs):
            rec.record(name, **kwargs)
            return _FakeGT(name=name, kwargs=kwargs)

        return _f

    def exponential_decay(**kwargs):
        rec.record("exponential_decay", **kwargs)

        def sched(step):
            return ("exponential_decay", kwargs, step)

        return sched

    def cosine_decay_schedule(**kwargs):
        rec.record("cosine_decay_schedule", **kwargs)

        def sched(step):
            return ("cosine_decay_schedule", kwargs, step)

        return sched

    def linear_schedule(**kwargs):
        rec.record("linear_schedule", **kwargs)

        def sched(step):
            return ("linear_schedule", kwargs, step)

        return sched

    optax_mod = types.SimpleNamespace(
        exponential_decay=exponential_decay,
        cosine_decay_schedule=cosine_decay_schedule,
        linear_schedule=linear_schedule,
        adam=_mk("adam"),
        sgd=_mk("sgd"),
        adagrad=_mk("adagrad"),
        rmsprop=_mk("rmsprop"),
        adabelief=_mk("adabelief"),
        amsgrad=_mk("amsgrad"),
        yogi=_mk("yogi"),
    )

    return optax_mod, rec


@pytest.fixture
def patch_optax(monkeypatch, optimizer_modules, fake_optax):

    _tmod, smod, bmod, _emod, _omod = optimizer_modules
    optax_mod, rec = fake_optax
    monkeypatch.setattr(smod, "optax", optax_mod, raising=True)
    monkeypatch.setattr(bmod, "optax", optax_mod, raising=True)
    return optax_mod, rec
