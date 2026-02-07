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

import pytest


def test_build_optimizer_adam_wires_optax_correctly(patch_optax, optimizer_modules):
    tmod, _smod, bmod, *_ = optimizer_modules
    _optax, rec = patch_optax

    cfg = tmod.Adam(b1=0.8, b2=0.88, eps=1e-6)
    gt = bmod.build_optimizer(cfg, 0.01)

    assert gt.name == "adam"
    name, kwargs = rec.calls[-1]
    assert name == "adam"
    assert kwargs["learning_rate"] == 0.01
    assert kwargs["b1"] == 0.8
    assert kwargs["b2"] == 0.88
    assert kwargs["eps"] == 1e-6


@pytest.mark.parametrize(
    "cfg, optax_name, extra_asserts",
    [
        (
            ("SGD", dict(momentum=0.2, nesterov=True)),
            "sgd",
            lambda kw: (kw["momentum"] == 0.2 and kw["nesterov"] is True),
        ),
        (
            ("Momentum", dict(momentum=0.9, nesterov=False)),
            "sgd",
            lambda kw: (kw["momentum"] == 0.9 and kw["nesterov"] is False),
        ),
        (
            ("Adagrad", dict(eps=1e-5, initial_accumulator_value=0.3)),
            "adagrad",
            lambda kw: (kw["eps"] == 1e-5 and kw["initial_accumulator_value"] == 0.3),
        ),
        (
            ("RMSProp", dict(decay=0.8, eps=1e-6, centered=True)),
            "rmsprop",
            lambda kw: (
                kw["decay"] == 0.8 and kw["eps"] == 1e-6 and kw["centered"] is True
            ),
        ),
        (
            ("AdaBelief", dict(b1=0.7, b2=0.77, eps=1e-9, eps_root=1e-10)),
            "adabelief",
            lambda kw: (kw["b1"] == 0.7 and kw["b2"] == 0.77),
        ),
        (
            ("AMSGrad", dict(b1=0.6, b2=0.66, eps=1e-8, eps_root=0.1, mu_dtype=None)),
            "amsgrad",
            lambda kw: (kw["eps_root"] == 0.1),
        ),
        (
            ("Yogi", dict(b1=0.5, b2=0.55, eps=1e-4)),
            "yogi",
            lambda kw: (kw["eps"] == 1e-4),
        ),
    ],
)
def test_build_optimizer_all_configs_wires_optax(
    patch_optax, optimizer_modules, cfg, optax_name, extra_asserts
):
    tmod, _smod, bmod, *_ = optimizer_modules
    _optax, rec = patch_optax

    cfg_name, kwargs = cfg
    Cfg = getattr(tmod, cfg_name)
    obj = Cfg(**kwargs)

    gt = bmod.build_optimizer(obj, 0.123)
    assert gt.name == optax_name

    name, called_kwargs = rec.calls[-1]
    assert name == optax_name
    assert called_kwargs["learning_rate"] == 0.123
    assert extra_asserts(called_kwargs)


def test_build_optimizer_lr_spec_calls_make_learning_rate(
    monkeypatch, patch_optax, optimizer_modules
):

    tmod, _smod, bmod, *_ = optimizer_modules
    _optax, rec = patch_optax

    def sentinel_schedule(step: int) -> float:
        return 0.5

    called = {"n": 0}

    def fake_make_learning_rate(spec):
        called["n"] += 1
        return sentinel_schedule

    monkeypatch.setattr(
        bmod, "make_learning_rate", fake_make_learning_rate, raising=True
    )

    cfg = tmod.SGD(momentum=0.0, nesterov=False)
    lr_spec = tmod.LinearDecay(init_value=0.1, end_value=0.0, transition_steps=10)

    gt = bmod.build_optimizer(cfg, lr_spec)
    assert gt.name == "sgd"

    assert called["n"] == 1

    name, kw = rec.calls[-1]
    assert name == "sgd"
    assert kw["learning_rate"] is sentinel_schedule
