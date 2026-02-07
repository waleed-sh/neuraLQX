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


def test_optimizer_class_unknown_optimizer_errors(optimizer_modules):
    *_mods, omod = optimizer_modules

    with pytest.raises(ValueError) as e:
        omod.Optimizer("does-not-exist")

    msg = str(e.value).lower()
    assert "unknown optimizer" in msg
    assert "adam" in msg


def test_optimizer_class_unknown_scheduler_errors(optimizer_modules):
    *_mods, omod = optimizer_modules

    with pytest.raises(ValueError) as e:
        omod.Optimizer("adam", scheduler_type="does-not-exist", scheduler_kwargs={})

    msg = str(e.value).lower()
    assert "unknown scheduler" in msg

    assert "exponential" in msg


def test_optimizer_class_defaults_to_1e3_lr_when_none(optimizer_modules):
    tmod, *_rest, omod = optimizer_modules

    opt = omod.Optimizer("adam")
    assert opt.optimizer_name == "adam"
    assert opt.lr_spec == 1e-3
    assert isinstance(opt.optimizer_cfg, tmod.Adam)


def test_optimizer_class_scheduler_builds_lr_spec_instance(optimizer_modules):
    tmod, *_rest, omod = optimizer_modules

    opt = omod.Optimizer(
        "sgd",
        scheduler_type="linear decay",
        scheduler_kwargs={"init_value": 0.1, "end_value": 0.0, "transition_steps": 10},
        momentum=0.1,
    )
    assert isinstance(opt.optimizer_cfg, tmod.SGD)
    assert isinstance(opt.lr_spec, tmod.LinearDecay)
    assert opt.lr_spec.init_value == 0.1


def test_optimizer_property_builds_and_caches(monkeypatch, optimizer_modules):
    *_mods, omod = optimizer_modules

    called = {"n": 0}

    def fake_build_optimizer(cfg, lr_spec):
        called["n"] += 1
        return ("GT", cfg, lr_spec)

    monkeypatch.setattr(omod, "build_optimizer", fake_build_optimizer, raising=True)

    opt = omod.Optimizer("adam", learning_rate=0.01)
    o1 = opt.optimizer
    o2 = opt.optimizer

    assert called["n"] == 1
    assert o1 == o2
