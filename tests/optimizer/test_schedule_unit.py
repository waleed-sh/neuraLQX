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


def test_build_lr_float_is_identity(optimizer_modules):
    _tmod, smod, *_ = optimizer_modules
    assert smod.build_lr(0.123) == 0.123


def test_build_lr_exponential_decay_calls_optax(patch_optax, optimizer_modules):
    tmod, smod, *_ = optimizer_modules
    _optax, rec = patch_optax

    spec = tmod.ExponentialDecay(
        init_value=1.0,
        transition_steps=5,
        decay_rate=0.9,
        staircase=True,
        end_value=None,
    )
    sched = smod.build_lr(spec)

    name, kwargs = rec.calls[-1]
    assert name == "exponential_decay"
    assert kwargs["init_value"] == 1.0
    assert kwargs["transition_steps"] == 5
    assert kwargs["decay_rate"] == 0.9
    assert kwargs["staircase"] is True
    assert kwargs["end_value"] is None

    assert callable(sched)
    assert sched(3)[0] == "exponential_decay"


def test_build_lr_cosine_decay_calls_optax(patch_optax, optimizer_modules):
    tmod, smod, *_ = optimizer_modules
    _optax, rec = patch_optax

    spec = tmod.CosineDecay(init_value=1.0, decay_steps=10, alpha=0.25)
    sched = smod.build_lr(spec)

    name, kwargs = rec.calls[-1]
    assert name == "cosine_decay_schedule"
    assert kwargs["init_value"] == 1.0
    assert kwargs["decay_steps"] == 10
    assert kwargs["alpha"] == 0.25

    assert callable(sched)
    assert sched(7)[0] == "cosine_decay_schedule"


def test_build_lr_linear_decay_calls_optax(patch_optax, optimizer_modules):
    tmod, smod, *_ = optimizer_modules
    _optax, rec = patch_optax

    spec = tmod.LinearDecay(
        init_value=1.0, end_value=0.0, transition_steps=20, transition_begin=3
    )
    sched = smod.build_lr(spec)

    name, kwargs = rec.calls[-1]
    assert name == "linear_schedule"
    assert kwargs["init_value"] == 1.0
    assert kwargs["end_value"] == 0.0
    assert kwargs["transition_steps"] == 20
    assert kwargs["transition_begin"] == 3

    assert callable(sched)
    assert sched(0)[0] == "linear_schedule"


def test_make_learning_rate_dispatches_to_build_lr(patch_optax, optimizer_modules):
    tmod, smod, *_ = optimizer_modules
    _optax, rec = patch_optax

    assert smod.make_learning_rate(0.5) == 0.5

    spec = tmod.CosineDecay(init_value=1.0, decay_steps=10)
    lr = smod.make_learning_rate(spec)
    assert callable(lr)
    assert rec.calls[-1][0] == "cosine_decay_schedule"
