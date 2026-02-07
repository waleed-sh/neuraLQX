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
import numpy as np

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")

import importlib


def _import_optimizer_module(mod_name: str):
    try:
        return importlib.import_module(f"neuralqx.optimizer.{mod_name}")
    except ModuleNotFoundError:
        return importlib.import_module(f"neuralqx.optimizers.{mod_name}")


tmod = _import_optimizer_module("types")
bmod = _import_optimizer_module("builds")
smod = _import_optimizer_module("schedule")
omod = _import_optimizer_module("optimizer")


def _one_step(gt, params, grads, state):
    updates, state = gt.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, state, updates


@pytest.mark.parametrize(
    "cfg",
    [
        tmod.Adam(),
        tmod.SGD(momentum=0.0, nesterov=False),
        tmod.Adagrad(),
        tmod.RMSProp(centered=False),
        tmod.AdaBelief(),
        tmod.AMSGrad(),
        tmod.Yogi(),
    ],
)
def test_build_optimizer_real_optax_moves_params_in_expected_direction(cfg):

    lr = 1e-2
    gt = bmod.build_optimizer(cfg, lr)

    params = {"w": jnp.array(1.0)}
    grads = {"w": jnp.array(1.0)}

    state = gt.init(params)
    new_params, state, updates = _one_step(gt, params, grads, state)

    assert float(new_params["w"]) < float(
        params["w"]
    ), f"{type(cfg).__name__} did not decrease parameter under positive gradient."
    assert np.isfinite(float(new_params["w"]))


def test_make_learning_rate_real_optax_values_are_sensible():

    exp = tmod.ExponentialDecay(
        init_value=1.0, transition_steps=1, decay_rate=0.5, staircase=True
    )
    sched = smod.make_learning_rate(exp)
    f = jax.jit(lambda s: sched(s))

    v0 = float(f(jnp.array(0, dtype=jnp.int32)))
    v5 = float(f(jnp.array(5, dtype=jnp.int32)))

    assert np.isclose(v0, 1.0)

    assert np.isclose(v5, 1.0 * (0.5**5), rtol=1e-6, atol=1e-6)

    cos = tmod.CosineDecay(init_value=2.0, decay_steps=10, alpha=0.25)
    cs = smod.make_learning_rate(cos)
    vstart = float(cs(0))
    vend = float(cs(10))
    assert np.isclose(vstart, 2.0, rtol=1e-6, atol=1e-6)
    assert np.isclose(vend, 2.0 * 0.25, rtol=1e-6, atol=1e-6)

    lin = tmod.LinearDecay(
        init_value=1.0, end_value=0.0, transition_steps=10, transition_begin=0
    )
    ls = smod.make_learning_rate(lin)
    assert np.isclose(float(ls(0)), 1.0, rtol=1e-6, atol=1e-6)
    assert np.isclose(float(ls(10)), 0.0, rtol=1e-6, atol=1e-6)


def test_scheduled_lr_reduces_update_magnitude_over_time_for_sgd():

    cfg = tmod.SGD(momentum=0.0, nesterov=False)
    lr_spec = tmod.ExponentialDecay(
        init_value=0.1, transition_steps=1, decay_rate=0.5, staircase=True
    )

    gt = bmod.build_optimizer(cfg, lr_spec)

    params = {"w": jnp.array(1.0)}
    grads = {"w": jnp.array(1.0)}
    state = gt.init(params)

    params1, state, updates0 = _one_step(gt, params, grads, state)
    u0 = float(jnp.abs(updates0["w"]))

    p = params1
    for _ in range(4):
        p, state, _ = _one_step(gt, p, grads, state)

    p2, state, updates5 = _one_step(gt, p, grads, state)
    u5 = float(jnp.abs(updates5["w"]))

    assert (
        u5 < u0
    ), f"Expected scheduled LR to reduce |update| over time (u0={u0}, u5={u5})."


def test_optimizer_class_builds_real_optax_and_exports():

    opt = omod.Optimizer(
        "adam",
        scheduler_type="cosine_decay",
        scheduler_kwargs={"init_value": 0.01, "decay_steps": 5, "alpha": 0.2},
        b1=0.9,
        b2=0.999,
        eps=1e-8,
    )

    gt = opt.build()
    assert hasattr(gt, "init") and hasattr(gt, "update")

    fields, values = opt.export_info()
    assert "Optimizer" in fields
    assert "Scheduler type" in fields
    assert values[fields.index("Optimizer")] == "Adam"
    assert values[fields.index("Scheduler type")] == "CosineDecay"
