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

import dataclasses
import pytest


def test_sampler_registry_contains_expected_aliases(sampler_modules):
    tmod, _, _, _ = sampler_modules
    reg = tmod.SAMPLER_REGISTRY

    assert "metropolis local" in reg
    assert "exact sampler" in reg
    assert "pt metropolis local" in reg
    assert "parallel tempering local" in reg

    assert reg["pt metropolis local"] is reg["parallel tempering local"]
    assert reg["pt metropolis exchange"] is reg["parallel tempering exchange"]
    assert reg["pt weighted sampler"] is reg["parallel tempering weighted sampler"]
    assert reg["pt hamiltonian list"] is reg["parallel tempering hamiltonian list"]


@pytest.mark.parametrize(
    "cls_name, kwargs",
    [
        (
            "MetropolisLocal",
            dict(n_chains_per_rank=2, sweep_size=3, machine_pow=2, reset_chains=True),
        ),
        (
            "MuLocal",
            dict(n_chains_per_rank=2, sweep_size=3, machine_pow=2, reset_chains=True),
        ),
        (
            "MetropolisKLocal",
            dict(
                n_chains_per_rank=1,
                sweep_size=1,
                machine_pow=1,
                reset_chains=False,
                n_flips=4,
            ),
        ),
        (
            "MetropolisExchange",
            dict(
                n_chains_per_rank=5,
                sweep_size=2,
                machine_pow=2,
                reset_chains=True,
                d_max=7,
            ),
        ),
        (
            "Weighted",
            dict(
                n_chains_per_rank=2,
                sweep_size=2,
                machine_pow=2,
                reset_chains=True,
                probabilities=[0.5, 0.5],
                rules=[object(), object()],
            ),
        ),
        (
            "PTLocal",
            dict(
                n_chains_per_rank=2,
                sweep_size=2,
                machine_pow=2,
                reset_chains=True,
                n_replicas=3,
                betas=[1.0, 0.5, 0.2],
            ),
        ),
        (
            "PTExchange",
            dict(
                n_chains_per_rank=2,
                sweep_size=2,
                machine_pow=2,
                reset_chains=True,
                n_replicas=2,
                betas="linear",
                d_max=1,
            ),
        ),
        ("Exact", dict()),
    ],
)
def test_config_dataclasses_are_frozen_and_detectable(
    sampler_modules, cls_name, kwargs
):
    tmod, _, _, _ = sampler_modules
    cls = getattr(tmod, cls_name)

    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen is True

    obj = cls(**kwargs)

    fields = dataclasses.fields(obj)
    if fields:
        field_name = fields[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field_name, getattr(obj, field_name))
    else:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, "anything", 1)
