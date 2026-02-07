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


def test_export_info_label_depends_on_mpi_available(monkeypatch, sampler_modules):
    _, _, _, emod = sampler_modules

    monkeypatch.setattr(emod._mpi, "available", False, raising=True)
    f, v = emod.export_info(
        sampler_type="Metropolis Local",
        number_of_samples=10,
        number_of_chains=7,
        number_of_sweeps=3,
        machine_pow=2,
        reset_chains=True,
    )
    assert "Number of chains" in f
    assert "Number of chains per MPI rank" not in f
    assert v[f.index("Number of chains")] == "7"

    monkeypatch.setattr(emod._mpi, "available", True, raising=True)
    f2, v2 = emod.export_info(
        sampler_type="Metropolis Local",
        number_of_samples=10,
        number_of_chains=7,
        number_of_sweeps=3,
        machine_pow=2,
        reset_chains=True,
    )
    assert "Number of chains per MPI rank" in f2
    assert v2[f2.index("Number of chains per MPI rank")] == "7"


def test_export_info_exact_sampler_special_case_is_string_exact(
    monkeypatch, sampler_modules
):

    _, _, _, emod = sampler_modules

    monkeypatch.setattr(emod._mpi, "available", True, raising=True)

    f, v = emod.export_info(
        sampler_type="Exact Sampler",
        number_of_samples=123,
        number_of_chains=7,
        number_of_sweeps=3,
        machine_pow=2,
        reset_chains=True,
    )
    assert f[:2] == ["Sampler type", "Number of samples"]
    assert v[:2] == ["Exact Sampler", "123"]
    assert "Number of chains" in f
    assert v[f.index("Number of chains")] is None

    f2, v2 = emod.export_info(
        sampler_type="exact sampler",
        number_of_samples=123,
        number_of_chains=7,
        number_of_sweeps=3,
        machine_pow=2,
        reset_chains=True,
    )

    assert "Number of chains per MPI rank" in f2
    assert v2[f2.index("Number of chains per MPI rank")] == "7"
