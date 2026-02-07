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


import importlib
import pytest


@pytest.fixture
def cmod():
    return importlib.import_module("neuralqx._cite")


def _pos(haystack: str, needle: str) -> int:
    p = haystack.find(needle)
    assert p != -1, f"Expected to find substring {needle!r}"
    return p


def test_cite_default_is_neuralqx_only(cmod):
    out = str(cmod.cite)

    assert "@software{neuralqx2026github" in out

    assert "@article{netket2:2019" not in out
    assert "@article{netket3:2022" not in out
    assert "@article{mpi4jax:2021" not in out
    assert "@misc{jax2018github" not in out
    assert "@misc{flax2020github" not in out


def test_str_and_repr_match(cmod):
    assert str(cmod.cite) == repr(cmod.cite)


def test_attribute_chaining_adds_groups(cmod):
    out = str(cmod.cite.netket.mpi.jax.flax)

    assert "@software{neuralqx2026github" in out

    assert "@article{netket2:2019" in out
    assert "@article{netket3:2022" in out

    assert "@article{mpi4jax:2021" in out
    assert "@misc{jax2018github" in out
    assert "@misc{flax2020github" in out


def test_chaining_does_not_mutate_original_instance(cmod):
    base = cmod.cite
    chained = base.netket

    base_out = str(base)
    assert "@software{neuralqx2026github" in base_out
    assert "@article{netket2:2019" not in base_out
    assert "@article{netket3:2022" not in base_out

    chained_out = str(chained)
    assert "@article{netket2:2019" in chained_out
    assert "@article{netket3:2022" in chained_out


def test_unknown_group_raises_attribute_error(cmod):
    with pytest.raises(AttributeError, match=r"No such citation group"):
        _ = cmod.cite.this_does_not_exist


def test_group_chaining_is_idempotent(cmod):
    out1 = str(cmod.cite.netket)
    out2 = str(cmod.cite.netket.netket)
    assert out1 == out2

    out3 = str(cmod.cite.mpi)
    out4 = str(cmod.cite.mpi.mpi)
    assert out3 == out4


def test_output_order_is_deterministic_and_stable(cmod):

    out = str(cmod.cite.netket.mpi.jax.flax)

    p_neuralqx = _pos(out, "@software{neuralqx2026github")
    p_netket2 = _pos(out, "@article{netket2:2019")
    p_netket3 = _pos(out, "@article{netket3:2022")
    p_mpi = _pos(out, "@article{mpi4jax:2021")
    p_jax = _pos(out, "@misc{jax2018github")
    p_flax = _pos(out, "@misc{flax2020github")

    assert p_neuralqx < p_netket2 < p_netket3 < p_mpi < p_jax < p_flax


def test_repr_separates_entries_by_blank_lines(cmod):
    out = str(cmod.cite.netket)

    assert "\n\n" in out

    assert "@article{netket2:2019" in out
    assert "@article{netket3:2022" in out
