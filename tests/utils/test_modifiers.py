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
def mmod():
    return importlib.import_module("neuralqx.utils.misc.modifiers")


@pytest.mark.parametrize(
    "mod,max_terms,expected",
    [
        (0, 9, "0"),
        (3, 9, "3"),
        (3, 10, "03"),
        (9, 10, "09"),
        (10, 10, "10"),
        (3, 999, "003"),
        (42, 999, "042"),
        (123, 999, "123"),
        (7, 1000, "0007"),
    ],
)
def test_get_modifier_value_padding_width(mmod, mod, max_terms, expected):
    assert mmod.get_modifier_value(mod, max_terms) == expected


def test_get_modifier_value_width_is_len_of_max_terms_string(mmod):
    assert mmod.get_modifier_value(1, 100) == "001"
    assert mmod.get_modifier_value(99, 100) == "099"


def test_get_modifier_value_negative_numbers_current_behavior(mmod):

    assert mmod.get_modifier_value(-1, 10) == "-1"

    assert mmod.get_modifier_value(-5, 999) == "-05"

    assert mmod.get_modifier_value(-12, 999) == "-12"


def test_get_modifier_string_concatenates_and_pads_each_component(mmod):
    out = mmod.get_modifier_string(1, 2, 10, max_terms=10)
    assert out == "010210"


def test_get_modifier_string_single_argument(mmod):
    assert mmod.get_modifier_string(7, max_terms=100) == "007"


def test_get_modifier_string_no_modifiers_returns_empty_string(mmod):
    assert mmod.get_modifier_string(max_terms=100) == ""


def test_get_modifier_string_respects_large_width(mmod):
    out = mmod.get_modifier_string(1, 23, 456, max_terms=10000)
    assert out == "000010002300456"


def test_get_modifier_string_raises_if_max_terms_missing(mmod):
    with pytest.raises(TypeError):
        mmod.get_modifier_string(1, 2, 3)
