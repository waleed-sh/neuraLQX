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


import copy
import pytest


def test_preprocess_constraint_parses_signs_and_global_minus():
    from neuralqx.hilbert.constraints.utils._common import preprocess_constraint

    ce = [
        [["0", "-2"], ["N", "3", "-4"]],
    ]
    out = preprocess_constraint(copy.deepcopy(ce))

    assert len(out) == 1
    inc, outg = out[0]

    assert inc == [(0, +1), (2, -1)]
    assert outg == [(3, -1), (4, +1)]


def test_preprocess_constraints_for_jax_matches_preprocess_constraint():
    from neuralqx.hilbert.constraints.utils._common import (
        preprocess_constraint,
        preprocess_constraints_for_jax,
    )

    ce = [
        [["1"], ["2"]],
        [["0", "-3"], ["N", "-4", "5"]],
        [["7"], ["N"]],
    ]
    a = preprocess_constraint(copy.deepcopy(ce))
    b = preprocess_constraints_for_jax(copy.deepcopy(ce))

    assert a == b


def test_preprocess_does_not_mutate_input_lists():
    from neuralqx.hilbert.constraints.utils._common import (
        preprocess_constraints_for_jax,
    )

    ce = [
        [["1"], ["N", "2", "-3"]],
    ]
    ce_copy = copy.deepcopy(ce)
    _ = preprocess_constraints_for_jax(ce)
    assert ce == ce_copy


def test_preprocess_rejects_non_int_tokens_other_than_N():
    from neuralqx.hilbert.constraints.utils._common import preprocess_constraint

    ce = [
        [["foo"], ["1"]],
    ]
    with pytest.raises(ValueError):
        preprocess_constraint(ce)
