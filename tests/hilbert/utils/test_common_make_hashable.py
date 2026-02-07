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


import numpy as np
import pytest


def test_make_hashable_nested_list_tuple_dict():
    jax = pytest.importorskip("jax")
    _ = pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils._common import make_hashable

    obj = {
        "a": [1, 2, (3, 4)],
        "b": {"x": [5, 6], "y": (7,)},
    }
    h = make_hashable(obj)

    assert isinstance(h, tuple)
    assert hash(h) == hash(make_hashable(obj))


def test_make_hashable_dict_order_independent():
    jax = pytest.importorskip("jax")
    _ = pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils._common import make_hashable

    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    assert make_hashable(d1) == make_hashable(d2)
    assert hash(make_hashable(d1)) == hash(make_hashable(d2))


def test_make_hashable_numpy_array_roundtrip():
    jax = pytest.importorskip("jax")
    _ = pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils._common import make_hashable

    arr = np.array([[1, 2], [3, 4]])
    h = make_hashable(arr)
    assert h == ((1, 2), (3, 4))
    assert isinstance(h, tuple)
    hash(h)


def test_make_hashable_does_not_mutate_input():
    jax = pytest.importorskip("jax")
    _ = pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils._common import make_hashable

    lst = [1, [2, 3]]
    _ = make_hashable(lst)
    assert lst == [1, [2, 3]]


def test_make_hashable_leaves_other_objects_unchanged():
    jax = pytest.importorskip("jax")
    _ = pytest.importorskip("jax.numpy")

    from neuralqx.hilbert.utils._common import make_hashable

    o = object()
    assert make_hashable(o) is o
