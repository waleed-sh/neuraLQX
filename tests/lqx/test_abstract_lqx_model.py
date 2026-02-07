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


def test_abstract_lqx_model_properties(toy_objects_gauge_invariant):
    H, GG, model_constraint, iface = toy_objects_gauge_invariant
    model = iface.model

    assert model.hilbert is H
    assert model.gauge_group is GG
    assert model.graph is H.graph

    assert model.is_computational is True
    assert model.spacetime_dimensions == 4
    assert model.model_name == "ToyModel"
    assert model.gauge_dimensions == H.gauge_dimensions
    assert model.hilbert_dimensions == H.dimensions

    assert model.constraint is model_constraint


def test_abstract_lqx_model_repr_contains_key_fields(toy_objects_gauge_invariant):
    *_rest, iface = toy_objects_gauge_invariant
    r = repr(iface.model)
    assert "model_name=" in r
    assert "spacetime_dimensions=" in r
    assert "hilbert_dimensions=" in r
    assert "gauge_dimensions=" in r
