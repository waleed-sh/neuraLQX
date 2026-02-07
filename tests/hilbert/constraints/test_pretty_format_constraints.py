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


def test_map_edge_string_replaces_coordinate_tuples():
    from neuralqx.hilbert.constraints.utils._common import map_edge_string

    coords_to_int = {
        (1.0, 2.0, 3.0): 7,
        (4.0, 5.0, 6.0): 9,
    }
    s = "((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), 0)"
    out = map_edge_string(s, coords_to_int)
    assert out == "(7, 9, 0)"


def test_pretty_format_constraints_empty_and_malformed():
    from neuralqx.hilbert.constraints.utils._common import pretty_format_constraints

    assert pretty_format_constraints([]) == "No constraints found."

    out = pretty_format_constraints([["bad"]])
    assert "Malformed constraint" in out


def test_pretty_format_constraints_formats_and_maps_vertices():
    from neuralqx.hilbert.constraints.utils._common import pretty_format_constraints

    constraints = [
        [
            ["((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), 0)"],
            ["-((1.0, 2.0, 3.0), (7.0, 8.0, 9.0), 0)"],
        ],
    ]
    coordinates_map = {
        0: (1.0, 2.0, 3.0),
        1: (4.0, 5.0, 6.0),
        2: (7.0, 8.0, 9.0),
    }

    out = pretty_format_constraints(constraints, coordinates_map=coordinates_map)
    assert "Edge (0, 1, 0) is fixed by" in out
    assert "- (0, 2, 0)" in out
