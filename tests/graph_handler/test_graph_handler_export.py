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


def test_export_planar_uses_primal_dual_draw_and_writes_file(
    GraphHandler, theta_planar_keyed_edges, tmp_path, monkeypatch
):
    import matplotlib.pyplot as plt

    called = {"primal": 0}

    def fake_draw_graph_and_dual(handler, plot=False, return_fig=False):
        called["primal"] += 1
        if return_fig:
            return plt.figure()
        return None

    mod = __import__(
        GraphHandler.__module__, fromlist=["_draw_graph_and_dual", "compute_signs"]
    )
    monkeypatch.setattr(mod, "_draw_graph_and_dual", fake_draw_graph_and_dual)

    monkeypatch.setattr(mod, "compute_signs", lambda *args, **kwargs: None)

    h = GraphHandler(theta_planar_keyed_edges, plot=False)

    out = tmp_path / "g.png"
    h.export(str(out), dpi=50)

    assert called["primal"] >= 1
    assert out.exists()
    assert out.stat().st_size > 0


def test_export_nonplanar_uses_3d_draw_and_writes_file(
    GraphHandler, theta_nonplanar_keyed_edges, tmp_path, monkeypatch
):
    import matplotlib.pyplot as plt

    called = {"nonplanar": 0}

    def fake_draw_nonplanar_graph(handler, plot=False, return_fig=False):
        called["nonplanar"] += 1
        if return_fig:
            return plt.figure()
        return None

    mod = __import__(
        GraphHandler.__module__, fromlist=["draw_nonplanar_graph", "compute_signs"]
    )
    monkeypatch.setattr(mod, "draw_nonplanar_graph", fake_draw_nonplanar_graph)

    monkeypatch.setattr(mod, "compute_signs", lambda *args, **kwargs: None)

    h = GraphHandler(theta_nonplanar_keyed_edges, plot=False)

    out = tmp_path / "g3d.png"
    h.export(str(out), dpi=50)

    assert called["nonplanar"] >= 1
    assert out.exists()
    assert out.stat().st_size > 0
