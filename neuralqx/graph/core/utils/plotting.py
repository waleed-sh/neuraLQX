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

r"""
Graph drawing helpers for GraphHandler.

This module provides Matplotlib based visualisation routines for both planar and non planar
graphs managed by GraphHandler. The main use cases are

- quick inspection of the primal directed multigraph, including parallel edges
- optional overlay of the dual graph, typically the line graph where primal edges become dual vertices
- 3D rendering of non planar embeddings, where vertices carry explicit Cartesian coordinates

Planar rendering uses NetworkX layout positions, and represents directed edges as curved
arrows. Parallel edges between the same ordered pair of vertices are fanned out by
assigning a distinct curvature parameter to each edge.

Non planar rendering assumes an explicit embedding :math:`p_v \in \mathbb{R}^3` for each
vertex :math:`v`. To improve depth perception, vertices and edges are coloured by their
projected depth along the current viewing direction. If :math:`\hat d` is the unit vector
pointing towards the camera, then the depth of a point :math:`p` is computed as

.. math::

   z_{\mathrm{view}}(p) = p \cdot \hat d

and the resulting scalar values are mapped to colours via a Matplotlib colour map.

These functions are intended for diagnostics and figure generation, and they may update
fields on the handler, for example cached layout positions and stored figure handles.
"""

from typing import Optional

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def reorder_edges(edges: list[tuple], nx_graph_edges: list[tuple]) -> list[tuple]:
    """
    Reorder a list of keyed edges to match the order used by a NetworkX multigraph.

    This is a convenience helper for keeping a separate edge list aligned with
    ``nx_graph.edges(keys=True)``. Both inputs are expected to contain the same set of
    keyed edge tuples ``(u, v, key)``.

    :param edges: Edge tuples to be reordered.
    :param nx_graph_edges: Reference ordering, typically obtained from NetworkX.
    :return: `edges` sorted to match the order of `nx_graph_edges`.
    :raises KeyError: If an edge in `edges` does not appear in `nx_graph_edges`.
    """

    order_map = {edge: idx for idx, edge in enumerate(nx_graph_edges)}
    return sorted(edges, key=lambda e: order_map[e])


def draw_graph(
    handler: "GraphHandler",
    max_rad: float = 0.3,
    arrow_length: float = 0.1,
    mutation_scale: float = 20,
) -> None:
    r"""
    Draw the primal directed multigraph with curved arrows for parallel edges.

    This routine visualises the handler's planar NetworkX multigraph using 2D positions stored
    in ``handler.positions``. Nodes are drawn with labels, and each directed edge is drawn as
    a Matplotlib :class:`matplotlib.patches.FancyArrowPatch`.

    Parallel edges between the same ordered pair ``(u, v)`` are fanned out by assigning each
    key a different arc curvature. If there are :math:`n` parallel edges, the curvature values
    are spread approximately symmetrically in the interval :math:`[-r_{\max}, r_{\max}]`,
    where :math:`r_{\max}` is controlled by `max_rad`.

    The midpoint and a perpendicular offset are computed for each arc in order to place edge
    labels cleanly. Edge label rendering is currently disabled in this helper, but the label
    positions are computed and can be enabled by restoring the commented text block.

    :param handler: GraphHandler instance providing `nx_graph`, `edges`, and `positions`.
    :param max_rad: Maximum absolute curvature parameter used when fanning out parallel edges.
    :param arrow_length: Reserved for future control of arrow sizing, currently not used.
    :param mutation_scale: Scale factor controlling the rendered arrow head size.
    :return: None.
    """

    G = handler.nx_graph
    edges = handler.edges
    pos = handler.positions

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#ed6a5a", node_size=2500)
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_color="#404040", font_size=24, font_weight="bold"
    )

    # group parallel edges
    parallels = {}
    for u, v, key in edges:
        parallels.setdefault((u, v), []).append(key)

    # draw each curved arrow and label
    for (u, v), keys in parallels.items():
        n = len(keys)
        # curvature values
        rads = (
            [(((i / (n - 1)) * 2 - 1) * max_rad) for i in range(n)] if n > 1 else [0.0]
        )
        for key, rad in zip(sorted(keys), rads):
            # draw curved arrow
            patch = FancyArrowPatch(
                posA=pos[u],
                posB=pos[v],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                color="#ed6a5a",
                linewidth=6,
                mutation_scale=mutation_scale,
                shrinkA=24,
                shrinkB=24,
            )
            ax.add_patch(patch)

            # compute label position: midpoint + perpendicular offset
            mid = (pos[u] + pos[v]) / 2
            direction = pos[v] - pos[u]
            unit = direction / np.linalg.norm(direction)
            perp = np.array([-unit[1], unit[0]]) * (
                rad * np.linalg.norm(direction) * 0.2
            )

            label_pos = mid + perp

            # place edge label (u,v,key)
            # edge_label = f"{(u, v, key)}"
            # ax.text(
            #     label_pos[0], label_pos[1], edge_label,
            #     ha='center', va='center',
            #     fontsize=12, color="#ed6a5a",
            #     backgroundcolor='white', zorder=5
            # )

    ax.axis("off")
    plt.show()


def _draw_graph_and_dual(
    handler: "GraphHandler",
    plot: bool = True,
    overlay_dual: bool = False,
    max_rad: float = 0.3,
    arrow_length: float = 0.1,
    mutation_scale: float = 20,
    return_fig: bool = False,
) -> Optional[Figure]:
    r"""
    Draw the primal directed multigraph and optionally overlay the dual graph.

    This routine computes a spring layout for the primal graph and stores it in
    ``handler.positions``. It then computes dual node positions as midpoints of primal edges
    and stores them in ``handler.dual_positions``. The dual graph is expected to be available
    as ``handler.dual_nx_graph``.

    Dual positioning is defined by the midpoint rule. For a primal edge :math:`e=(u,v,k)` with
    layout positions :math:`p_u` and :math:`p_v`, the dual vertex position is

    .. math::

       p^{\ast}_e = \tfrac{1}{2}\,(p_u + p_v)

    When `overlay_dual` is True, dual nodes and labels are drawn on top of the primal plot.
    This is useful to verify the correspondence between primal edges and dual vertices.

    This function may also register graph objects via handler internal methods and store the
    generated figure in ``handler.graph_and_dual_plot``.

    :param handler: GraphHandler instance providing the primal and dual NetworkX graphs.
    :param plot: If True, display the figure with Matplotlib. If False, close it after creation.
    :param overlay_dual: If True, draw the dual nodes and labels on the same axes.
    :param max_rad: Maximum absolute curvature parameter for parallel edges in the primal graph.
    :param arrow_length: Reserved for future control of arrow sizing, currently not used.
    :param mutation_scale: Scale factor controlling the rendered arrow head size.
    :param return_fig: If True, return the Matplotlib figure object instead of displaying it.
    :return: The Matplotlib :class:`matplotlib.figure.Figure` if `return_fig` is True, otherwise None.
    """

    g = handler.nx_graph
    edges = handler.edges

    seed = 42 if handler.is_planar else 90
    pos = nx.spring_layout(g, seed=seed)
    handler.positions = pos

    # compute dual positions
    dg_pos = {(u, v, key): (pos[u] + pos[v]) / 2 for u, v, key in g.edges(keys=True)}
    handler.dual_positions = dg_pos
    handler._register_graph_data(g, "graph")
    handler._register_graph_data(handler.dual_nx_graph, "dualGraph")

    # if not plot:
    #     return

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#ed6a5a", node_size=2500)
    nx.draw_networkx_labels(
        g, pos, ax=ax, font_color="#404040", font_size=24, font_weight="bold"
    )

    # fan out and draw each curved arrow with label
    parallels = {}
    for u, v, key in edges:
        parallels.setdefault((u, v), []).append(key)

    for (u, v), keys in parallels.items():
        n = len(keys)
        rads = (
            [(((i / (n - 1)) * 2 - 1) * max_rad) for i in range(n)] if n > 1 else [0.0]
        )
        for key, rad in zip(sorted(keys), rads):
            patch = FancyArrowPatch(
                posA=pos[u],
                posB=pos[v],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                color="#ed6a5a",
                linewidth=6,
                mutation_scale=mutation_scale,
                shrinkA=24,
                shrinkB=24,
            )
            ax.add_patch(patch)

            mid = (pos[u] + pos[v]) / 2
            direction = pos[v] - pos[u]
            unit = direction / np.linalg.norm(direction)
            perp = np.array([-unit[1], unit[0]]) * (
                rad * np.linalg.norm(direction) * 0.2
            )
            # label_pos = mid + perp

            # edge_label = f"{(u, v, key)}"
            # ax.text(
            #     label_pos[0], label_pos[1], edge_label,
            #     ha='center', va='center',
            #     fontsize=12, color="#ed6a5a",
            #     backgroundcolor='white', zorder=5
            # )

    # optionally overlay dual nodes and labels
    if overlay_dual:
        lg = handler.dual_nx_graph
        nx.draw_networkx_nodes(lg, dg_pos, ax=ax, node_color="#7ebdc2", node_size=2500)
        labels = {node: str(node) for node in lg.nodes()}
        nx.draw_networkx_labels(
            lg,
            dg_pos,
            labels=labels,
            ax=ax,
            font_size=16,
            font_color="whitesmoke",
            font_weight="bold",
        )

    ax.axis("off")
    handler.graph_and_dual_plot = fig

    if return_fig:
        return fig

    if plot:
        plt.show()
    else:
        plt.close(fig)


def draw_nonplanar_graph(
    handler: "GraphHandler",
    node_size: float = 35000,
    edge_width: float = 5.0,
    show_labels: bool = True,
    alpha_nodes: float = 0.9,
    alpha_edges: float = 0.85,
    figsize: tuple = (12, 9),
    elev: float = 25,
    azim: float = 45,
    cmap_name: str = "viridis",
    mesh_alpha: float = 0.08,
    mesh_density: int = 10,
    return_fig: bool = False,
    plot: bool = True,
) -> Optional[Figure]:
    r"""
    Draw a non planar embedded graph in 3D with depth based colouring.

    This routine assumes the handler stores a mapping from integer vertex labels to original
    3D coordinates in ``handler._nonplanar_int2orig_map``. The edge list ``handler.edges`` is
    assumed to reference the integer labels, typically in the keyed form ``(u, v, k)``.

    Depth based colouring is computed from the current view direction defined by the elevation
    and azimuth angles. A unit view vector :math:`\hat d` is constructed, and each vertex at
    position :math:`p_v` is assigned a scalar depth

    .. math::

       z_{\mathrm{view}}(v) = p_v \cdot \hat d

    Vertices are coloured by this depth, and edges are coloured by the mean depth of their
    endpoints. A faint 3D mesh is drawn in the background to provide spatial reference.

    If `show_labels` is True, each vertex is annotated with its integer index. The text colour
    is chosen adaptively using a luminance heuristic so that labels remain readable against the
    node face colour.

    :param handler: GraphHandler instance containing a non planar coordinate map and edge list.
    :param node_size: Marker size control for nodes, scaled internally for Matplotlib 3D scatter.
    :param edge_width: Line width for edges in the 3D line collection.
    :param show_labels: If True, draw integer vertex labels at each vertex location.
    :param alpha_nodes: Opacity applied to node markers.
    :param alpha_edges: Opacity applied to edge segments.
    :param figsize: Figure size passed to Matplotlib.
    :param elev: Elevation angle in degrees for the 3D view.
    :param azim: Azimuth angle in degrees for the 3D view.
    :param cmap_name: Name of the Matplotlib colour map used for depth colouring.
    :param mesh_alpha: Opacity of the background mesh surfaces and wireframes.
    :param mesh_density: Grid resolution used to draw the background mesh.
    :param return_fig: If True, return the figure object and do not display or close it.
    :param plot: If True, display the plot. If False, close the figure unless `return_fig` is True.
    :return: The Matplotlib :class:`matplotlib.figure.Figure` if `return_fig` is True, otherwise None.
    :raises ValueError: If the handler does not contain non planar coordinates.
    """

    # coordinate map
    coords_map = getattr(handler, "_nonplanar_int2orig_map", None)

    # sanity check to enforce only non-planar graphs
    if coords_map is None:
        raise ValueError("No non-planar coordinates found in handler.")

    edges = handler.edges

    vertices = np.array([coords_map[v] for v in sorted(coords_map)])
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # compute depth from current view direction
    elev_r, azim_r = np.deg2rad(elev), np.deg2rad(azim)
    view_dir = np.array(
        [
            np.cos(elev_r) * np.sin(azim_r),
            -np.cos(elev_r) * np.cos(azim_r),
            np.sin(elev_r),
        ]
    )
    view_dir /= np.linalg.norm(view_dir)

    depths = vertices @ view_dir
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=depths.min(), vmax=depths.max())
    vertex_colors = cmap(norm(depths))

    segs, seg_colors = [], []
    for u, v, k in edges:
        p1, p2 = np.array(coords_map[u]), np.array(coords_map[v])
        segs.append([p1, p2])
        mean_depth = ((p1 @ view_dir) + (p2 @ view_dir)) / 2
        seg_colors.append(cmap(norm(mean_depth)))

    # figure setup
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # faint 3D mesh background for spatial reference
    lim = np.array([x.min(), x.max(), y.min(), y.max(), z.min(), z.max()])
    X, Y = np.meshgrid(
        np.linspace(lim[0], lim[1], mesh_density),
        np.linspace(lim[2], lim[3], mesh_density),
    )
    Z0 = np.full_like(X, lim[4])
    Z1 = np.full_like(X, lim[5])

    # bottom and top meshes
    ax.plot_surface(X, Y, Z0, color="gray", alpha=mesh_alpha, linewidth=0, zorder=0)
    ax.plot_surface(X, Y, Z1, color="gray", alpha=mesh_alpha, linewidth=0, zorder=0)

    # vertical mesh slices
    for zslice in np.linspace(lim[4], lim[5], 4):
        ax.plot_wireframe(
            X,
            Y,
            np.full_like(X, zslice),
            color="gray",
            alpha=mesh_alpha * 1.3,
            linewidth=0.3,
            rstride=1,
            cstride=1,
        )

    # draw edges
    lc = Line3DCollection(
        segs,
        linewidths=edge_width,
        colors=seg_colors,
        alpha=alpha_edges,
    )
    ax.add_collection3d(lc)

    # draw nodes
    ax.scatter(
        x,
        y,
        z,
        s=node_size / 50,
        c=vertex_colors,
        edgecolors="black",
        linewidths=0.8,
        alpha=alpha_nodes,
        depthshade=True,
    )

    # adaptive vertex labels
    if show_labels:
        for i, (vx, vy, vz, col) in enumerate(zip(x, y, z, vertex_colors)):
            # compute perceived brightness (YIQ/luminance heuristic)
            r, g, b = col[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.45 else "black"
            ax.text(
                vx,
                vy,
                vz,
                str(i),
                color=text_color,
                fontsize=12.5,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=500,
                path_effects=[],
            )

    # style
    ax.view_init(elev=elev, azim=azim)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_color((0.6, 0.6, 0.6, 0.1))

    ax.set_facecolor("white")

    ax.grid(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    for spine in getattr(ax, "spines", {}).values():
        spine.set_visible(False)

    # colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.03, shrink=0.8)
    cbar.set_label(
        "Projected Depth (along viewing direction)", rotation=270, labelpad=18
    )

    plt.tight_layout()

    if return_fig:
        return fig

    if plot:
        plt.show()
    else:
        plt.close(fig)
