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


"""Matplotlib plotting backend for graph objects.

This module contains the plotting implementation used by ``Graph.plot``. It
supports deterministic planar layouts, simple three-dimensional layouts,
optional line-graph dual overlays, and presentation-oriented styling for graph
inspection during examples and debugging.
"""

import math
import random

from collections import defaultdict
from typing import Any

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects

from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from neuralqx.utils.typing import Edge
from neuralqx.utils.typing import PlottableGraph
from neuralqx.utils.typing import Vertex


def plot_graph(
    graph: PlottableGraph,
    *,
    show: bool = True,
    ax: Any | None = None,
    include_dual: bool = False,
    with_labels: bool = True,
    title: str | None = None,
    layout: str = "spring",
    seed: int | None = None,
    figsize: tuple[float, float] = (7.5, 6.2),
    dpi: int = 160,
    node_size: float = 980.0,
    edge_width: float = 3.2,
    node_color: str = "#ed6a5a",
    edge_color: str = "#ed6a5a",
    dual_color: str = "#7ebdc2",
    label_color: str = "#404040",
    label_font_size: float = 9.5,
    max_rad: float = 0.24,
    elev: float = 25.0,
    azim: float = 45.0,
    cmap: str = "viridis",
    reference_mesh: bool = True,
    depth_colorbar: bool = True,
) -> tuple[Any, Any]:
    """Plots a compact visualisation of a graph.

    Planar graphs are drawn as directed curved arrows with parallel edges
    fanned out symmetrically. Non-planar graphs are drawn in 3D with a
    depth-shaded colour treatment. By default, plotting uses a deterministic
    force layout for readability: seed ``42`` for planar graphs and seed ``90``
    for non-planar graphs, matching the old neuraLQX plotting convention.

    Args:
        graph: Graph-like object exposing the public graph plotting interface.
        show: If ``True``, call ``matplotlib.pyplot.show`` before returning.
        ax: Optional existing matplotlib axes.
        include_dual: If ``True``, overlay the line-graph dual.
        with_labels: If ``True``, annotate vertices by their index.
        title: Optional axes title.
        layout: ``"spring"`` for deterministic visual layout or
            ``"coordinates"`` to use graph coordinates directly.
        seed: Optional layout seed. Defaults to ``42`` for planar and ``90``
            for non-planar spring layouts.
        figsize: Figure size used when ``ax`` is not supplied.
        dpi: Figure resolution used when ``ax`` is not supplied.
        node_size: Marker size for primal vertices.
        edge_width: Width of primal edges.
        node_color: Color used for primal vertices.
        edge_color: Color used for primal edges.
        dual_color: Color used for dual edges.
        label_color: Color used for labels.
        label_font_size: Font size used for vertex labels.
        max_rad: Maximum curvature used to fan out parallel planar edges.
        elev: 3D elevation angle in degrees.
        azim: 3D azimuth angle in degrees.
        cmap: Matplotlib colormap used for non-planar depth shading.
        reference_mesh: If ``True``, draw a faint 3D mesh for depth cues.
        depth_colorbar: If ``True``, add a projected-depth colorbar for 3D
            plots.

    Returns:
        The matplotlib ``(figure, axes)`` pair.
    """

    is_3d = not graph.is_planar
    if seed is None:
        seed = 90 if is_3d else 42
    positions = _plot_positions(graph, layout=layout, is_3d=is_3d, seed=seed)
    dual_positions = _dual_plot_positions(graph, positions)

    if ax is None:
        if is_3d:
            figure = plt.figure(figsize=figsize, dpi=dpi)
            ax = figure.add_subplot(projection="3d")
        else:
            figure, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        figure = ax.figure

    if is_3d:
        _plot_nonplanar(
            graph,
            ax,
            positions,
            node_size=node_size,
            edge_width=edge_width,
            with_labels=with_labels,
            label_color=label_color,
            label_font_size=label_font_size,
            elev=elev,
            azim=azim,
            cmap_name=cmap,
            reference_mesh=reference_mesh,
            depth_colorbar=depth_colorbar,
        )
    else:
        _plot_planar(
            graph,
            ax,
            positions,
            node_size=node_size,
            edge_width=edge_width,
            node_color=node_color,
            edge_color=edge_color,
            label_color=label_color,
            label_font_size=label_font_size,
            with_labels=with_labels,
            max_rad=max_rad,
        )

    if include_dual:
        _plot_dual_overlay(
            graph,
            ax,
            dual_positions,
            color=dual_color,
            node_size=max(node_size * 0.28, 80.0),
            edge_width=max(edge_width * 0.72, 1.0),
            is_3d=is_3d,
        )

    if title is not None:
        ax.set_title(title, fontsize=11, color="#303030", pad=8)
    if not is_3d:
        ax.set_aspect("equal", adjustable="box")
        _pad_planar_axes(ax, positions)
        ax.set_axis_off()
    else:
        _set_equal_3d_axes(ax, positions)
        ax.view_init(elev=elev, azim=azim)
        _style_3d_axes(ax)
    figure.tight_layout(pad=0.2)

    if show:
        plt.show()
    return figure, ax


def _plot_planar(
    graph: PlottableGraph,
    ax: Any,
    positions: dict[Vertex, tuple[float, ...]],
    *,
    node_size: float,
    edge_width: float,
    node_color: str,
    edge_color: str,
    label_color: str,
    label_font_size: float,
    with_labels: bool,
    max_rad: float,
) -> None:
    """Draws a planar graph on two-dimensional matplotlib axes.

    Args:
        graph: Graph-like object exposing vertices and canonical edges.
        ax: Matplotlib axes receiving the drawing.
        positions: Mapping from graph vertices to two-dimensional positions.
        node_size: Marker size for primal vertices.
        edge_width: Width of primal directed edges.
        node_color: Color used for primal vertices.
        edge_color: Color used for primal edges.
        label_color: Color used for vertex labels.
        label_font_size: Font size used for vertex labels.
        with_labels: Whether to draw vertex index labels.
        max_rad: Maximum curvature for fanning out parallel edges.
    """
    _draw_curved_edges(
        ax,
        graph.edges,
        positions,
        color=edge_color,
        linewidth=edge_width,
        max_rad=max_rad,
        arrowstyle="-|>",
        mutation_scale=14,
        shrink=12,
        alpha=0.95,
    )

    xs = [positions[vertex][0] for vertex in graph.vertices]
    ys = [positions[vertex][1] for vertex in graph.vertices]
    ax.scatter(
        xs,
        ys,
        s=node_size,
        color=node_color,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )
    if with_labels:
        for index, vertex in enumerate(graph.vertices):
            pos = positions[vertex]
            ax.text(
                pos[0],
                pos[1],
                str(index),
                ha="center",
                va="center",
                fontsize=label_font_size,
                fontweight="bold",
                color=label_color,
                zorder=4,
            )


def _draw_curved_edges(
    ax: Any,
    edges: list[Edge],
    positions: dict[Any, tuple[float, ...]],
    *,
    color: str,
    linewidth: float,
    max_rad: float,
    arrowstyle: str,
    mutation_scale: float,
    shrink: float,
    alpha: float,
) -> None:
    """Draws directed planar edges with deterministic curvature.

    Args:
        ax: Matplotlib axes receiving the arrow patches.
        edges: Directed edges to draw.
        positions: Mapping from edge endpoints to two-dimensional positions.
        color: Edge color.
        linewidth: Edge line width.
        max_rad: Maximum absolute curvature for a parallel-edge group.
        arrowstyle: Matplotlib arrow style passed to ``FancyArrowPatch``.
        mutation_scale: Arrowhead scaling factor.
        shrink: Endpoint shrink distance used to keep arrows outside markers.
        alpha: Patch opacity.
    """
    grouped: dict[tuple[Any, Any], list[Edge]] = defaultdict(list)
    for edge in edges:
        grouped[(edge.start, edge.end)].append(edge)

    for edge_group in grouped.values():
        rads = _curvature_values(len(edge_group), max_rad=max_rad)
        for edge, rad in zip(sorted(edge_group, key=lambda item: item.key), rads):
            start = positions[edge.start]
            end = positions[edge.end]
            patch = FancyArrowPatch(
                posA=(start[0], start[1]),
                posB=(end[0], end[1]),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle=arrowstyle,
                color=color,
                linewidth=linewidth,
                mutation_scale=mutation_scale,
                shrinkA=shrink,
                shrinkB=shrink,
                alpha=alpha,
                zorder=2,
            )
            ax.add_patch(patch)


def _curvature_values(count: int, *, max_rad: float) -> list[float]:
    """Returns symmetric curvature values for parallel planar edges.

    Args:
        count: Number of parallel edges in one directed group.
        max_rad: Maximum absolute curvature to assign.

    Returns:
        Curvature values centered around zero.
    """
    if count <= 1:
        return [0.0]
    if count % 2 == 1:
        step = max_rad / (count // 2)
        return [step * (index - count // 2) for index in range(count)]
    step = (2.0 * max_rad) / count
    start = -max_rad + 0.5 * step
    return [start + step * index for index in range(count)]


def _plot_nonplanar(
    graph: PlottableGraph,
    ax: Any,
    positions: dict[Vertex, tuple[float, ...]],
    *,
    node_size: float,
    edge_width: float,
    with_labels: bool,
    label_color: str,
    label_font_size: float,
    elev: float,
    azim: float,
    cmap_name: str,
    reference_mesh: bool,
    depth_colorbar: bool,
) -> None:
    """Draws a non-planar graph on three-dimensional matplotlib axes.

    Args:
        graph: Graph-like object exposing vertices and canonical edges.
        ax: Matplotlib 3D axes receiving the drawing.
        positions: Mapping from graph vertices to three-dimensional positions.
        node_size: Marker size for primal vertices.
        edge_width: Width of primal edges.
        with_labels: Whether to draw vertex index labels.
        label_color: Fallback color used for vertex labels.
        label_font_size: Font size used for vertex labels.
        elev: View elevation angle in degrees.
        azim: View azimuth angle in degrees.
        cmap_name: Matplotlib colormap name used for depth shading.
        reference_mesh: Whether to draw a faint reference grid.
        depth_colorbar: Whether to add a projected-depth colorbar.
    """
    view_dir = _view_direction(elev=elev, azim=azim)
    depths = {vertex: _dot3(positions[vertex], view_dir) for vertex in graph.vertices}
    norm = mcolors.Normalize(
        vmin=min(depths.values(), default=0.0),
        vmax=max(depths.values(), default=1.0),
    )
    colormap = matplotlib.colormaps.get_cmap(cmap_name)

    if reference_mesh:
        _draw_3d_reference_mesh(ax, positions)

    segments = []
    segment_colours = []
    for edge in graph.edges:
        start = positions[edge.start]
        end = positions[edge.end]
        segments.append([start, end])
        depth = 0.5 * (depths[edge.start] + depths[edge.end])
        segment_colours.append(colormap(norm(depth)))

    if segments:
        collection = Line3DCollection(
            segments,
            colors=segment_colours,
            linewidths=edge_width,
            alpha=0.82,
        )
        ax.add_collection3d(collection)

    for index, vertex in enumerate(graph.vertices):
        pos = positions[vertex]
        colour = colormap(norm(depths[vertex]))
        ax.scatter(
            [pos[0]],
            [pos[1]],
            [pos[2]],
            s=node_size * 0.82,
            color=colour,
            edgecolors="#202020",
            linewidths=0.7,
            alpha=0.95,
            depthshade=True,
        )
        if with_labels:
            ax.text(
                pos[0],
                pos[1],
                pos[2],
                str(index),
                fontsize=label_font_size + 0.8,
                fontweight="bold",
                color=_label_color_for_rgba(colour, fallback=label_color),
                ha="center",
                va="center",
                zorder=500,
                path_effects=[
                    patheffects.withStroke(linewidth=1.6, foreground="white")
                ],
            )

    if depth_colorbar:
        scalar = cm.ScalarMappable(cmap=colormap, norm=norm)
        scalar.set_array([])
        colorbar = ax.figure.colorbar(scalar, ax=ax, pad=0.02, shrink=0.74)
        colorbar.set_label(
            "Projected depth",
            rotation=270,
            labelpad=12,
            fontsize=8.5,
        )
        colorbar.ax.tick_params(labelsize=7)


def _plot_dual_overlay(
    graph: PlottableGraph,
    ax: Any,
    positions: dict[Edge, tuple[float, ...]],
    *,
    color: str,
    node_size: float,
    edge_width: float,
    is_3d: bool,
) -> None:
    """Overlays line-graph dual vertices and edges on an existing plot.

    Args:
        graph: Graph-like object exposing dual edges.
        ax: Matplotlib axes receiving the overlay.
        positions: Mapping from primal edges to dual-vertex positions.
        color: Color used for dual vertices and edges.
        node_size: Marker size for dual vertices.
        edge_width: Line width for dual edges.
        is_3d: Whether ``ax`` is a three-dimensional axes object.
    """
    if is_3d:
        for first, second in graph.dual_edges:
            start = positions[first]
            end = positions[second]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                linestyle="--",
                linewidth=edge_width,
                alpha=0.55,
            )
        xs = [positions[edge][0] for edge in graph.edges]
        ys = [positions[edge][1] for edge in graph.edges]
        zs = [positions[edge][2] for edge in graph.edges]
        ax.scatter(xs, ys, zs, s=node_size, color=color, alpha=0.75)
        return

    for first, second in graph.dual_edges:
        start = positions[first]
        end = positions[second]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linestyle="--",
            linewidth=edge_width,
            alpha=0.52,
            zorder=1,
        )
    xs = [positions[edge][0] for edge in graph.edges]
    ys = [positions[edge][1] for edge in graph.edges]
    ax.scatter(
        xs,
        ys,
        s=node_size,
        color=color,
        edgecolors="white",
        linewidths=0.8,
        alpha=0.82,
        zorder=5,
    )


def _pad_planar_axes(
    ax: Any,
    positions: dict[Any, tuple[float, ...]],
) -> None:
    """Pads two-dimensional axes around plotted vertex positions.

    Args:
        ax: Matplotlib axes whose limits should be adjusted.
        positions: Position mapping used for the current planar drawing.
    """
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    if not xs or not ys:
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)
    ax.set_xlim(x_min - 0.12 * width, x_max + 0.12 * width)
    ax.set_ylim(y_min - 0.12 * height, y_max + 0.12 * height)


def _set_equal_3d_axes(
    ax: Any,
    positions: dict[Any, tuple[float, ...]],
) -> None:
    """Sets equal visual scale on three-dimensional axes.

    Args:
        ax: Matplotlib 3D axes whose limits should be adjusted.
        positions: Three-dimensional position mapping for the current graph.
    """
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    zs = [pos[2] for pos in positions.values()]
    if not xs or not ys or not zs:
        return

    x_mid = 0.5 * (min(xs) + max(xs))
    y_mid = 0.5 * (min(ys) + max(ys))
    z_mid = 0.5 * (min(zs) + max(zs))
    radius = 0.55 * max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
        1.0,
    )
    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_zlim(z_mid - radius, z_mid + radius)

    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def _style_3d_axes(ax: Any) -> None:
    """Applies quiet presentation styling to three-dimensional axes.

    Args:
        ax: Matplotlib 3D axes to style.
    """
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.line.set_color((0.55, 0.55, 0.55, 0.18))
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((0.55, 0.55, 0.55, 0.12))
        except AttributeError:
            pass


def _draw_3d_reference_mesh(
    ax: Any,
    positions: dict[Any, tuple[float, ...]],
    *,
    density: int = 7,
) -> None:
    """Draws a faint reference mesh behind a three-dimensional graph.

    Args:
        ax: Matplotlib 3D axes receiving the mesh.
        positions: Three-dimensional position mapping used to bound the mesh.
        density: Number of grid lines along the x and y directions.
    """
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    zs = [pos[2] for pos in positions.values()]
    if not xs or not ys or not zs:
        return

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
    x_values = _linspace(x_min, x_max, density)
    y_values = _linspace(y_min, y_max, density)
    z_values = _linspace(z_min, z_max, 4)
    colour = (0.45, 0.45, 0.45, 0.09)

    for z_value in z_values:
        for x_value in x_values:
            ax.plot(
                [x_value, x_value],
                [y_min, y_max],
                [z_value, z_value],
                color=colour,
                linewidth=0.35,
                zorder=0,
            )
        for y_value in y_values:
            ax.plot(
                [x_min, x_max],
                [y_value, y_value],
                [z_value, z_value],
                color=colour,
                linewidth=0.35,
                zorder=0,
            )


def _linspace(start: float, stop: float, count: int) -> list[float]:
    """Returns evenly spaced values between two endpoints.

    Args:
        start: First endpoint.
        stop: Last endpoint.
        count: Number of values to generate.

    Returns:
        List containing ``count`` values. If ``count`` is one or smaller, the
        midpoint is returned.
    """
    if count <= 1:
        return [0.5 * (start + stop)]
    return [start + (stop - start) * index / (count - 1) for index in range(count)]


def _plot_positions(
    graph: PlottableGraph,
    *,
    layout: str,
    is_3d: bool,
    seed: int,
) -> dict[Vertex, tuple[float, ...]]:
    """Computes vertex positions for a requested plotting layout.

    Args:
        graph: Graph-like object exposing canonical vertices and edges.
        layout: Layout strategy, either ``"coordinates"`` or ``"spring"``.
        is_3d: Whether to produce three-dimensional positions.
        seed: Seed for deterministic spring layout initialization.

    Returns:
        Mapping from graph vertices to plotting coordinates.

    Raises:
        ValueError: If ``layout`` is not supported.
    """
    if layout == "coordinates":
        return {vertex: graph.graph_positions[vertex] for vertex in graph.vertices}
    if layout != "spring":
        raise ValueError("layout must be either 'spring' or 'coordinates'.")
    dimensions = 3 if is_3d else 2
    return _spring_layout(graph, dimensions=dimensions, seed=seed)


def _dual_plot_positions(
    graph: PlottableGraph,
    positions: dict[Vertex, tuple[float, ...]],
) -> dict[Edge, tuple[float, ...]]:
    """Computes dual-vertex positions from primal edge midpoints.

    Args:
        graph: Graph-like object exposing canonical edges.
        positions: Primal vertex positions in two or three dimensions.

    Returns:
        Mapping from each primal edge to the midpoint of its endpoints.
    """
    out = {}
    for edge in graph.edges:
        start = positions[edge.start]
        end = positions[edge.end]
        out[edge] = tuple(
            0.5 * (start[index] + end[index]) for index in range(len(start))
        )
    return out


def _spring_layout(
    graph: PlottableGraph,
    *,
    dimensions: int,
    seed: int,
    iterations: int = 120,
) -> dict[Vertex, tuple[float, ...]]:
    """Computes a deterministic force-directed graph layout.

    Args:
        graph: Graph-like object exposing canonical vertices and edges.
        dimensions: Number of coordinate dimensions to produce.
        seed: Seed for deterministic initial coordinates.
        iterations: Number of force-relaxation iterations.

    Returns:
        Normalized coordinate mapping for graph vertices.
    """
    rng = random.Random(seed)
    vertices = list(graph.vertices)
    if not vertices:
        return {}

    positions = {
        vertex: [rng.uniform(-1.0, 1.0) for _ in range(dimensions)]
        for vertex in vertices
    }
    area = max(float(len(vertices)), 1.0)
    k = math.sqrt(area / len(vertices))
    temperature = 0.35 * math.sqrt(area)

    for iteration in range(iterations):
        displacement = {vertex: [0.0] * dimensions for vertex in vertices}
        for i, first in enumerate(vertices):
            for second in vertices[i + 1 :]:
                delta = [
                    positions[first][axis] - positions[second][axis]
                    for axis in range(dimensions)
                ]
                distance = max(math.sqrt(sum(item * item for item in delta)), 1e-9)
                force = (k * k) / distance
                for axis in range(dimensions):
                    value = delta[axis] / distance * force
                    displacement[first][axis] += value
                    displacement[second][axis] -= value

        for edge in graph.edges:
            first = edge.start
            second = edge.end
            delta = [
                positions[first][axis] - positions[second][axis]
                for axis in range(dimensions)
            ]
            distance = max(math.sqrt(sum(item * item for item in delta)), 1e-9)
            force = (distance * distance) / k
            for axis in range(dimensions):
                value = delta[axis] / distance * force
                displacement[first][axis] -= value
                displacement[second][axis] += value

        cooling = 1.0 - iteration / max(iterations, 1)
        for vertex in vertices:
            length = max(
                math.sqrt(sum(item * item for item in displacement[vertex])),
                1e-9,
            )
            step = min(length, temperature * cooling)
            for axis in range(dimensions):
                positions[vertex][axis] += displacement[vertex][axis] / length * step

    return _normalise_layout(positions)


def _normalise_layout(
    positions: dict[Vertex, list[float]],
) -> dict[Vertex, tuple[float, ...]]:
    """Centers and scales layout coordinates into a unit-radius view.

    Args:
        positions: Mutable layout coordinates produced by the spring solver.

    Returns:
        Immutable centered coordinate tuples scaled by maximum radius.
    """
    dimensions = len(next(iter(positions.values())))
    centers = [
        sum(position[axis] for position in positions.values()) / len(positions)
        for axis in range(dimensions)
    ]
    centered = {
        vertex: [position[axis] - centers[axis] for axis in range(dimensions)]
        for vertex, position in positions.items()
    }
    radius = max(
        math.sqrt(sum(component * component for component in position))
        for position in centered.values()
    )
    scale = 1.0 if radius <= 1e-12 else radius
    return {
        vertex: tuple(component / scale for component in position)
        for vertex, position in centered.items()
    }


def _view_direction(*, elev: float, azim: float) -> tuple[float, float, float]:
    """Returns the normalized view direction for 3D depth shading.

    Args:
        elev: View elevation angle in degrees.
        azim: View azimuth angle in degrees.

    Returns:
        Unit vector pointing along the projected viewing direction.
    """
    elev_r = math.radians(elev)
    azim_r = math.radians(azim)
    direction = (
        math.cos(elev_r) * math.sin(azim_r),
        -math.cos(elev_r) * math.cos(azim_r),
        math.sin(elev_r),
    )
    norm = math.sqrt(sum(item * item for item in direction))
    return tuple(item / norm for item in direction)


def _dot3(left: tuple[float, ...], right: tuple[float, float, float]) -> float:
    """Returns the three-dimensional dot product of two vectors.

    Args:
        left: Vector with at least three components.
        right: Three-component vector.

    Returns:
        Dot product using the first three components of ``left``.
    """
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _label_color_for_rgba(rgba: tuple[float, ...], *, fallback: str) -> str:
    """Chooses a readable label color for an RGBA marker color.

    Args:
        rgba: Color tuple whose first three components are RGB channels.
        fallback: Color to use when ``rgba`` does not contain RGB channels.

    Returns:
        Dark text for bright colors and white text for dark colors.
    """
    if len(rgba) < 3:
        return fallback
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "#202020" if luminance > 0.55 else "white"


__all__ = [
    "plot_graph",
]
