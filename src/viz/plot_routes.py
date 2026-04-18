"""
src/viz/plot_routes.py
-----------------------
Route visualisation utilities.

Generates:
  - Single-vehicle route maps.
  - Multi-vehicle route maps with colour-coded vehicles.
  - Cluster visualisation for decomposition experiments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.core.graph_model import ProblemGraph

matplotlib.use("Agg")  # Non-interactive backend for server/CI environments

logger = logging.getLogger(__name__)

# Colour palette for multiple vehicles
_VEHICLE_COLOURS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#8dd3c7", "#bebada",
]


def plot_single_route(
    route: Sequence[int],
    graph: ProblemGraph,
    title: str = "Route",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot a single-vehicle route on a 2-D map.

    Parameters
    ----------
    route : Sequence[int]
        Ordered node IDs (depot-inclusive).
    graph : ProblemGraph
    title : str
    save_path : str | Path | None
        If given, save figure to this path.
    show : bool
        If True, call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure

    Complexity
    ----------
    O(N)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_route(ax, list(route), graph, colour="#377eb8", label="Route")
    _annotate_nodes(ax, graph)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_multi_vehicle_routes(
    routes: list[list[int]],
    graph: ProblemGraph,
    title: str = "Multi-Vehicle Routes",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot multiple vehicle routes on a single 2-D map.

    Parameters
    ----------
    routes : list[list[int]]
        One route per vehicle (depot-inclusive).
    graph : ProblemGraph
    title : str
    save_path : str | Path | None
    show : bool

    Returns
    -------
    matplotlib.figure.Figure

    Complexity
    ----------
    O(K * N)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    patches = []
    for k, route in enumerate(routes):
        colour = _VEHICLE_COLOURS[k % len(_VEHICLE_COLOURS)]
        _draw_route(ax, route, graph, colour=colour, label=f"V{k}")
        patches.append(mpatches.Patch(color=colour, label=f"Vehicle {k}"))

    _annotate_nodes(ax, graph)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(handles=patches, loc="best", fontsize=8)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_clusters(
    graph: ProblemGraph,
    labels: np.ndarray,
    customer_ids: list[int],
    title: str = "Customer Clusters",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Visualise customer cluster assignments.

    Parameters
    ----------
    graph : ProblemGraph
    labels : np.ndarray shape (N,)
        Cluster label per customer.
    customer_ids : list[int]
        Customer IDs corresponding to each label entry.
    title : str
    save_path : str | Path | None
    show : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = sorted(set(labels.tolist()))
    for lbl in unique_labels:
        colour = _VEHICLE_COLOURS[int(lbl) % len(_VEHICLE_COLOURS)]
        idxs = [i for i, l in enumerate(labels) if l == lbl]
        xs = [graph.coords(customer_ids[i])[0] for i in idxs]
        ys = [graph.coords(customer_ids[i])[1] for i in idxs]
        ax.scatter(xs, ys, c=colour, s=60, label=f"Cluster {lbl}", zorder=3)

    # Depot
    depot_x, depot_y = graph.coords(graph.depot_id)
    ax.scatter([depot_x], [depot_y], c="black", marker="*", s=200, zorder=4, label="Depot")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _draw_route(
    ax: plt.Axes,
    route: list[int],
    graph: ProblemGraph,
    colour: str,
    label: str,
) -> None:
    """Draw route arrows on an existing Axes."""
    if not route:
        return
    coords = [graph.coords(n) for n in route]
    xs, ys = zip(*coords)

    # Draw path
    ax.plot(xs, ys, "-o", color=colour, linewidth=1.5, markersize=5, label=label)

    # Arrows for direction
    for k in range(len(coords) - 1):
        x0, y0 = coords[k]
        x1, y1 = coords[k + 1]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color=colour, lw=1.0),
        )


def _annotate_nodes(ax: plt.Axes, graph: ProblemGraph) -> None:
    """Label all nodes with their IDs on the given Axes."""
    for nid in graph.all_ids:
        x, y = graph.coords(nid)
        marker = "*" if nid == graph.depot_id else "o"
        colour = "black" if nid == graph.depot_id else "steelblue"
        size = 150 if nid == graph.depot_id else 60
        ax.scatter([x], [y], c=colour, marker=marker, s=size, zorder=5)
        ax.annotate(str(nid), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)


def _save(fig: plt.Figure, path: str | Path) -> None:
    """Save figure to file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved figure to %s", path)
    plt.close(fig)
