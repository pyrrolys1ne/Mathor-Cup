"""
src/viz/plot_tradeoff.py
-------------------------
Trade-off and sensitivity analysis plots for Problem 4.

Generates:
  - Vehicle-count sensitivity curve (objective vs K).
  - Stacked bar: travel time vs time-window penalty per vehicle.
  - Objective comparison across experiment configurations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def plot_sensitivity_curve(
    df: pd.DataFrame,
    title: str = "Vehicle Count Sensitivity",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot objective value vs number of vehicles.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``SensitivityResult.to_dataframe()``.
        Required columns: ``n_vehicles``, ``objective``,
        ``total_travel_time``, ``total_penalty``, ``capacity_feasible``.
    title : str
    save_path : str | Path | None
    show : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    feasible = df[df["capacity_feasible"]]
    infeasible = df[~df["capacity_feasible"]]

    ax.plot(
        feasible["n_vehicles"],
        feasible["objective"],
        "o-",
        color="#377eb8",
        linewidth=2,
        label="Feasible",
    )
    if not infeasible.empty:
        ax.scatter(
            infeasible["n_vehicles"],
            infeasible["objective"],
            marker="x",
            color="#e41a1c",
            s=80,
            zorder=5,
            label="Infeasible",
        )

    ax.set_xlabel("Number of Vehicles (K)")
    ax.set_ylabel("Objective (Travel Time + Penalty)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_stacked_cost(
    vehicle_data: list[dict],
    title: str = "Cost Breakdown per Vehicle",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Stacked bar chart of travel time and penalty per vehicle.

    Parameters
    ----------
    vehicle_data : list[dict]
        Each dict must have: ``vehicle_id``, ``travel_time``, ``penalty``.
    title : str
    save_path : str | Path | None
    show : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = pd.DataFrame(vehicle_data)
    vehicle_ids = df["vehicle_id"].astype(str).tolist()
    travel = df["travel_time"].tolist()
    penalty = df["penalty"].tolist()

    x = range(len(vehicle_ids))
    fig, ax = plt.subplots(figsize=(max(6, len(vehicle_ids) * 0.8), 5))
    bars_travel = ax.bar(x, travel, label="Travel Time", color="#377eb8")
    bars_penalty = ax.bar(x, penalty, bottom=travel, label="TW Penalty", color="#e41a1c")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"V{v}" for v in vehicle_ids])
    ax.set_xlabel("Vehicle")
    ax.set_ylabel("Cost")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_objective_comparison(
    labels: list[str],
    objectives: list[float],
    title: str = "Objective Comparison",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Bar chart comparing objectives across configurations.

    Parameters
    ----------
    labels : list[str]
        Configuration labels (e.g., ['Q1', 'Q2', 'Q3 hybrid', 'Q4']).
    objectives : list[float]
        Corresponding objective values.
    title : str
    save_path : str | Path | None
    show : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    colours = ["#377eb8", "#4daf4a", "#ff7f00", "#e41a1c"][: len(labels)]
    ax.bar(labels, objectives, color=colours if len(colours) == len(labels) else "#377eb8")
    ax.set_ylabel("Objective Value")
    ax.set_title(title)
    for i, v in enumerate(objectives):
        ax.text(i, v * 1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()

    if save_path:
        _save(fig, save_path)
    if show:
        plt.show()
    return fig


def _save(fig: plt.Figure, path: str | Path) -> None:
    """保存图像到文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved figure to %s", path)
    plt.close(fig)

