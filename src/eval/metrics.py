"""
src/eval/metrics.py
---------------------
Evaluation metrics for logistics optimisation results.

Provides:
  - Single-route metrics (Q1/Q2/Q3)
  - Multi-vehicle metrics (Q4)
  - CSV export helpers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.core.graph_model import ProblemGraph
from src.core.time_window import RouteTimingResult, simulate_route_timing

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 单路径指标
# ---------------------------------------------------------------------------


def single_route_metrics(
    route: Sequence[int],
    graph: ProblemGraph,
    alpha: float = 10.0,
    beta: float = 20.0,
) -> dict[str, object]:
    """Compute all metrics for a single vehicle route.

    Parameters
    ----------
    route : Sequence[int]
        Ordered node IDs (depot-inclusive recommended).
    graph : ProblemGraph
    alpha, beta : float
        Time-window penalty coefficients.

    Returns
    -------
    dict
        Keys:
        - ``route``: list[int]
        - ``total_travel_time``: float
        - ``total_service_time``: float
        - ``total_penalty``: float
        - ``objective``: float  (travel + penalty)
        - ``n_customers``: int
        - ``per_node``: list[dict]  — per-customer breakdown

    Complexity
    ----------
    O(N)
    """
    timing = simulate_route_timing(route, graph, alpha, beta)
    tw_enabled = not (abs(alpha) < 1e-12 and abs(beta) < 1e-12)
    per_node = [
        {
            "node_id": r.node_id,
            "arrival_time": r.arrival_time,
            "early_violation": r.early_violation if tw_enabled else 0.0,
            "late_violation": r.late_violation if tw_enabled else 0.0,
            "penalty": r.penalty if tw_enabled else 0.0,
        }
        for r in timing.records
    ]
    return {
        "route": timing.route,
        "total_travel_time": timing.total_travel_time,
        "total_service_time": timing.total_service_time,
        "total_penalty": timing.total_penalty,
        "objective": timing.total_travel_time + timing.total_penalty,
        "n_customers": len(timing.records),
        "per_node": per_node,
    }


# ---------------------------------------------------------------------------
# 多车辆指标
# ---------------------------------------------------------------------------


def multi_vehicle_metrics(
    routes: list[list[int]],
    graph: ProblemGraph,
    vehicle_capacity: float,
    alpha: float = 10.0,
    beta: float = 20.0,
) -> dict[str, object]:
    """Compute all metrics for a multi-vehicle solution.

    Parameters
    ----------
    routes : list[list[int]]
        One route per vehicle (depot-inclusive).
    graph : ProblemGraph
    vehicle_capacity : float
    alpha, beta : float

    Returns
    -------
    dict
        Keys include per-vehicle breakdown and global totals.

    Complexity
    ----------
    O(sum of route lengths)
    """
    vehicles = []
    total_travel = 0.0
    total_penalty = 0.0
    total_load = 0.0

    for k, route in enumerate(routes):
        timing = simulate_route_timing(route, graph, alpha, beta)
        load = sum(graph.demand(n) for n in route if n != graph.depot_id)
        total_travel += timing.total_travel_time
        total_penalty += timing.total_penalty
        total_load += load
        vehicles.append(
            {
                "vehicle_id": k,
                "route": timing.route,
                "n_customers": len(timing.records),
                "travel_time": timing.total_travel_time,
                "penalty": timing.total_penalty,
                "load": load,
                "load_ratio": load / vehicle_capacity if vehicle_capacity > 0 else 0.0,
                "capacity_ok": load <= vehicle_capacity,
                "per_node": [
                    {
                        "node_id": r.node_id,
                        "arrival_time": r.arrival_time,
                        "early_violation": r.early_violation,
                        "late_violation": r.late_violation,
                        "penalty": r.penalty,
                    }
                    for r in timing.records
                ],
            }
        )

    return {
        "n_vehicles": len(routes),
        "total_travel_time": total_travel,
        "total_penalty": total_penalty,
        "objective": total_travel + total_penalty,
        "total_load": total_load,
        "vehicles": vehicles,
    }


# ---------------------------------------------------------------------------
# 导出辅助函数
# ---------------------------------------------------------------------------


def metrics_to_dataframe(metrics: dict[str, object]) -> pd.DataFrame:
    """Convert per-node metrics to a flat DataFrame.

    Parameters
    ----------
    metrics : dict
        Output of ``single_route_metrics`` or ``multi_vehicle_metrics``.

    Returns
    -------
    pd.DataFrame

    Complexity
    ----------
    O(N)
    """
    if "per_node" in metrics:
        # 单车辆
        rows = metrics["per_node"]  # type: ignore[index]
    elif "vehicles" in metrics:
        # 多车辆
        rows = []
        for v in metrics["vehicles"]:  # type: ignore[index]
            for rec in v["per_node"]:
                rows.append({**rec, "vehicle_id": v["vehicle_id"]})
    else:
        rows = []

    return pd.DataFrame(rows)


def save_metrics_csv(
    metrics: dict[str, object],
    output_path: str | Path,
) -> None:
    """Save per-node metrics to CSV file.

    Parameters
    ----------
    metrics : dict
    output_path : str | Path

    Complexity
    ----------
    O(N)
    """
    df = metrics_to_dataframe(metrics)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved metrics CSV to %s", output_path)

