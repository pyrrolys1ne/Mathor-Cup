"""
src/eval/sensitivity.py
------------------------
问题四敏感性分析模块，研究车辆数 K 对目标值的影响。

在 min_vehicles 到 max_vehicles 范围内扫描 K，记录:
    - 总旅行时间
    - 时间窗惩罚总和
    - 综合目标值
    - 容量可行性
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.core.graph_model import ProblemGraph

logger = logging.getLogger(__name__)


@dataclass
class SensitivityPoint:
    """Single data point in the sensitivity curve.

    Attributes
    ----------
    n_vehicles : int
    total_travel_time : float
    total_penalty : float
    objective : float
    capacity_feasible : bool
    """

    n_vehicles: int
    total_travel_time: float
    total_penalty: float
    objective: float
    capacity_feasible: bool


@dataclass
class SensitivityResult:
    """Full sensitivity curve for Q4.

    Attributes
    ----------
    points : list[SensitivityPoint]
        One entry per vehicle count tried.
    best_k : int
        Vehicle count with minimum objective among feasible solutions.
    """

    points: list[SensitivityPoint] = field(default_factory=list)
    best_k: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 pandas DataFrame。"""
        return pd.DataFrame(
            [
                {
                    "n_vehicles": p.n_vehicles,
                    "total_travel_time": p.total_travel_time,
                    "total_penalty": p.total_penalty,
                    "objective": p.objective,
                    "capacity_feasible": p.capacity_feasible,
                }
                for p in self.points
            ]
        )

    def save_csv(self, path: str | Path) -> None:
        """将敏感性数据保存为 CSV。"""
        df = self.to_dataframe()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved sensitivity CSV to %s", path)


def run_vehicle_sensitivity(
    graph: ProblemGraph,
    solve_fn: "Callable[[ProblemGraph, int, float], list[list[int]]]",  # noqa: F821
    vehicle_capacity: float,
    alpha: float = 10.0,
    beta: float = 20.0,
    min_vehicles: int = 2,
    max_vehicles: int = 10,
    step: int = 1,
    weight_vehicles: float = 1.0,
    weight_travel: float = 1.0,
    weight_penalty: float = 1.0,
) -> SensitivityResult:
    """Run sensitivity analysis by sweeping the number of vehicles.

    For each K in [min_vehicles, max_vehicles] (step ``step``),
    calls ``solve_fn(graph, K, vehicle_capacity)`` to obtain vehicle routes,
    then evaluates the solution.

    Parameters
    ----------
    graph : ProblemGraph
    solve_fn : callable
        Function ``(graph, n_vehicles, vehicle_capacity) -> list[list[int]]``
        that returns vehicle routes for a given fleet size.
    vehicle_capacity : float
    alpha, beta : float
    min_vehicles, max_vehicles, step : int
    weight_vehicles, weight_travel, weight_penalty : float
        Weights of weighted objective
        ``weight_vehicles * n_vehicles + weight_travel * travel + weight_penalty * penalty``.

    Returns
    -------
    SensitivityResult

    Complexity
    ----------
    O((max_vehicles - min_vehicles) / step * solver_complexity)
    """
    from src.core.capacity import check_capacity
    from src.core.time_window import batch_route_penalties

    points: list[SensitivityPoint] = []

    for k in range(min_vehicles, max_vehicles + 1, step):
        logger.info("Sensitivity: solving for K=%d vehicles ...", k)
        try:
            routes = solve_fn(graph, k, vehicle_capacity)
            timings = batch_route_penalties(routes, graph, alpha, beta)
            travel = sum(t.total_travel_time for t in timings)
            penalty = sum(t.total_penalty for t in timings)
            n_used = len(routes)
            cap_report = check_capacity(routes, graph, vehicle_capacity)
            weighted_obj = (
                weight_vehicles * n_used
                + weight_travel * travel
                + weight_penalty * penalty
            )
            points.append(
                SensitivityPoint(
                    n_vehicles=n_used,
                    total_travel_time=travel,
                    total_penalty=penalty,
                    objective=weighted_obj,
                    capacity_feasible=cap_report.feasible,
                )
            )
            logger.info(
                "K=%d: used=%d, weighted_obj=%.2f, feasible=%s",
                k,
                n_used,
                weighted_obj,
                cap_report.feasible,
            )
        except Exception as exc:
            logger.warning("Sensitivity K=%d failed: %s", k, exc)

    # 查找可行解中的最优 K
    feasible = [p for p in points if p.capacity_feasible]
    best_k = min(feasible, key=lambda p: p.objective).n_vehicles if feasible else 0

    return SensitivityResult(points=points, best_k=best_k)

