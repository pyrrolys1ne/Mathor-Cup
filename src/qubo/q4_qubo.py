"""
src/qubo/q4_qubo.py
--------------------
QUBO construction helpers for Problem 4:
  Multi-vehicle, time-window penalty, capacity constraint, 50 customers.

Due to the scale (50 customers × multiple vehicles), we use a decomposition
strategy (see src/algorithms/vehicle_assignment.py) and solve per-vehicle
sub-QUBOs using the Q1/Q2 builder.

This module provides:
  1. A helper to build per-vehicle QUBOs from pre-assigned customer groups.
  2. A multi-vehicle objective evaluator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.core.capacity import CapacityReport, check_capacity, route_demand
from src.core.graph_model import ProblemGraph, subgraph
from src.core.time_window import RouteTimingResult, simulate_route_timing
from src.qubo.q1_qubo import Q1QUBOResult, build_q1_qubo, decode_q1_solution

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class VehicleResult:
    """Result for a single vehicle route in Q4.

    Attributes
    ----------
    vehicle_id : int
    route : list[int]
    timing : RouteTimingResult
    load : float
    """

    vehicle_id: int
    route: list[int]
    timing: RouteTimingResult
    load: float


@dataclass
class Q4Result:
    """Combined multi-vehicle result for Problem 4.

    Attributes
    ----------
    vehicle_results : list[VehicleResult]
    total_travel_time : float
    total_penalty : float
    objective : float
        total_travel_time + total_penalty
    n_vehicles : int
    capacity_report : CapacityReport
    """

    vehicle_results: list[VehicleResult] = field(default_factory=list)
    total_travel_time: float = 0.0
    total_penalty: float = 0.0
    objective: float = 0.0
    n_vehicles: int = 0
    capacity_report: CapacityReport | None = None


# ---------------------------------------------------------------------------
# 单车辆 QUBO 构造
# ---------------------------------------------------------------------------


def build_vehicle_qubo(
    graph: ProblemGraph,
    customer_ids: list[int],
    penalty_visit: float = 500.0,
    penalty_position: float = 500.0,
) -> tuple[Q1QUBOResult, ProblemGraph]:
    """Build a QUBO for one vehicle's assigned customers.

    Extracts a sub-graph containing the depot + assigned customers, then
    constructs a Q1-style QUBO on the sub-graph.

    Parameters
    ----------
    graph : ProblemGraph
        Full problem graph.
    customer_ids : list[int]
        Customers assigned to this vehicle.
    penalty_visit, penalty_position : float
        QUBO penalty coefficients.

    Returns
    -------
    tuple[Q1QUBOResult, ProblemGraph]
        QUBO result and the sub-graph used for decoding.

    Complexity
    ----------
    O(K³) where K = len(customer_ids).
    """
    sub = subgraph(graph, customer_ids)
    qubo_result = build_q1_qubo(sub, penalty_visit, penalty_position)
    return qubo_result, sub


# ---------------------------------------------------------------------------
# 多车辆评估
# ---------------------------------------------------------------------------


def evaluate_q4_solution(
    vehicle_routes: list[list[int]],
    graph: ProblemGraph,
    vehicle_capacity: float,
    alpha: float = 10.0,
    beta: float = 20.0,
) -> Q4Result:
    """Evaluate a complete multi-vehicle solution for Q4.

    Parameters
    ----------
    vehicle_routes : list[list[int]]
        One route per vehicle (depot-inclusive).
    graph : ProblemGraph
    vehicle_capacity : float
    alpha, beta : float
        Time-window penalty coefficients.

    Returns
    -------
    Q4Result

    Complexity
    ----------
    O(K * max_route_length)
    """
    result = Q4Result(n_vehicles=len(vehicle_routes))
    vehicle_results: list[VehicleResult] = []

    for k, route in enumerate(vehicle_routes):
        timing = simulate_route_timing(route, graph, alpha, beta)
        load = route_demand(route, graph)
        vehicle_results.append(
            VehicleResult(vehicle_id=k, route=route, timing=timing, load=load)
        )
        result.total_travel_time += timing.total_travel_time
        result.total_penalty += timing.total_penalty

    result.objective = result.total_travel_time + result.total_penalty
    result.vehicle_results = vehicle_results
    result.capacity_report = check_capacity(vehicle_routes, graph, vehicle_capacity)

    logger.info(
        "Q4 evaluation: K=%d, travel=%.2f, penalty=%.2f, obj=%.2f, capacity_ok=%s",
        result.n_vehicles,
        result.total_travel_time,
        result.total_penalty,
        result.objective,
        result.capacity_report.feasible,
    )
    return result

