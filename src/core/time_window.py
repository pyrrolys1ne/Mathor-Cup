"""
src/core/time_window.py
------------------------
Time-window penalty calculation and route arrival-time simulation.

Penalty formula (from problem specification):
    penalty_i = alpha * max(0, e_i - t_i)^2 + beta * max(0, t_i - l_i)^2

Default: alpha=10 (early arrival), beta=20 (late arrival).

No-wait assumption: the vehicle does NOT wait at the depot or any customer
for the time window to open — it arrives and begins service immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.core.graph_model import ProblemGraph


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ArrivalRecord:
    """Stores arrival / service timing and penalty for a single node.

    Attributes
    ----------
    node_id : int
    arrival_time : float
        Time at which the vehicle arrives at the node (= service start, no wait).
    early_violation : float
        max(0, e_i - t_i) — how much earlier than window opening.
    late_violation : float
        max(0, t_i - l_i) — how much later than window closing.
    penalty : float
        Computed penalty value.
    """

    node_id: int
    arrival_time: float
    early_violation: float
    late_violation: float
    penalty: float


@dataclass
class RouteTimingResult:
    """Complete timing analysis for a single vehicle route.

    Attributes
    ----------
    route : list[int]
        Full node sequence (depot → … → depot).
    total_travel_time : float
        Sum of all arc travel times (excluding service times for transit).
    total_service_time : float
        Sum of service durations at each visited customer.
    total_penalty : float
        Sum of time-window penalties across all customers.
    records : list[ArrivalRecord]
        Per-node timing records (customers only, not depot).
    """

    route: list[int]
    total_travel_time: float
    total_service_time: float
    total_penalty: float
    records: list[ArrivalRecord]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_penalty(
    arrival: float,
    earliest: float,
    latest: float,
    alpha: float = 10.0,
    beta: float = 20.0,
) -> tuple[float, float, float]:
    """Compute time-window penalty for a single node visit.

    Mathematical definition:
        early_viol = max(0, e_i - t_i)
        late_viol  = max(0, t_i - l_i)
        penalty    = alpha * early_viol^2 + beta * late_viol^2

    Parameters
    ----------
    arrival : float
        Actual arrival time at the node.
    earliest : float
        Earliest allowed service start (e_i).
    latest : float
        Latest allowed service start (l_i).
    alpha : float
        Penalty coefficient for early arrival (default 10).
    beta : float
        Penalty coefficient for late arrival (default 20).

    Returns
    -------
    tuple[float, float, float]
        ``(early_violation, late_violation, penalty)``

    Complexity
    ----------
    O(1)
    """
    early_viol = max(0.0, earliest - arrival)
    late_viol = max(0.0, arrival - latest)
    penalty = alpha * early_viol**2 + beta * late_viol**2
    return early_viol, late_viol, penalty


def simulate_route_timing(
    route: Sequence[int],
    graph: ProblemGraph,
    alpha: float = 10.0,
    beta: float = 20.0,
) -> RouteTimingResult:
    """Simulate a vehicle traversal and compute timings + penalties.

    The vehicle departs from the depot at time 0, travels the given route
    (no waiting), and returns to the depot.  Service time at each customer
    is added to the running clock after arrival.

    Parameters
    ----------
    route : Sequence[int]
        Ordered node IDs.  Depot bookends (0) are added automatically if
        absent.
    graph : ProblemGraph
        Problem instance providing travel times, time windows, service times.
    alpha : float
        Early-arrival penalty coefficient.
    beta : float
        Late-arrival penalty coefficient.

    Returns
    -------
    RouteTimingResult
        Full timing analysis with per-node ``ArrivalRecord`` objects.

    Complexity
    ----------
    O(|route|)
    """
    full_route = list(route)
    depot = graph.depot_id

    # Ensure depot bookends
    if not full_route or full_route[0] != depot:
        full_route = [depot] + full_route
    if full_route[-1] != depot:
        full_route.append(depot)

    current_time = 0.0
    total_travel = 0.0
    total_service = 0.0
    total_penalty = 0.0
    records: list[ArrivalRecord] = []

    for k in range(len(full_route) - 1):
        from_node = full_route[k]
        to_node = full_route[k + 1]

        # Travel to next node
        arc_time = graph.travel(from_node, to_node)
        current_time += arc_time
        total_travel += arc_time

        if to_node == depot:
            break  # Return leg — no service or penalty at depot

        # Compute service start (no waiting)
        arrival_time = current_time
        e_i, l_i = graph.time_window(to_node)
        early_viol, late_viol, pen = compute_penalty(arrival_time, e_i, l_i, alpha, beta)

        total_penalty += pen
        records.append(
            ArrivalRecord(
                node_id=to_node,
                arrival_time=arrival_time,
                early_violation=early_viol,
                late_violation=late_viol,
                penalty=pen,
            )
        )

        # Service time consumed before departing
        s_i = graph.service_time(to_node)
        current_time += s_i
        total_service += s_i

    return RouteTimingResult(
        route=full_route,
        total_travel_time=total_travel,
        total_service_time=total_service,
        total_penalty=total_penalty,
        records=records,
    )


def batch_route_penalties(
    routes: list[Sequence[int]],
    graph: ProblemGraph,
    alpha: float = 10.0,
    beta: float = 20.0,
) -> list[RouteTimingResult]:
    """Compute timing results for multiple routes.

    Parameters
    ----------
    routes : list[Sequence[int]]
        List of route node sequences.
    graph : ProblemGraph
    alpha, beta : float
        Penalty coefficients.

    Returns
    -------
    list[RouteTimingResult]

    Complexity
    ----------
    O(sum of route lengths)
    """
    return [simulate_route_timing(r, graph, alpha, beta) for r in routes]


def compute_penalty_array(
    arrivals: np.ndarray,
    earliests: np.ndarray,
    latests: np.ndarray,
    alpha: float = 10.0,
    beta: float = 20.0,
) -> np.ndarray:
    """Vectorised penalty computation for an array of nodes.

    Parameters
    ----------
    arrivals : np.ndarray shape (N,)
        Arrival times.
    earliests : np.ndarray shape (N,)
        Earliest time windows.
    latests : np.ndarray shape (N,)
        Latest time windows.
    alpha, beta : float

    Returns
    -------
    np.ndarray shape (N,)
        Per-node penalties.

    Complexity
    ----------
    O(N)
    """
    early_viol = np.maximum(0.0, earliests - arrivals)
    late_viol = np.maximum(0.0, arrivals - latests)
    return alpha * early_viol**2 + beta * late_viol**2
