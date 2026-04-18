"""
src/core/capacity.py
---------------------
Capacity constraint utilities for multi-vehicle routing (Q4).

Each vehicle has a maximum capacity Q.  The total demand of all customers
assigned to a vehicle must not exceed Q.

Also provides helpers for feasibility checking and demand-aware clustering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.graph_model import ProblemGraph


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class CapacityReport:
    """Summary of capacity feasibility for a set of vehicle routes.

    Attributes
    ----------
    feasible : bool
        True iff all routes satisfy the capacity constraint.
    vehicle_loads : list[float]
        Total demand served by each vehicle.
    violations : list[int]
        Indices of vehicles that exceed capacity.
    vehicle_capacity : float
        The capacity limit used for checking.
    """

    feasible: bool
    vehicle_loads: list[float]
    violations: list[int]
    vehicle_capacity: float

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Capacity feasibility: {'OK' if self.feasible else 'VIOLATED'}",
            f"  Vehicle capacity : {self.vehicle_capacity}",
            f"  Number of routes : {len(self.vehicle_loads)}",
        ]
        for k, load in enumerate(self.vehicle_loads):
            status = "OK" if load <= self.vehicle_capacity else "OVER"
            lines.append(f"  Vehicle {k}: load={load:.1f} [{status}]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def check_capacity(
    routes: list[list[int]],
    graph: ProblemGraph,
    vehicle_capacity: float,
) -> CapacityReport:
    """Check whether all vehicle routes satisfy the capacity constraint.

    Parameters
    ----------
    routes : list[list[int]]
        Each inner list is an ordered sequence of node IDs for one vehicle,
        including the depot at start/end.
    graph : ProblemGraph
        Provides node demands via ``graph.demand(node_id)``.
    vehicle_capacity : float
        Maximum load per vehicle.

    Returns
    -------
    CapacityReport

    Complexity
    ----------
    O(sum of route lengths)
    """
    loads: list[float] = []
    violations: list[int] = []

    for k, route in enumerate(routes):
        load = sum(
            graph.demand(n)
            for n in route
            if n != graph.depot_id
        )
        loads.append(load)
        if load > vehicle_capacity:
            violations.append(k)

    return CapacityReport(
        feasible=(len(violations) == 0),
        vehicle_loads=loads,
        violations=violations,
        vehicle_capacity=vehicle_capacity,
    )


def minimum_vehicles(
    graph: ProblemGraph,
    vehicle_capacity: float,
) -> int:
    """Compute the theoretical lower bound on the number of vehicles needed.

    Lower bound = ceil(total_demand / vehicle_capacity).

    Parameters
    ----------
    graph : ProblemGraph
    vehicle_capacity : float

    Returns
    -------
    int
        Minimum number of vehicles required.

    Complexity
    ----------
    O(N)
    """
    total = sum(graph.demand(n) for n in graph.customer_ids)
    return int(np.ceil(total / vehicle_capacity))


def split_route_by_capacity(
    customer_ids: list[int],
    graph: ProblemGraph,
    vehicle_capacity: float,
) -> list[list[int]]:
    """Greedily split a customer list into capacity-feasible groups.

    Customers are processed in the given order; a new vehicle is started
    whenever adding the next customer would exceed capacity.

    Parameters
    ----------
    customer_ids : list[int]
        Ordered list of customer IDs (no depot).
    graph : ProblemGraph
    vehicle_capacity : float

    Returns
    -------
    list[list[int]]
        Each inner list is a group of customer IDs for one vehicle.
        Does not include depot; callers should prepend/append depot.

    Complexity
    ----------
    O(N)
    """
    routes: list[list[int]] = []
    current_route: list[int] = []
    current_load = 0.0

    for cid in customer_ids:
        d = graph.demand(cid)
        if current_route and current_load + d > vehicle_capacity:
            routes.append(current_route)
            current_route = []
            current_load = 0.0
        current_route.append(cid)
        current_load += d

    if current_route:
        routes.append(current_route)

    return routes


def route_demand(route: list[int], graph: ProblemGraph) -> float:
    """Compute total demand for a single route.

    Parameters
    ----------
    route : list[int]
        Node IDs (depot may be included; its demand is 0).
    graph : ProblemGraph

    Returns
    -------
    float

    Complexity
    ----------
    O(|route|)
    """
    return sum(graph.demand(n) for n in route if n != graph.depot_id)
