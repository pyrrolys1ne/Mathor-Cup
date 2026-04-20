"""
src/algorithms/vehicle_assignment.py
--------------------------------------
Vehicle assignment strategies for multi-vehicle routing (Q4).

Given a set of clusters, assigns clusters to vehicles respecting
capacity constraints and minimising total routes.
"""

from __future__ import annotations

import logging

import numpy as np

from src.core.capacity import split_route_by_capacity
from src.core.graph_model import ProblemGraph

logger = logging.getLogger(__name__)


def assign_customers_to_vehicles(
    customer_ids: list[int],
    graph: ProblemGraph,
    vehicle_capacity: float,
    cluster_labels: np.ndarray | None = None,
    seed: int = 42,
) -> list[list[int]]:
    """Assign customers to vehicles respecting capacity constraints.

    Strategy:
    1. If cluster labels are provided, use one vehicle per cluster (if capacity allows).
    2. If a cluster exceeds capacity, split it greedily.
    3. If no cluster labels, split the full customer list greedily.

    Parameters
    ----------
    customer_ids : list[int]
        All customer IDs to assign.
    graph : ProblemGraph
    vehicle_capacity : float
    cluster_labels : np.ndarray | None
        Cluster label per customer (same order as customer_ids).
    seed : int
        Unused; kept for interface consistency.

    Returns
    -------
    list[list[int]]
        One list of customer IDs per vehicle (no depot).

    Complexity
    ----------
    O(N)
    """
    if cluster_labels is None:
        # 无聚类时使用贪心分组
        return split_route_by_capacity(customer_ids, graph, vehicle_capacity)

    # 按聚类分组
    cluster_map: dict[int, list[int]] = {}
    for cid, label in zip(customer_ids, cluster_labels):
        cluster_map.setdefault(int(label), []).append(cid)

    routes: list[list[int]] = []
    for label, cids in sorted(cluster_map.items()):
        sub_routes = split_route_by_capacity(cids, graph, vehicle_capacity)
        routes.extend(sub_routes)
        if len(sub_routes) > 1:
            logger.debug(
                "Cluster %d split into %d vehicles due to capacity.", label, len(sub_routes)
            )

    logger.info("Vehicle assignment: %d vehicles for %d customers", len(routes), len(customer_ids))
    return routes


def lexicographic_vehicle_min(
    customer_ids: list[int],
    graph: ProblemGraph,
    vehicle_capacity: float,
) -> int:
    """Compute the minimum number of vehicles required (lower bound).

    Lower bound = ceil(total_demand / vehicle_capacity).

    Parameters
    ----------
    customer_ids : list[int]
    graph : ProblemGraph
    vehicle_capacity : float

    Returns
    -------
    int

    Complexity
    ----------
    O(N)
    """
    total_demand = sum(graph.demand(cid) for cid in customer_ids)
    import math
    return math.ceil(total_demand / vehicle_capacity)

