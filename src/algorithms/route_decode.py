"""
src/algorithms/route_decode.py
--------------------------------
Decode QUBO binary solutions into valid routes.

Handles sub-graph decoding where node IDs in the sub-graph may differ
from the full problem node IDs.
"""

from __future__ import annotations

import logging

import numpy as np

from src.core.graph_model import ProblemGraph
from src.qubo.q1_qubo import Q1QUBOResult, decode_q1_solution

logger = logging.getLogger(__name__)


def decode_sub_route(
    x: np.ndarray,
    sub_graph: ProblemGraph,
    qubo_result: Q1QUBOResult,
) -> list[int]:
    """Decode a QUBO solution on a sub-graph to a customer-only route.

    The sub-graph uses the *original* node IDs (depot=0, customers=original IDs).
    This function decodes the binary vector, strips the depot from both ends,
    and returns only the customer IDs in order.

    Parameters
    ----------
    x : np.ndarray
        Binary QUBO solution vector.
    sub_graph : ProblemGraph
        Sub-graph on which QUBO was built.
    qubo_result : Q1QUBOResult
        QUBO metadata.

    Returns
    -------
    list[int]
        Ordered customer IDs for the sub-route (no depot).

    Complexity
    ----------
    O(K²) for decoding.
    """
    # 第一题解码返回局部节点索引 零表示仓库
    # 子问题需要将局部索引映射回全局节点编号
    route_local = decode_q1_solution(x, qubo_result.n_nodes, qubo_result.var_idx)

    local_to_global = [sub_graph.depot_id] + sorted(sub_graph.customer_ids)
    customers_global: list[int] = []
    for nid_local in route_local:
        nid_local_i = int(nid_local)
        if nid_local_i == 0:
            continue
        if 0 <= nid_local_i < len(local_to_global):
            customers_global.append(int(local_to_global[nid_local_i]))
        else:
            logger.warning(
                "Sub-route decode local index out of range: %d (size=%d)",
                nid_local_i,
                len(local_to_global),
            )
    return customers_global


def nearest_neighbour_route(
    graph: ProblemGraph,
    start: int | None = None,
    seed: int = 42,
) -> list[int]:
    """Construct a route using the nearest-neighbour heuristic.

    Starting from the depot (or ``start``), always move to the nearest
    unvisited customer, then return to depot.

    Parameters
    ----------
    graph : ProblemGraph
    start : int | None
        Starting node (defaults to depot 0).
    seed : int
        Random seed (used only if tie-breaking is needed).

    Returns
    -------
    list[int]
        Route including depot at start and end.

    Complexity
    ----------
    O(N²)
    """
    rng = np.random.default_rng(seed)
    if start is None:
        start = graph.depot_id

    unvisited = set(graph.customer_ids)
    route = [start]
    current = start

    while unvisited:
        # 选择最近的未访问节点
        nearest = min(unvisited, key=lambda n: graph.travel(current, n))
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    route.append(graph.depot_id)
    return route

