"""
src/solvers/hybrid_large_scale.py
-----------------------------------
Hybrid large-scale solver for Problems 3 and 4 (50 customers).

Strategy
--------
1. **Clustering**: Divide customers into K groups using K-Means (or DBSCAN).
2. **Sub-problem solving**: Build a QUBO for each cluster and solve with SA
   (or Kaiwu if available).
3. **Route stitching**: Connect sub-routes via depot and nearest-neighbour
   inter-cluster links.
4. **Local repair**: Apply 2-opt / or-opt to the stitched global route.

This decomposition keeps each QUBO at ≤ (cluster_size+1)² variables,
staying within quantum device limits while covering the full 50-customer
instance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.algorithms.clustering import cluster_customers
from src.algorithms.local_search import two_opt, or_opt
from src.algorithms.route_decode import decode_sub_route
from src.core.graph_model import ProblemGraph, subgraph
from src.qubo.q1_qubo import build_q1_qubo
from src.solvers.sa_solver import SAConfig, SAResult, solve_qubo_sa, solve_route_sa

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HybridConfig:
    """Configuration for the hybrid large-scale solver.

    Attributes
    ----------
    cluster_method : str
        Clustering algorithm: ``'kmeans'`` or ``'dbscan'``.
    n_clusters : int
        Number of clusters (only for K-Means).
    sub_solver : str
        Sub-problem solver: ``'sa'`` or ``'kaiwu'``.
    local_search_iter : int
        Number of local-search improvement iterations after stitching.
    seed : int
        Random seed.
    sa_cfg : SAConfig
        SA config used for sub-problem solving.
    """

    cluster_method: str = "kmeans"
    n_clusters: int = 5
    sub_solver: str = "sa"
    local_search_iter: int = 500
    seed: int = 42
    sa_cfg: SAConfig = field(default_factory=SAConfig)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class HybridResult:
    """Output of the hybrid solver.

    Attributes
    ----------
    global_route : list[int]
        Full customer sequence (depot-to-depot).
    cluster_routes : list[list[int]]
        Per-cluster routes before stitching.
    best_cost : float
        Objective value of the final route.
    cluster_labels : np.ndarray
        Cluster assignment for each customer.
    """

    global_route: list[int]
    cluster_routes: list[list[int]]
    best_cost: float
    cluster_labels: np.ndarray


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def solve_hybrid(
    graph: ProblemGraph,
    cost_fn: "Callable[[list[int]], float]",  # noqa: F821
    cfg: HybridConfig | None = None,
) -> HybridResult:
    """Run the hybrid decomposition solver.

    Parameters
    ----------
    graph : ProblemGraph
        Full problem graph (50 customers).
    cost_fn : Callable[[list[int]], float]
        Maps a full customer permutation to objective value.
    cfg : HybridConfig | None

    Returns
    -------
    HybridResult

    Complexity
    ----------
    O(K * C³) for QUBO building + O(N²) for local search.
    K = n_clusters, C = avg customers per cluster.
    """
    if cfg is None:
        cfg = HybridConfig()

    customer_ids = graph.customer_ids

    # Step 1: Cluster customers
    labels, centres = cluster_customers(
        graph,
        customer_ids,
        method=cfg.cluster_method,
        n_clusters=cfg.n_clusters,
        seed=cfg.seed,
    )

    # Group customers by cluster
    cluster_map: dict[int, list[int]] = {}
    for cid, label in zip(customer_ids, labels):
        cluster_map.setdefault(int(label), []).append(cid)

    logger.info(
        "Hybrid solver: %d clusters, sizes=%s",
        len(cluster_map),
        {k: len(v) for k, v in sorted(cluster_map.items())},
    )

    # Step 2: Solve each sub-problem
    cluster_routes: list[list[int]] = []
    for cluster_id, cids in sorted(cluster_map.items()):
        sub_route = _solve_subproblem(graph, cids, cfg)
        cluster_routes.append(sub_route)
        logger.debug("Cluster %d route: %s", cluster_id, sub_route)

    # Step 3: Stitch sub-routes into a global route
    stitched = _stitch_routes(cluster_routes, graph)
    logger.info("Stitched global route length: %d", len(stitched))

    # Step 4: Local repair
    repaired = two_opt(stitched, graph, n_iter=cfg.local_search_iter)
    repaired = or_opt(repaired, graph, n_iter=cfg.local_search_iter // 2)

    final_cost = cost_fn(repaired)
    logger.info("Hybrid solver final cost: %.4f", final_cost)

    return HybridResult(
        global_route=[graph.depot_id] + repaired + [graph.depot_id],
        cluster_routes=cluster_routes,
        best_cost=final_cost,
        cluster_labels=labels,
    )


# ---------------------------------------------------------------------------
# Sub-problem solver
# ---------------------------------------------------------------------------


def _solve_subproblem(
    graph: ProblemGraph,
    customer_ids: list[int],
    cfg: HybridConfig,
) -> list[int]:
    """Solve a single cluster sub-problem.

    Returns a customer-only permutation (no depot bookends).

    Parameters
    ----------
    graph : ProblemGraph
    customer_ids : list[int]
    cfg : HybridConfig

    Returns
    -------
    list[int]
        Ordered customer IDs for this cluster.
    """
    if len(customer_ids) <= 1:
        return list(customer_ids)

    if cfg.sub_solver == "kaiwu":
        try:
            from src.solvers.kaiwu_solver import KaiwuUnavailableError, solve_qubo_kaiwu

            sub = subgraph(graph, customer_ids)
            qubo_result = build_q1_qubo(sub)
            x = solve_qubo_kaiwu(qubo_result.Q)
            route = decode_sub_route(x, sub, qubo_result)
            return route
        except Exception as exc:
            logger.warning("Kaiwu sub-solver failed (%s); falling back to SA.", exc)

    # SA on permutation space (default / fallback)
    sub = subgraph(graph, customer_ids)
    depot = graph.depot_id

    def _cost(perm: list[int]) -> float:
        return sub.route_travel_time(perm)

    result = solve_route_sa(customer_ids, _cost, cfg.sa_cfg)
    return result.best_solution


# ---------------------------------------------------------------------------
# Route stitching
# ---------------------------------------------------------------------------


def _stitch_routes(
    cluster_routes: list[list[int]],
    graph: ProblemGraph,
) -> list[int]:
    """Concatenate cluster routes into a single customer sequence.

    Orders clusters by nearest-neighbour from the depot, then appends
    each cluster's customer sequence in order.

    Parameters
    ----------
    cluster_routes : list[list[int]]
        Customer-only permutations for each cluster (no depot).
    graph : ProblemGraph

    Returns
    -------
    list[int]
        Flat customer permutation (no depot).

    Complexity
    ----------
    O(K²) for cluster ordering.
    """
    if not cluster_routes:
        return []

    # Compute representative node for each cluster (first customer)
    unvisited = list(range(len(cluster_routes)))
    current_node = graph.depot_id
    ordered: list[list[int]] = []

    while unvisited:
        # Find cluster whose first node is nearest to current position
        best_idx = min(
            unvisited,
            key=lambda ci: graph.travel(current_node, cluster_routes[ci][0])
            if cluster_routes[ci]
            else float("inf"),
        )
        ordered.append(cluster_routes[best_idx])
        current_node = cluster_routes[best_idx][-1] if cluster_routes[best_idx] else current_node
        unvisited.remove(best_idx)

    # Flatten
    return [node for sub in ordered for node in sub]
