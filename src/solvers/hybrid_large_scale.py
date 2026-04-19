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
from typing import Callable, Sequence

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
    cost_fn: "Callable[[list[int]], float]",
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
        sub_route = _solve_subproblem(graph, cids, cfg, cost_fn)
        cluster_routes.append(sub_route)
        logger.debug("Cluster %d route: %s", cluster_id, sub_route)

    # Step 3: Stitch sub-routes into a global route
    stitched = _stitch_routes(cluster_routes, graph)
    logger.info("Stitched global route length: %d", len(stitched))

    # Step 4: Local repair (objective-safe acceptance)
    repaired = list(stitched)
    best_cost = cost_fn(repaired)

    cand_two_opt = two_opt(repaired, graph, n_iter=cfg.local_search_iter)
    cand_cost = cost_fn(cand_two_opt)
    if cand_cost < best_cost:
        repaired = cand_two_opt
        best_cost = cand_cost

    cand_or_opt = or_opt(repaired, graph, n_iter=cfg.local_search_iter // 2)
    cand_cost = cost_fn(cand_or_opt)
    if cand_cost < best_cost:
        repaired = cand_or_opt
        best_cost = cand_cost

    # Focused relocate on high late-violation nodes.
    cand_reloc = _late_node_relocate(
        repaired,
        graph,
        cost_fn,
        n_iter=max(30, cfg.local_search_iter // 4),
        max_nodes=10,
        max_shifts=12,
    )
    cand_cost = cost_fn(cand_reloc)
    if cand_cost < best_cost:
        repaired = cand_reloc
        best_cost = cand_cost

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
    cost_fn: "Callable[[list[int]], float]",
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
            if len(route) >= 3:
                improved = two_opt(route, graph, n_iter=max(50, cfg.local_search_iter // 5))
                if cost_fn(improved) < cost_fn(route):
                    route = improved
            return route
        except Exception as exc:
            logger.warning("Kaiwu sub-solver failed (%s); falling back to SA.", exc)

    # SA on permutation space (default / fallback)
    result = solve_route_sa(customer_ids, cost_fn, cfg.sa_cfg)
    best = result.best_solution
    if len(best) >= 3:
        improved = two_opt(best, graph, n_iter=max(50, cfg.local_search_iter // 5))
        if cost_fn(improved) < cost_fn(best):
            best = improved
    return best


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

    # Time-window-aware nearest stitching: distance + lateness-risk score.
    unvisited = list(range(len(cluster_routes)))
    current_node = graph.depot_id
    current_time = 0.0
    ordered: list[list[int]] = []

    while unvisited:
        # Prefer clusters that are close and whose earliest latest-time is
        # unlikely to be missed by current ETA.
        def _score(ci: int) -> float:
            route = cluster_routes[ci]
            if not route:
                return float("inf")
            first = route[0]
            dist = float(graph.travel(current_node, first))
            eta = current_time + dist
            urgency = min(float(graph.time_window(n)[1]) for n in route)
            lateness_risk = max(0.0, eta - urgency)
            return dist + 5.0 * lateness_risk

        best_idx = min(
            unvisited,
            key=_score,
        )
        chosen = cluster_routes[best_idx]
        ordered.append(chosen)

        if chosen:
            current_time += float(graph.travel(current_node, chosen[0]))
            prev = chosen[0]
            current_time += float(graph.service_time(prev))
            for nxt in chosen[1:]:
                current_time += float(graph.travel(prev, nxt))
                current_time += float(graph.service_time(nxt))
                prev = nxt
            current_node = chosen[-1]

        unvisited.remove(best_idx)

    # Flatten
    return [node for sub in ordered for node in sub]


def _late_violation_profile(route: list[int], graph: ProblemGraph) -> list[tuple[int, int, float]]:
    """Return (position, node_id, late_violation) for nodes with late violation."""
    if not route:
        return []

    prof: list[tuple[int, int, float]] = []
    t = 0.0
    prev = graph.depot_id
    for pos, node in enumerate(route):
        t += float(graph.travel(prev, node))
        _, l_i = graph.time_window(node)
        late = max(0.0, t - float(l_i))
        if late > 0:
            prof.append((pos, node, late))
        t += float(graph.service_time(node))
        prev = node

    prof.sort(key=lambda x: x[2], reverse=True)
    return prof


def _late_node_relocate(
    route: list[int],
    graph: ProblemGraph,
    cost_fn: Callable[[list[int]], float],
    n_iter: int = 120,
    max_nodes: int = 10,
    max_shifts: int = 12,
) -> list[int]:
    """Relocate heavily late nodes earlier if objective improves."""
    if len(route) < 4:
        return list(route)

    best = list(route)
    best_cost = cost_fn(best)

    for _ in range(max(1, n_iter)):
        late_nodes = _late_violation_profile(best, graph)[:max_nodes]
        if not late_nodes:
            break

        improved = False
        for pos, _node, _late in late_nodes:
            upper = min(pos, max_shifts)
            for target in range(0, upper):
                cand = list(best)
                moved = cand.pop(pos)
                cand.insert(target, moved)
                c = cost_fn(cand)
                if c + 1e-9 < best_cost:
                    best = cand
                    best_cost = c
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best
