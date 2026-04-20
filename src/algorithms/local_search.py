"""
src/algorithms/local_search.py
--------------------------------
Classical local-search improvement operators for TSP routes.

Operators implemented:
  - 2-opt: reverse a sub-segment of the route.
  - or-opt: remove one customer and reinsert at best position.

Both operators accept a *customer-only* permutation (no depot bookends)
and a ProblemGraph for travel-time lookups.
"""

from __future__ import annotations

import logging

from src.core.graph_model import ProblemGraph

logger = logging.getLogger(__name__)


def two_opt(
    route: list[int],
    graph: ProblemGraph,
    n_iter: int = 1000,
) -> list[int]:
    """Improve a customer route using the 2-opt operator.

    Iteratively reverses sub-segments while improvement is found,
    for up to ``n_iter`` non-improving iterations.

    Parameters
    ----------
    route : list[int]
        Customer-only permutation (depot excluded).
    graph : ProblemGraph
        Provides travel times for cost evaluation.
    n_iter : int
        Maximum number of non-improving iterations before stopping.

    Returns
    -------
    list[int]
        Improved customer permutation.

    Complexity
    ----------
    O(n_iter * N²) worst case.
    """
    best = list(route)
    best_cost = graph.route_travel_time(best)
    no_improve = 0

    while no_improve < n_iter:
        improved = False
        n = len(best)
        for i in range(n - 1):
            for j in range(i + 2, n):
                candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1:]
                cost = graph.route_travel_time(candidate)
                if cost < best_cost - 1e-9:
                    best = candidate
                    best_cost = cost
                    improved = True
        if not improved:
            no_improve += 1
        else:
            no_improve = 0

    logger.debug("2-opt finished: cost=%.4f", best_cost)
    return best


def or_opt(
    route: list[int],
    graph: ProblemGraph,
    n_iter: int = 500,
    segment_sizes: tuple[int, ...] = (1, 2, 3),
) -> list[int]:
    """Improve a customer route using the or-opt operator.

    For each segment size in ``segment_sizes``, tries all possible
    removals and reinsertion positions.

    Parameters
    ----------
    route : list[int]
        Customer-only permutation.
    graph : ProblemGraph
    n_iter : int
        Maximum non-improving passes before stopping.
    segment_sizes : tuple[int, ...]
        Segment sizes to try relocating (default 1, 2, 3).

    Returns
    -------
    list[int]
        Improved customer permutation.

    Complexity
    ----------
    O(n_iter * |segment_sizes| * N²) worst case.
    """
    best = list(route)
    best_cost = graph.route_travel_time(best)
    no_improve = 0

    while no_improve < n_iter:
        improved = False
        n = len(best)
        for seg_len in segment_sizes:
            for i in range(n - seg_len + 1):
                segment = best[i : i + seg_len]
                rest = best[:i] + best[i + seg_len:]
                for j in range(len(rest) + 1):
                    candidate = rest[:j] + segment + rest[j:]
                    if candidate == best:
                        continue
                    cost = graph.route_travel_time(candidate)
                    if cost < best_cost - 1e-9:
                        best = candidate
                        best_cost = cost
                        improved = True
        if not improved:
            no_improve += 1
        else:
            no_improve = 0

    logger.debug("or-opt finished: cost=%.4f", best_cost)
    return best

