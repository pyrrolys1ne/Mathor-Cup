"""
src/qubo/q2_qubo.py
--------------------
QUBO construction for Problem 2:
  Single vehicle, time-window penalty (soft), no capacity constraint, 15 customers.

Strategy
--------
Time-window penalties are non-linear in the route order variables, so they
cannot be directly embedded into a QUBO without linearisation.

We adopt **Approach A** (see docs/qubo_derivation.md):
  - The QUBO structure is identical to Q1 (visit + position + cost).
  - After decoding the binary solution to a route, the time-window penalty
    is computed classically on the decoded route.
  - The combined objective  obj = TravelTime + TW_penalty  is used for
    selection / post-optimisation.

This module re-exports the Q1 builder and adds a thin wrapper that
evaluates the combined objective post-decoding.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np

from src.core.graph_model import ProblemGraph
from src.core.time_window import RouteTimingResult, simulate_route_timing
from src.qubo.q1_qubo import Q1QUBOResult, build_q1_qubo, decode_q1_solution

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class Q2Result(NamedTuple):
    """Combined result for Problem 2.

    Attributes
    ----------
    route : list[int]
        Decoded route (depot-to-depot).
    timing : RouteTimingResult
        Full timing analysis including per-node penalties.
    qubo_result : Q1QUBOResult
        The underlying QUBO that was solved.
    objective : float
        total_travel_time + total_penalty (combined objective).
    """

    route: list[int]
    timing: RouteTimingResult
    qubo_result: Q1QUBOResult
    objective: float


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_q2_qubo(
    graph: ProblemGraph,
    penalty_visit: float = 500.0,
    penalty_position: float = 500.0,
) -> Q1QUBOResult:
    """Build QUBO for Problem 2.

    The QUBO structure is identical to Q1; time-window penalties are
    evaluated post-decoding.

    Parameters
    ----------
    graph : ProblemGraph
    penalty_visit : float
    penalty_position : float

    Returns
    -------
    Q1QUBOResult
        Same type as Q1 — re-used directly.

    Complexity
    ----------
    O(N³) — identical to Q1.
    """
    logger.info("Building Q2 QUBO (same structure as Q1 + post-decode TW evaluation)")
    return build_q1_qubo(graph, penalty_visit, penalty_position)


# ---------------------------------------------------------------------------
# Post-decode evaluator
# ---------------------------------------------------------------------------


def evaluate_q2_solution(
    x: np.ndarray,
    qubo_result: Q1QUBOResult,
    graph: ProblemGraph,
    alpha: float = 10.0,
    beta: float = 20.0,
) -> Q2Result:
    """Decode a binary solution and evaluate the combined Q2 objective.

    Parameters
    ----------
    x : np.ndarray
        Binary solution vector from solver.
    qubo_result : Q1QUBOResult
        QUBO metadata (n_nodes, var_idx).
    graph : ProblemGraph
    alpha : float
        Early-arrival penalty coefficient.
    beta : float
        Late-arrival penalty coefficient.

    Returns
    -------
    Q2Result

    Complexity
    ----------
    O(N²) for decode + O(N) for timing.
    """
    route = decode_q1_solution(x, qubo_result.n_nodes, qubo_result.var_idx)
    timing = simulate_route_timing(route, graph, alpha=alpha, beta=beta)
    objective = timing.total_travel_time + timing.total_penalty

    logger.info(
        "Q2 decode: travel=%.2f, penalty=%.2f, obj=%.2f",
        timing.total_travel_time,
        timing.total_penalty,
        objective,
    )
    return Q2Result(
        route=route,
        timing=timing,
        qubo_result=qubo_result,
        objective=objective,
    )
