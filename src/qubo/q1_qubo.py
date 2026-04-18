"""
src/qubo/q1_qubo.py
--------------------
QUBO construction for Problem 1:
  Single vehicle, no time windows, no capacity constraint, 15 customers.

Encoding
--------
  x[i, p] ∈ {0, 1}: node i is visited at position p in the route.
  n = N+1 (total nodes including depot)
  Variable flat index: k = i*n + p

Objective
---------
  H = H_visit + H_position + H_cost

  H_visit    = A * sum_i (1 - sum_p x[i,p])^2   (each node visited once)
  H_position = B * sum_p (1 - sum_i x[i,p])^2   (each position has one node)
  H_cost     = sum_{i,j,p} T[i,j] * x[i,p] * x[j,(p+1)%n]  (travel time)

See docs/qubo_derivation.md for full derivation.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np

from src.core.graph_model import ProblemGraph
from src.qubo.penalties import (
    QDict,
    merge_qdicts,
    one_hot_penalty,
    qdict_to_matrix,
    route_cost_penalty,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class Q1QUBOResult(NamedTuple):
    """QUBO construction output for Problem 1.

    Attributes
    ----------
    Q : np.ndarray
        Shape (n_vars, n_vars) upper-triangular QUBO matrix.
    n_vars : int
        Total number of binary variables = (N+1)^2.
    n_nodes : int
        Total node count including depot (N+1).
    var_idx : callable
        Function mapping (node_id, position) -> flat index.
    """

    Q: np.ndarray
    n_vars: int
    n_nodes: int
    var_idx: object  # Callable[[int, int], int]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_q1_qubo(
    graph: ProblemGraph,
    penalty_visit: float = 500.0,
    penalty_position: float = 500.0,
) -> Q1QUBOResult:
    """Construct the QUBO matrix for Problem 1 (pure TSP, no TW/capacity).

    Parameters
    ----------
    graph : ProblemGraph
        Full problem graph with depot (node 0) and customers.
    penalty_visit : float
        Penalty strength A for the "visit-once" constraint.
    penalty_position : float
        Penalty strength B for the "one-per-position" constraint.

    Returns
    -------
    Q1QUBOResult

    Complexity
    ----------
    O(N³) for H_cost, O(N²) for constraints.
    """
    n = graph.n_customers + 1  # total nodes (0..N)
    n_vars = n * n

    # Variable index function: node i at position p → flat index
    def var_idx(node_id: int, position: int) -> int:
        """Map (node_id, position) pair to flat QUBO variable index.

        Parameters
        ----------
        node_id : int  0..N
        position : int  0..N

        Returns
        -------
        int  in [0, n_vars)

        Complexity: O(1)
        """
        return node_id * n + position

    logger.info(
        "Building Q1 QUBO: n_nodes=%d, n_vars=%d, A=%.1f, B=%.1f",
        n,
        n_vars,
        penalty_visit,
        penalty_position,
    )

    # H_visit: each node visited exactly once
    q_visit: QDict = {}
    for i in range(n):
        indices = [var_idx(i, p) for p in range(n)]
        partial = one_hot_penalty(indices, penalty_visit)
        q_visit = merge_qdicts(q_visit, partial)

    # H_position: each position occupied by exactly one node
    q_position: QDict = {}
    for p in range(n):
        indices = [var_idx(i, p) for i in range(n)]
        partial = one_hot_penalty(indices, penalty_position)
        q_position = merge_qdicts(q_position, partial)

    # H_cost: minimise total travel time
    q_cost = route_cost_penalty(graph.travel_time, n, var_idx)

    # Merge
    q_total = merge_qdicts(q_visit, q_position, q_cost)

    # Convert to dense matrix
    Q = qdict_to_matrix(q_total, n_vars)

    logger.debug(
        "Q1 QUBO built: shape=%s, non-zero entries=%d",
        Q.shape,
        int((Q != 0).sum()),
    )
    return Q1QUBOResult(Q=Q, n_vars=n_vars, n_nodes=n, var_idx=var_idx)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def decode_q1_solution(
    x: np.ndarray,
    n_nodes: int,
    var_idx_fn: object,
) -> list[int]:
    """Decode a binary QUBO solution vector into an ordered route.

    Parameters
    ----------
    x : np.ndarray
        Binary solution vector, shape (n_vars,).
    n_nodes : int
        Total node count (N+1).
    var_idx_fn : callable
        Same function used during QUBO construction.

    Returns
    -------
    list[int]
        Ordered route [0, c1, c2, ..., cN, 0] starting and ending at depot.
        If the solution is infeasible (missing/duplicate assignments), a
        best-effort route is returned and a warning is logged.

    Complexity
    ----------
    O(N²) for position extraction.
    """
    n = n_nodes
    route_positions: dict[int, int] = {}  # node_id -> assigned_position

    for node_id in range(n):
        for pos in range(n):
            idx = var_idx_fn(node_id, pos)
            if x[idx] > 0.5:
                if node_id in route_positions:
                    logger.warning(
                        "Node %d assigned to multiple positions; keeping first.", node_id
                    )
                else:
                    route_positions[node_id] = pos

    # Check all nodes assigned
    missing = [i for i in range(n) if i not in route_positions]
    if missing:
        logger.warning("Nodes with no position assignment: %s (decoding best-effort)", missing)
        # Assign missing nodes to leftover positions
        used_positions = set(route_positions.values())
        free_positions = [p for p in range(n) if p not in used_positions]
        for node_id, pos in zip(missing, free_positions):
            route_positions[node_id] = pos

    # Build ordered route from position assignments
    ordered = sorted(route_positions.keys(), key=lambda nid: route_positions[nid])

    # Depot must be first; rotate if needed
    if ordered and ordered[0] != 0:
        depot_idx = ordered.index(0)
        ordered = ordered[depot_idx:] + ordered[:depot_idx]

    # Return closed route: depot at start and end
    route = ordered + [0]
    return route
