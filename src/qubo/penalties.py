"""
src/qubo/penalties.py
----------------------
Penalty-term helpers for QUBO construction.

All functions return dictionaries mapping ``(row_idx, col_idx) -> coefficient``
that can be merged into the global Q matrix (upper-triangular convention).

Linear terms are stored on the diagonal: ``(k, k)``.
Quadratic cross-terms are stored with ``row < col``: ``(k, l)`` where k < l.

Mathematical background
-----------------------
For a QUBO  min x^T Q x  with  x ∈ {0,1}^n:
  - x_i^2 = x_i  (binary), so linear terms land on the diagonal.
  - Cross terms  2 * c * x_i * x_j  are split symmetrically → upper triangle
    stores coefficient  2c,  lower triangle stores 0 (upper-triangular form).

Constraint  "sum_j x_j = 1"  expands to:
    A * (1 - sum_j x_j)^2
  = A - 2A * sum_j x_j + A * sum_{j,k} x_j x_k
  = A - 2A * sum_j x_j + A * sum_j x_j^2 + 2A * sum_{j<k} x_j x_k
  = (constant) + sum_j (A - 2A) * x_j + 2A * sum_{j<k} x_j x_k
  = (constant) - A * sum_j x_j + 2A * sum_{j<k} x_j x_k

Constants are dropped (QUBO only cares about variable terms).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

QDict = dict[tuple[int, int], float]  # QUBO coefficient dictionary


# ---------------------------------------------------------------------------
# Constraint penalties
# ---------------------------------------------------------------------------


def one_hot_penalty(
    variable_indices: Sequence[int],
    strength: float,
) -> QDict:
    """Build QUBO penalty terms that enforce sum(x_i for i in group) == 1.

    Derivation (see module docstring):
        H = strength * (1 - sum_j x_j)^2
    Variable part:
        -strength * x_j   (diagonal)
        +2*strength * x_j*x_k  for j < k  (off-diagonal upper triangle)

    Parameters
    ----------
    variable_indices : Sequence[int]
        Flat QUBO variable indices that form the one-hot group.
    strength : float
        Penalty coefficient A.

    Returns
    -------
    QDict
        QUBO coefficient additions for this constraint.

    Complexity
    ----------
    O(K²) where K = len(variable_indices).
    """
    q: QDict = {}
    idxs = list(variable_indices)
    for j in idxs:
        q[(j, j)] = q.get((j, j), 0.0) - strength
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            j, k = idxs[a], idxs[b]
            q[(j, k)] = q.get((j, k), 0.0) + 2.0 * strength
    return q


def route_cost_penalty(
    travel_time: np.ndarray,
    n: int,
    var_idx_fn: "Callable[[int, int], int]",  # noqa: F821
) -> QDict:
    """Build QUBO cost terms for TSP route travel time.

    Encodes:
        H_C = sum_{i,j} T_{ij} * sum_p x_{i,p} * x_{j,(p+1) mod n}

    Parameters
    ----------
    travel_time : np.ndarray
        Shape (n, n) travel-time matrix (full node set: depot + customers).
    n : int
        Total number of nodes (N+1).
    var_idx_fn : callable (node_id, position) -> flat_qubo_index
        Maps a (node, position) pair to the corresponding QUBO variable index.

    Returns
    -------
    QDict

    Complexity
    ----------
    O(N³)
    """
    q: QDict = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            t_ij = float(travel_time[i, j])
            if t_ij == 0.0:
                continue
            for p in range(n):
                p_next = (p + 1) % n
                u = var_idx_fn(i, p)
                v = var_idx_fn(j, p_next)
                if u == v:
                    q[(u, u)] = q.get((u, u), 0.0) + t_ij
                elif u < v:
                    q[(u, v)] = q.get((u, v), 0.0) + t_ij
                else:
                    q[(v, u)] = q.get((v, u), 0.0) + t_ij
    return q


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------


def merge_qdicts(*qdicts: QDict) -> QDict:
    """Merge multiple QDict objects by summing common keys.

    Parameters
    ----------
    *qdicts : QDict

    Returns
    -------
    QDict

    Complexity
    ----------
    O(total entries)
    """
    merged: QDict = {}
    for qd in qdicts:
        for key, val in qd.items():
            merged[key] = merged.get(key, 0.0) + val
    return merged


def qdict_to_matrix(q: QDict, n_vars: int) -> np.ndarray:
    """Convert a QDict to a dense upper-triangular QUBO matrix.

    Parameters
    ----------
    q : QDict
    n_vars : int
        Number of QUBO variables.

    Returns
    -------
    np.ndarray
        Shape (n_vars, n_vars) float64, upper triangular.

    Complexity
    ----------
    O(n_vars²) for allocation + O(|q|) for filling.
    """
    Q = np.zeros((n_vars, n_vars), dtype=np.float64)
    for (i, j), val in q.items():
        if i <= j:
            Q[i, j] += val
        else:
            Q[j, i] += val
    return Q


def evaluate_qubo(Q: np.ndarray, x: np.ndarray) -> float:
    """Evaluate QUBO objective  x^T Q x.

    Parameters
    ----------
    Q : np.ndarray
        Shape (n, n) QUBO matrix (upper triangular).
    x : np.ndarray
        Binary solution vector, shape (n,).

    Returns
    -------
    float
        Objective value.

    Complexity
    ----------
    O(N²)
    """
    return float(x @ Q @ x)
