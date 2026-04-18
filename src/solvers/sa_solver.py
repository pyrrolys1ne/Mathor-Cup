"""
src/solvers/sa_solver.py
-------------------------
Simulated Annealing (SA) solver for TSP/VRP.

This solver works directly on the *permutation* space (not QUBO), which
is more efficient for larger instances.  It accepts the same interface
as the Kaiwu QUBO solver so backends are interchangeable.

Two modes:
  1. ``solve_qubo``  — accepts a QUBO matrix Q and binary x; SA on spin space.
  2. ``solve_route`` — SA on permutation space (recommended for N > 10).

The permutation-space SA is the primary engine for Q1–Q4.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SAConfig:
    """Hyper-parameters for simulated annealing.

    Attributes
    ----------
    initial_temp : float
        Starting temperature T_0.
    cooling_rate : float
        Geometric cooling factor r (T_{k+1} = r * T_k).
    min_temp : float
        Stop when temperature falls below this value.
    n_iter_per_temp : int
        Number of candidate moves evaluated per temperature level.
    seed : int
        Random seed for reproducibility.
    """

    initial_temp: float = 1000.0
    cooling_rate: float = 0.995
    min_temp: float = 1e-4
    n_iter_per_temp: int = 200
    seed: int = 42


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SAResult:
    """Output of a simulated annealing run.

    Attributes
    ----------
    best_solution : list[int]
        Best route (permutation of customer IDs, no depot bookends).
    best_cost : float
        Objective value of the best route.
    history : list[float]
        Best-cost values recorded each time a new best is found.
    elapsed_sec : float
        Wall-clock time of the run.
    n_iterations : int
        Total number of iterations executed.
    """

    best_solution: list[int]
    best_cost: float
    history: list[float] = field(default_factory=list)
    elapsed_sec: float = 0.0
    n_iterations: int = 0


# ---------------------------------------------------------------------------
# Permutation-space SA (primary)
# ---------------------------------------------------------------------------


def solve_route_sa(
    customer_ids: list[int],
    cost_fn: Callable[[list[int]], float],
    cfg: SAConfig | None = None,
) -> SAResult:
    """Run simulated annealing on permutation (route) space.

    The state is a permutation of *customer_ids* (depot not included).
    The ``cost_fn`` evaluates a permutation and returns a scalar cost
    (lower is better).

    Move operator: random 2-opt swap or or-opt (single customer reinsertion).

    Parameters
    ----------
    customer_ids : list[int]
        Customer node IDs to permute (depot excluded).
    cost_fn : Callable[[list[int]], float]
        Maps a customer permutation to objective value.  Should include
        depot legs in its internal computation.
    cfg : SAConfig | None
        SA hyper-parameters; defaults applied if None.

    Returns
    -------
    SAResult

    Complexity
    ----------
    O(T * n_iter * N) where T = log(T_min/T_0) / log(r).
    """
    if cfg is None:
        cfg = SAConfig()

    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    # Initialise with a random permutation
    current = list(customer_ids)
    rng.shuffle(current)
    current_cost = cost_fn(current)

    best = list(current)
    best_cost = current_cost
    history = [best_cost]

    temp = cfg.initial_temp
    total_iters = 0
    t_start = time.perf_counter()

    while temp > cfg.min_temp:
        for _ in range(cfg.n_iter_per_temp):
            total_iters += 1
            candidate = _random_move(current, rng)
            candidate_cost = cost_fn(candidate)
            delta = candidate_cost - current_cost

            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current = candidate
                current_cost = candidate_cost
                if current_cost < best_cost:
                    best = list(current)
                    best_cost = current_cost
                    history.append(best_cost)

        temp *= cfg.cooling_rate

    elapsed = time.perf_counter() - t_start
    logger.info(
        "SA finished: best_cost=%.4f, iters=%d, time=%.2fs",
        best_cost,
        total_iters,
        elapsed,
    )
    return SAResult(
        best_solution=best,
        best_cost=best_cost,
        history=history,
        elapsed_sec=elapsed,
        n_iterations=total_iters,
    )


# ---------------------------------------------------------------------------
# QUBO-space SA (for compatibility with QUBO interface)
# ---------------------------------------------------------------------------


def solve_qubo_sa(
    Q: np.ndarray,
    cfg: SAConfig | None = None,
) -> np.ndarray:
    """Run SA directly on a QUBO matrix to find binary solution vector.

    Uses single bit-flip moves.  For large QUBO matrices this is much
    slower than permutation-space SA; prefer ``solve_route_sa`` when possible.

    Parameters
    ----------
    Q : np.ndarray
        Shape (n, n) upper-triangular QUBO matrix.
    cfg : SAConfig | None

    Returns
    -------
    np.ndarray
        Best binary solution vector found, shape (n,).

    Complexity
    ----------
    O(T * n_iter * N) where each energy evaluation is O(N).
    """
    if cfg is None:
        cfg = SAConfig()

    rng = np.random.default_rng(cfg.seed)
    n = Q.shape[0]

    # Precompute symmetric Q for efficient energy computation
    Q_sym = Q + Q.T - np.diag(np.diag(Q))

    # Initialise randomly
    x = rng.integers(0, 2, size=n).astype(np.float64)
    energy = float(x @ Q @ x)

    best_x = x.copy()
    best_energy = energy

    temp = cfg.initial_temp
    total_iters = 0
    t_start = time.perf_counter()

    while temp > cfg.min_temp:
        for _ in range(cfg.n_iter_per_temp):
            total_iters += 1
            bit = int(rng.integers(0, n))
            # Energy change for flipping bit i:  ΔE = (1-2*x_i)*(Q_sym @ x)[i]
            delta = float((1 - 2 * x[bit]) * (Q_sym @ x)[bit])
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                x[bit] = 1 - x[bit]
                energy += delta
                if energy < best_energy:
                    best_x = x.copy()
                    best_energy = energy

        temp *= cfg.cooling_rate

    elapsed = time.perf_counter() - t_start
    logger.info(
        "QUBO SA finished: best_energy=%.4f, iters=%d, time=%.2fs",
        best_energy,
        total_iters,
        elapsed,
    )
    return best_x


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _random_move(route: list[int], rng: random.Random) -> list[int]:
    """Apply a random 2-opt or or-opt move to a route.

    Parameters
    ----------
    route : list[int]
    rng : random.Random

    Returns
    -------
    list[int]
        Modified route (new list; original unmodified).

    Complexity
    ----------
    O(N)
    """
    n = len(route)
    if n < 2:
        return list(route)

    move_type = rng.random()
    candidate = list(route)

    if move_type < 0.5 or n < 3:
        # 2-opt: reverse a sub-segment
        i, j = sorted(rng.sample(range(n), 2))
        candidate[i : j + 1] = candidate[i : j + 1][::-1]
    else:
        # or-opt: remove one customer and reinsert at different position
        i = rng.randrange(n)
        node = candidate.pop(i)
        j = rng.randrange(len(candidate) + 1)
        candidate.insert(j, node)

    return candidate
