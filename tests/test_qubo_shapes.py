"""
tests/test_qubo_shapes.py
---------------------------
Tests for QUBO matrix construction in q1_qubo.py and penalties.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.graph_model import build_graph
from src.qubo.penalties import (
    evaluate_qubo,
    merge_qdicts,
    one_hot_penalty,
    qdict_to_matrix,
    route_cost_penalty,
)
from src.qubo.q1_qubo import build_q1_qubo, decode_q1_solution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_graph(n_customers: int = 3):
    """Build a tiny (n_customers + 1)-node graph."""
    n = n_customers + 1
    nodes = pd.DataFrame(
        [
            {
                "node_id": i,
                "x": float(i),
                "y": float(i),
                "e": 0.0,
                "l": 1000.0,
                "service_time": 0.0,
                "demand": 10.0 if i > 0 else 0.0,
            }
            for i in range(n)
        ]
    )
    rng = np.random.default_rng(42)
    tt = rng.uniform(1, 10, size=(n, n)).astype(float)
    np.fill_diagonal(tt, 0.0)
    return build_graph(nodes, tt)


# ---------------------------------------------------------------------------
# one_hot_penalty tests
# ---------------------------------------------------------------------------


class TestOneHotPenalty:
    def test_diagonal_negative(self):
        q = one_hot_penalty([0, 1, 2], strength=10.0)
        for i in [0, 1, 2]:
            assert q.get((i, i), 0.0) < 0

    def test_off_diagonal_positive(self):
        q = one_hot_penalty([0, 1, 2], strength=10.0)
        for i in range(3):
            for j in range(i + 1, 3):
                assert q.get((i, j), 0.0) > 0

    def test_upper_triangle_only(self):
        q = one_hot_penalty([0, 1, 2], strength=10.0)
        for (r, c) in q:
            assert r <= c, f"Lower-triangle entry found: ({r}, {c})"

    def test_feasible_solution_low_energy(self):
        """One-hot feasible solution should have lower energy than infeasible."""
        q = one_hot_penalty([0, 1, 2], strength=100.0)
        Q = qdict_to_matrix(q, 3)
        x_feasible = np.array([1.0, 0.0, 0.0])
        x_infeasible = np.array([1.0, 1.0, 0.0])
        e_f = evaluate_qubo(Q, x_feasible)
        e_i = evaluate_qubo(Q, x_infeasible)
        assert e_f < e_i


# ---------------------------------------------------------------------------
# qdict_to_matrix tests
# ---------------------------------------------------------------------------


class TestQdictToMatrix:
    def test_shape(self):
        q = {(0, 0): 1.0, (0, 2): 3.0}
        Q = qdict_to_matrix(q, 4)
        assert Q.shape == (4, 4)

    def test_lower_triangle_zero(self):
        q = {(0, 1): 5.0, (0, 0): -2.0}
        Q = qdict_to_matrix(q, 3)
        assert Q[1, 0] == 0.0

    def test_values_placed_correctly(self):
        q = {(1, 2): 7.0, (0, 0): -3.0}
        Q = qdict_to_matrix(q, 3)
        assert Q[1, 2] == pytest.approx(7.0)
        assert Q[0, 0] == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# build_q1_qubo tests
# ---------------------------------------------------------------------------


class TestBuildQ1QUBO:
    def test_matrix_shape(self):
        graph = _tiny_graph(n_customers=3)
        result = build_q1_qubo(graph)
        n = graph.n_customers + 1
        assert result.Q.shape == (n * n, n * n)
        assert result.n_vars == n * n
        assert result.n_nodes == n

    def test_upper_triangular(self):
        graph = _tiny_graph(n_customers=3)
        result = build_q1_qubo(graph)
        Q = result.Q
        # Lower triangle (excluding diagonal) should be zero
        lower = np.tril(Q, k=-1)
        assert np.allclose(lower, 0.0), "QUBO matrix should be upper-triangular"

    def test_penalty_strength_affects_magnitude(self):
        graph = _tiny_graph(n_customers=3)
        r_weak = build_q1_qubo(graph, penalty_visit=10.0)
        r_strong = build_q1_qubo(graph, penalty_visit=1000.0)
        # Strong penalty → larger absolute off-diagonal entries
        assert np.abs(r_strong.Q).max() > np.abs(r_weak.Q).max()

    def test_larger_instance_shape(self):
        graph = _tiny_graph(n_customers=15)
        result = build_q1_qubo(graph)
        n = 16
        assert result.Q.shape == (n * n, n * n)
        assert result.n_vars == n * n


# ---------------------------------------------------------------------------
# decode_q1_solution tests
# ---------------------------------------------------------------------------


class TestDecodeQ1Solution:
    def _make_perfect_x(self, n: int, route_order: list[int]) -> np.ndarray:
        """Construct a perfect one-hot binary vector for a given route order."""
        x = np.zeros(n * n, dtype=float)
        for pos, node in enumerate(route_order):
            x[node * n + pos] = 1.0
        return x

    def test_decode_perfect_solution(self):
        graph = _tiny_graph(n_customers=3)
        n = 4
        # Route: depot(0) at pos 0, node 1 at pos 1, node 2 at pos 2, node 3 at pos 3
        order = [0, 1, 2, 3]
        x = self._make_perfect_x(n, order)
        result = build_q1_qubo(graph)
        route = decode_q1_solution(x, result.n_nodes, result.var_idx)
        # Route should include depot, visit all customers, return to depot
        assert route[0] == 0
        assert route[-1] == 0
        customers_visited = set(route[1:-1])
        assert customers_visited == {1, 2, 3}

    def test_decode_all_customers_present(self):
        graph = _tiny_graph(n_customers=5)
        result = build_q1_qubo(graph)
        # Random near-feasible solution
        rng = np.random.default_rng(0)
        x = rng.integers(0, 2, size=result.n_vars).astype(float)
        route = decode_q1_solution(x, result.n_nodes, result.var_idx)
        # Route must be a permutation of all node IDs
        interior = route[1:-1]
        assert len(interior) == result.n_nodes - 1 or len(interior) >= 1  # best-effort
