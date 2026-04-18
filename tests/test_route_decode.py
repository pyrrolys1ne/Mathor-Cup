"""
tests/test_route_decode.py
----------------------------
Tests for src/algorithms/route_decode.py, local_search.py, and
sa_solver.py route decoding/optimisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.algorithms.local_search import or_opt, two_opt
from src.algorithms.route_decode import nearest_neighbour_route
from src.core.graph_model import build_graph
from src.solvers.sa_solver import SAConfig, SAResult, solve_route_sa


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _small_graph(n_customers: int = 5, seed: int = 0):
    """Build a small random graph."""
    rng = np.random.default_rng(seed)
    n = n_customers + 1
    nodes = pd.DataFrame(
        [
            {
                "node_id": i,
                "x": float(rng.uniform(0, 20)),
                "y": float(rng.uniform(0, 20)),
                "e": 0.0,
                "l": 1000.0,
                "service_time": 0.0,
                "demand": 10.0 if i > 0 else 0.0,
            }
            for i in range(n)
        ]
    )
    tt = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = nodes.loc[i, "x"] - nodes.loc[j, "x"]
                dy = nodes.loc[i, "y"] - nodes.loc[j, "y"]
                tt[i, j] = float(np.sqrt(dx**2 + dy**2))
    return build_graph(nodes, tt)


# ---------------------------------------------------------------------------
# nearest_neighbour_route tests
# ---------------------------------------------------------------------------


class TestNearestNeighbourRoute:
    def test_visits_all_customers(self):
        graph = _small_graph(n_customers=5)
        route = nearest_neighbour_route(graph)
        customers_in_route = set(route[1:-1])
        assert customers_in_route == set(graph.customer_ids)

    def test_starts_and_ends_at_depot(self):
        graph = _small_graph(n_customers=5)
        route = nearest_neighbour_route(graph)
        assert route[0] == 0
        assert route[-1] == 0

    def test_route_length(self):
        graph = _small_graph(n_customers=5)
        route = nearest_neighbour_route(graph)
        assert len(route) == graph.n_customers + 2  # depot + customers + depot

    def test_deterministic_with_same_seed(self):
        graph = _small_graph(n_customers=5, seed=42)
        r1 = nearest_neighbour_route(graph, seed=0)
        r2 = nearest_neighbour_route(graph, seed=0)
        assert r1 == r2


# ---------------------------------------------------------------------------
# two_opt tests
# ---------------------------------------------------------------------------


class TestTwoOpt:
    def test_non_worse_than_initial(self):
        graph = _small_graph(n_customers=6)
        initial = graph.customer_ids[:]
        initial_cost = graph.route_travel_time(initial)
        improved = two_opt(initial, graph, n_iter=50)
        improved_cost = graph.route_travel_time(improved)
        assert improved_cost <= initial_cost + 1e-6

    def test_visits_all_customers(self):
        graph = _small_graph(n_customers=6)
        initial = graph.customer_ids[:]
        improved = two_opt(initial, graph)
        assert set(improved) == set(graph.customer_ids)

    def test_single_customer_unchanged(self):
        graph = _small_graph(n_customers=1)
        result = two_opt([1], graph)
        assert result == [1]


# ---------------------------------------------------------------------------
# or_opt tests
# ---------------------------------------------------------------------------


class TestOrOpt:
    def test_non_worse_than_initial(self):
        graph = _small_graph(n_customers=6)
        initial = graph.customer_ids[:]
        initial_cost = graph.route_travel_time(initial)
        improved = or_opt(initial, graph, n_iter=50)
        improved_cost = graph.route_travel_time(improved)
        assert improved_cost <= initial_cost + 1e-6

    def test_visits_all_customers(self):
        graph = _small_graph(n_customers=6)
        initial = graph.customer_ids[:]
        improved = or_opt(initial, graph)
        assert set(improved) == set(graph.customer_ids)


# ---------------------------------------------------------------------------
# solve_route_sa tests
# ---------------------------------------------------------------------------


class TestSolveRouteSA:
    def _cost_fn(self, graph):
        def fn(perm: list[int]) -> float:
            return graph.route_travel_time(perm)
        return fn

    def test_returns_sa_result(self):
        graph = _small_graph(n_customers=4)
        cfg = SAConfig(initial_temp=100, cooling_rate=0.9, min_temp=1.0, n_iter_per_temp=5, seed=42)
        result = solve_route_sa(graph.customer_ids, self._cost_fn(graph), cfg)
        assert isinstance(result, SAResult)

    def test_visits_all_customers(self):
        graph = _small_graph(n_customers=4)
        cfg = SAConfig(initial_temp=100, cooling_rate=0.9, min_temp=1.0, n_iter_per_temp=5, seed=42)
        result = solve_route_sa(graph.customer_ids, self._cost_fn(graph), cfg)
        assert set(result.best_solution) == set(graph.customer_ids)

    def test_reproducible_with_seed(self):
        graph = _small_graph(n_customers=4)
        cfg = SAConfig(initial_temp=100, cooling_rate=0.9, min_temp=1.0, n_iter_per_temp=10, seed=7)
        r1 = solve_route_sa(graph.customer_ids, self._cost_fn(graph), cfg)
        r2 = solve_route_sa(graph.customer_ids, self._cost_fn(graph), cfg)
        assert r1.best_solution == r2.best_solution
        assert r1.best_cost == pytest.approx(r2.best_cost)

    def test_improves_over_iterations(self):
        graph = _small_graph(n_customers=8, seed=99)
        cfg_few = SAConfig(initial_temp=500, cooling_rate=0.95, min_temp=1.0, n_iter_per_temp=5, seed=42)
        cfg_many = SAConfig(initial_temp=500, cooling_rate=0.95, min_temp=0.1, n_iter_per_temp=100, seed=42)
        r_few = solve_route_sa(graph.customer_ids, self._cost_fn(graph), cfg_few)
        r_many = solve_route_sa(graph.customer_ids, self._cost_fn(graph), cfg_many)
        # More iterations should yield same or better cost
        assert r_many.best_cost <= r_few.best_cost + 1e-6
