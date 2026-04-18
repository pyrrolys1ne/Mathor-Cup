"""
tests/test_time_window_penalty.py
-----------------------------------
Tests for src/core/time_window.py penalty calculations and route timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.graph_model import build_graph
from src.core.time_window import (
    ArrivalRecord,
    compute_penalty,
    compute_penalty_array,
    simulate_route_timing,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _simple_graph():
    """Build a tiny 4-node graph: depot(0) + 3 customers."""
    nodes = pd.DataFrame(
        [
            {"node_id": 0, "x": 0.0, "y": 0.0, "e": 0.0, "l": 1000.0, "service_time": 0.0, "demand": 0.0},
            {"node_id": 1, "x": 1.0, "y": 0.0, "e": 5.0,  "l": 15.0,  "service_time": 2.0, "demand": 10.0},
            {"node_id": 2, "x": 2.0, "y": 0.0, "e": 20.0, "l": 30.0,  "service_time": 3.0, "demand": 10.0},
            {"node_id": 3, "x": 3.0, "y": 0.0, "e": 0.0,  "l": 1000.0,"service_time": 1.0, "demand": 5.0},
        ]
    )
    tt = np.array(
        [
            [0, 5, 10, 15],
            [5, 0,  5, 10],
            [10, 5, 0,  5],
            [15, 10, 5, 0],
        ],
        dtype=float,
    )
    return build_graph(nodes, tt)


# ---------------------------------------------------------------------------
# compute_penalty unit tests
# ---------------------------------------------------------------------------


class TestComputePenalty:
    def test_on_time_no_penalty(self):
        ev, lv, pen = compute_penalty(10.0, 5.0, 15.0)
        assert ev == 0.0
        assert lv == 0.0
        assert pen == 0.0

    def test_early_arrival(self):
        # Arrival at 3, window [5, 15]
        ev, lv, pen = compute_penalty(3.0, 5.0, 15.0, alpha=10, beta=20)
        assert ev == pytest.approx(2.0)
        assert lv == 0.0
        assert pen == pytest.approx(10 * 4.0)  # 10 * (5-3)^2

    def test_late_arrival(self):
        # Arrival at 20, window [5, 15]
        ev, lv, pen = compute_penalty(20.0, 5.0, 15.0, alpha=10, beta=20)
        assert ev == 0.0
        assert lv == pytest.approx(5.0)
        assert pen == pytest.approx(20 * 25.0)  # 20 * (20-15)^2

    def test_exact_boundary_no_penalty(self):
        ev, lv, pen = compute_penalty(5.0, 5.0, 15.0)
        assert ev == 0.0
        assert pen == 0.0
        ev, lv, pen = compute_penalty(15.0, 5.0, 15.0)
        assert lv == 0.0
        assert pen == 0.0

    def test_custom_coefficients(self):
        ev, lv, pen = compute_penalty(0.0, 2.0, 10.0, alpha=1.0, beta=1.0)
        assert ev == pytest.approx(2.0)
        assert pen == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# compute_penalty_array unit tests
# ---------------------------------------------------------------------------


class TestComputePenaltyArray:
    def test_vectorised_matches_scalar(self):
        arrivals = np.array([3.0, 10.0, 20.0])
        earliests = np.array([5.0, 5.0, 5.0])
        latests = np.array([15.0, 15.0, 15.0])
        result = compute_penalty_array(arrivals, earliests, latests)

        for i in range(3):
            _, _, expected = compute_penalty(arrivals[i], earliests[i], latests[i])
            assert result[i] == pytest.approx(expected)

    def test_all_zero_when_in_window(self):
        arrivals = np.array([8.0, 10.0, 12.0])
        earliests = np.array([5.0, 5.0, 5.0])
        latests = np.array([15.0, 15.0, 15.0])
        result = compute_penalty_array(arrivals, earliests, latests)
        assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# simulate_route_timing tests
# ---------------------------------------------------------------------------


class TestSimulateRouteTiming:
    def test_simple_route_travel_time(self):
        graph = _simple_graph()
        # Route: depot(0) → 1 → 2 → 3 → depot(0)
        # Travel: 0→1=5, 1→2=5, 2→3=5, 3→0=15  (total = 30)
        # Service: 2+3+1 = 6
        result = simulate_route_timing([0, 1, 2, 3, 0], graph)
        assert result.total_travel_time == pytest.approx(30.0)
        assert result.total_service_time == pytest.approx(6.0)

    def test_arrival_times_correct(self):
        graph = _simple_graph()
        result = simulate_route_timing([0, 1, 2, 3, 0], graph)
        arrivals = {r.node_id: r.arrival_time for r in result.records}
        # Node 1 arrives at t=5 (0→1 travel time)
        assert arrivals[1] == pytest.approx(5.0)
        # Node 2 arrives at t=5+2(service at 1)+5=12
        assert arrivals[2] == pytest.approx(12.0)
        # Node 3 arrives at t=12+3(service at 2)+5=20
        assert arrivals[3] == pytest.approx(20.0)

    def test_time_window_penalties(self):
        graph = _simple_graph()
        # Node 1: window [5, 15], arrives at 5 → no penalty
        # Node 2: window [20, 30], arrives at 12 → early by 8 → penalty=10*64=640
        # Node 3: window [0, 1000], arrives at 20 → no penalty
        result = simulate_route_timing([0, 1, 2, 3, 0], graph, alpha=10, beta=20)
        records = {r.node_id: r for r in result.records}
        assert records[1].penalty == pytest.approx(0.0)
        assert records[2].early_violation == pytest.approx(8.0)
        assert records[2].penalty == pytest.approx(10 * 64.0)
        assert records[3].penalty == pytest.approx(0.0)

    def test_depot_added_automatically(self):
        graph = _simple_graph()
        # Route without explicit depot bookends
        result = simulate_route_timing([1, 2, 3], graph)
        assert result.route[0] == 0
        assert result.route[-1] == 0

    def test_empty_route(self):
        graph = _simple_graph()
        result = simulate_route_timing([], graph)
        assert result.total_travel_time == pytest.approx(0.0)
        assert len(result.records) == 0

    def test_total_penalty_sum(self):
        graph = _simple_graph()
        result = simulate_route_timing([0, 1, 2, 3, 0], graph)
        expected = sum(r.penalty for r in result.records)
        assert result.total_penalty == pytest.approx(expected)
