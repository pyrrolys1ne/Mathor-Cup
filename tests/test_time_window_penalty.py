"""
tests/test_time_window_penalty.py
---------------------------------
time_window 惩罚计算与路径时序测试。
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
# 测试夹具
# ---------------------------------------------------------------------------


def _simple_graph():
    """构造四节点小图，含仓库与三个客户。"""
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
# compute_penalty 单元测试
# ---------------------------------------------------------------------------


class TestComputePenalty:
    def test_on_time_no_penalty(self):
        ev, lv, pen = compute_penalty(10.0, 5.0, 15.0)
        assert ev == 0.0
        assert lv == 0.0
        assert pen == 0.0

    def test_early_arrival(self):
        # 到达时刻 3，时间窗为 5 到 15
        ev, lv, pen = compute_penalty(3.0, 5.0, 15.0, alpha=10, beta=20)
        assert ev == pytest.approx(2.0)
        assert lv == 0.0
        assert pen == pytest.approx(10 * 4.0)  # 10 乘 5 减 3 的平方

    def test_late_arrival(self):
        # 到达时刻 20，时间窗为 5 到 15
        ev, lv, pen = compute_penalty(20.0, 5.0, 15.0, alpha=10, beta=20)
        assert ev == 0.0
        assert lv == pytest.approx(5.0)
        assert pen == pytest.approx(20 * 25.0)  # 20 乘 20 减 15 的平方

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
# compute_penalty_array 单元测试
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
# simulate_route_timing 测试
# ---------------------------------------------------------------------------


class TestSimulateRouteTiming:
    def test_simple_route_travel_time(self):
        graph = _simple_graph()
        # 路径为 0 到 1 到 2 到 3 到 0
        # 旅行时间总和为 30
        # 服务时间总和为 6
        result = simulate_route_timing([0, 1, 2, 3, 0], graph)
        assert result.total_travel_time == pytest.approx(30.0)
        assert result.total_service_time == pytest.approx(6.0)

    def test_arrival_times_correct(self):
        graph = _simple_graph()
        result = simulate_route_timing([0, 1, 2, 3, 0], graph)
        arrivals = {r.node_id: r.arrival_time for r in result.records}
        # 节点 1 到达时刻为 5
        assert arrivals[1] == pytest.approx(5.0)
        # 节点 2 到达时刻为 12
        assert arrivals[2] == pytest.approx(12.0)
        # 节点 3 到达时刻为 20
        assert arrivals[3] == pytest.approx(20.0)

    def test_time_window_penalties(self):
        graph = _simple_graph()
        # 节点 1 无惩罚，节点 2 早到 8 产生惩罚，节点 3 无惩罚
        result = simulate_route_timing([0, 1, 2, 3, 0], graph, alpha=10, beta=20)
        records = {r.node_id: r for r in result.records}
        assert records[1].penalty == pytest.approx(0.0)
        assert records[2].early_violation == pytest.approx(8.0)
        assert records[2].penalty == pytest.approx(10 * 64.0)
        assert records[3].penalty == pytest.approx(0.0)

    def test_depot_added_automatically(self):
        graph = _simple_graph()
        # 路径未显式包含仓库时应自动补齐
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
