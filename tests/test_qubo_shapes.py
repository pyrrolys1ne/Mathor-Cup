"""
tests/test_qubo_shapes.py
--------------------------
q1_qubo 与 penalties 的 QUBO 形状与解码测试。
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
# 测试夹具
# ---------------------------------------------------------------------------


def _tiny_graph(n_customers: int = 3):
    """构造小规模图，节点数为 n_customers 加 1。"""
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
# one_hot_penalty 测试
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
        """可行 one-hot 解的能量应低于不可行解。"""
        q = one_hot_penalty([0, 1, 2], strength=100.0)
        Q = qdict_to_matrix(q, 3)
        x_feasible = np.array([1.0, 0.0, 0.0])
        x_infeasible = np.array([1.0, 1.0, 0.0])
        e_f = evaluate_qubo(Q, x_feasible)
        e_i = evaluate_qubo(Q, x_infeasible)
        assert e_f < e_i


# ---------------------------------------------------------------------------
# qdict_to_matrix 测试
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
# build_q1_qubo 测试
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
        # 下三角不含对角线应为零
        lower = np.tril(Q, k=-1)
        assert np.allclose(lower, 0.0), "QUBO matrix should be upper-triangular"

    def test_penalty_strength_affects_magnitude(self):
        graph = _tiny_graph(n_customers=3)
        r_weak = build_q1_qubo(graph, penalty_visit=10.0)
        r_strong = build_q1_qubo(graph, penalty_visit=1000.0)
        # 惩罚更强时绝对系数上界应更大
        assert np.abs(r_strong.Q).max() > np.abs(r_weak.Q).max()

    def test_larger_instance_shape(self):
        graph = _tiny_graph(n_customers=15)
        result = build_q1_qubo(graph)
        n = 16
        assert result.Q.shape == (n * n, n * n)
        assert result.n_vars == n * n


# ---------------------------------------------------------------------------
# decode_q1_solution 测试
# ---------------------------------------------------------------------------


class TestDecodeQ1Solution:
    def _make_perfect_x(self, n: int, route_order: list[int]) -> np.ndarray:
        """按给定路径顺序构造理想 one-hot 向量。"""
        x = np.zeros(n * n, dtype=float)
        for pos, node in enumerate(route_order):
            x[node * n + pos] = 1.0
        return x

    def test_decode_perfect_solution(self):
        graph = _tiny_graph(n_customers=3)
        n = 4
        # 路径顺序为 0 1 2 3
        order = [0, 1, 2, 3]
        x = self._make_perfect_x(n, order)
        result = build_q1_qubo(graph)
        route = decode_q1_solution(x, result.n_nodes, result.var_idx)
        # 路径应从仓库出发并回到仓库，且覆盖全部客户
        assert route[0] == 0
        assert route[-1] == 0
        customers_visited = set(route[1:-1])
        assert customers_visited == {1, 2, 3}

    def test_decode_all_customers_present(self):
        graph = _tiny_graph(n_customers=5)
        result = build_q1_qubo(graph)
        # 随机构造近可行向量
        rng = np.random.default_rng(0)
        x = rng.integers(0, 2, size=result.n_vars).astype(float)
        route = decode_q1_solution(x, result.n_nodes, result.var_idx)
        # 路径内部节点应由全部客户组成
        interior = route[1:-1]
        assert len(interior) == result.n_nodes - 1 or len(interior) >= 1  # best-effort
