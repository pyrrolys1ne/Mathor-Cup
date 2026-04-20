"""
src/io/validate_data.py
------------------------
加载实例后的数据校验模块。

主要检查项:
    - 节点属性缺失值
    - 节点编号重复
    - 旅行时间矩阵维度是否为 N+1 乘 N+1
    - 对角线是否为零
    - 旅行时间是否非负
    - node_id 与 demand 整数一致性
    - 时间窗是否满足 e_i 小于等于 l_i
    - 需求量是否非负
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 校验结果结构
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """保存一次校验的结果。

    Attributes
    ----------
    passed : bool
        True if all *critical* checks passed (warnings don't affect this flag).
    errors : list[str]
        Critical issues that must be fixed before solving.
    warnings : list[str]
        Non-critical issues; logged but execution continues.
    stats : dict[str, Any]
        Summary statistics (node counts, matrix shape, etc.).
    """

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        """记录关键错误并标记校验失败。"""
        self.errors.append(msg)
        self.passed = False
        logger.error("Validation ERROR: %s", msg)

    def add_warning(self, msg: str) -> None:
        """记录非关键告警。"""
        self.warnings.append(msg)
        logger.warning("Validation WARNING: %s", msg)

    def summary(self) -> str:
        """返回可读的校验摘要文本。"""
        lines = [
            f"Validation {'PASSED' if self.passed else 'FAILED'}",
            f"  Errors   : {len(self.errors)}",
            f"  Warnings : {len(self.warnings)}",
        ]
        for e in self.errors:
            lines.append(f"  [ERR] {e}")
        for w in self.warnings:
            lines.append(f"  [WRN] {w}")
        lines.append("  Stats:")
        for k, v in self.stats.items():
            lines.append(f"    {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主校验流程
# ---------------------------------------------------------------------------


def validate_instance(
    nodes: pd.DataFrame,
    travel_time: np.ndarray,
    expected_n_customers: int | None = None,
) -> ValidationReport:
    """Validate a loaded problem instance.

    Parameters
    ----------
    nodes : pd.DataFrame
        Node-attribute DataFrame (depot + customers).  Must contain columns:
        ``node_id, x, y, e, l, service_time, demand``.
    travel_time : np.ndarray
        Square travel-time matrix of shape ``(N+1, N+1)``.
    expected_n_customers : int | None
        If provided, verify that ``len(nodes) - 1 == expected_n_customers``.

    Returns
    -------
    ValidationReport
        Detailed report; call ``.summary()`` for a printable overview.
        Raises ``ValueError`` if any *critical* check fails and
        ``RAISE_ON_ERROR`` is True (see :func:`validate_or_raise`).

    Complexity
    ----------
    O(N²) dominated by matrix scan.
    """
    report = ValidationReport()
    n_nodes = len(nodes)
    n_customers = n_nodes - 1

    # 基础统计
    report.stats["n_nodes_total"] = n_nodes
    report.stats["n_customers"] = n_customers
    report.stats["matrix_shape"] = travel_time.shape

    # 1 缺失值
    _check_missing_values(nodes, report)

    # 2 重复节点编号
    _check_duplicate_ids(nodes, report)

    # 3 节点编号范围
    _check_node_id_range(nodes, n_nodes, report)

    # 4 矩阵维度
    _check_matrix_dimensions(travel_time, n_nodes, report)

    # 5 对角线为零
    _check_diagonal_zeros(travel_time, report)

    # 6 旅行时间非负
    _check_non_negative(travel_time, report)

    # 7 时间窗顺序
    _check_time_window_order(nodes, report)

    # 8 需求量非负
    _check_demand_non_negative(nodes, report)

    # 9 客户数与配置一致
    if expected_n_customers is not None:
        _check_customer_count(n_customers, expected_n_customers, report)

    # 10 整数一致性
    _check_integer_columns(nodes, report)

    logger.info(
        "Validation complete. passed=%s, errors=%d, warnings=%d",
        report.passed,
        len(report.errors),
        len(report.warnings),
    )
    return report


def validate_or_raise(
    nodes: pd.DataFrame,
    travel_time: np.ndarray,
    expected_n_customers: int | None = None,
) -> ValidationReport:
    """Like :func:`validate_instance`, but raises ``ValueError`` on failure.

    Parameters
    ----------
    nodes, travel_time, expected_n_customers :
        Same as :func:`validate_instance`.

    Returns
    -------
    ValidationReport

    Raises
    ------
    ValueError
        If any critical check fails (report.passed is False).
    """
    report = validate_instance(nodes, travel_time, expected_n_customers)
    if not report.passed:
        raise ValueError(
            f"Data validation failed with {len(report.errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in report.errors)
        )
    return report


# ---------------------------------------------------------------------------
# 逐项检查
# ---------------------------------------------------------------------------


def _check_missing_values(nodes: pd.DataFrame, report: ValidationReport) -> None:
    """检查节点表中的 NaN 与 None。"""
    nan_counts = nodes.isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if not cols_with_nan.empty:
        for col, count in cols_with_nan.items():
            report.add_error(
                f"Column '{col}' has {count} missing value(s). Fill or impute before solving."
            )
    else:
        report.stats["missing_values"] = 0


def _check_duplicate_ids(nodes: pd.DataFrame, report: ValidationReport) -> None:
    """检查 node_id 是否唯一。"""
    if "node_id" not in nodes.columns:
        report.add_error("Column 'node_id' is absent from node DataFrame.")
        return
    dups = nodes["node_id"][nodes["node_id"].duplicated(keep=False)]
    if not dups.empty:
        report.add_error(f"Duplicate node_id values found: {sorted(dups.unique().tolist())}")
    else:
        report.stats["duplicate_ids"] = 0


def _check_node_id_range(
    nodes: pd.DataFrame, n_nodes: int, report: ValidationReport
) -> None:
    """检查节点编号是否完整覆盖 0 到 N。"""
    if "node_id" not in nodes.columns:
        return
    ids = set(nodes["node_id"].astype(int).tolist())
    expected = set(range(n_nodes))
    extra = ids - expected
    missing = expected - ids
    if extra:
        report.add_error(f"Unexpected node IDs (outside 0..{n_nodes-1}): {sorted(extra)}")
    if missing:
        report.add_error(f"Missing node IDs: {sorted(missing)}")


def _check_matrix_dimensions(
    travel_time: np.ndarray, n_nodes: int, report: ValidationReport
) -> None:
    """检查矩阵为方阵且维度与节点数一致。"""
    r, c = travel_time.shape
    if r != c:
        report.add_error(f"Travel-time matrix is not square: {r}×{c}")
    elif r != n_nodes:
        report.add_error(
            f"Travel-time matrix size ({r}×{c}) does not match node count ({n_nodes})."
        )


def _check_diagonal_zeros(
    travel_time: np.ndarray, report: ValidationReport
) -> None:
    """检查对角线是否全部为零。"""
    diag = np.diag(travel_time)
    nonzero = np.where(diag != 0)[0]
    if nonzero.size > 0:
        report.add_error(
            f"Non-zero diagonal in travel-time matrix at indices: {nonzero.tolist()}"
        )
    else:
        report.stats["diagonal_zeros"] = True


def _check_non_negative(
    travel_time: np.ndarray, report: ValidationReport
) -> None:
    """检查旅行时间是否存在负值。"""
    neg_mask = travel_time < 0
    if neg_mask.any():
        neg_count = int(neg_mask.sum())
        # 给出若干示例
        rows, cols = np.where(neg_mask)
        examples = [(int(r), int(c)) for r, c in zip(rows[:5], cols[:5])]
        report.add_error(
            f"Travel-time matrix has {neg_count} negative value(s). "
            f"Examples (row, col): {examples}"
        )
    else:
        report.stats["min_travel_time"] = float(travel_time.min())
        report.stats["max_travel_time"] = float(travel_time.max())


def _check_time_window_order(nodes: pd.DataFrame, report: ValidationReport) -> None:
    """检查每个节点是否满足 e_i 小于等于 l_i。"""
    if "e" not in nodes.columns or "l" not in nodes.columns:
        return
    bad = nodes[nodes["e"] > nodes["l"]]
    if not bad.empty:
        ids = bad["node_id"].tolist() if "node_id" in bad.columns else bad.index.tolist()
        report.add_error(
            f"Time-window violation (e > l) at node_id(s): {ids}"
        )
    else:
        report.stats["time_window_order"] = "OK"


def _check_demand_non_negative(nodes: pd.DataFrame, report: ValidationReport) -> None:
    """检查需求量是否非负。"""
    if "demand" not in nodes.columns:
        return
    bad = nodes[nodes["demand"] < 0]
    if not bad.empty:
        ids = bad["node_id"].tolist() if "node_id" in bad.columns else bad.index.tolist()
        report.add_error(f"Negative demand at node_id(s): {ids}")
    else:
        report.stats["total_demand"] = float(nodes["demand"].sum())


def _check_customer_count(
    actual: int, expected: int, report: ValidationReport
) -> None:
    """检查客户数量是否与配置一致。"""
    if actual != expected:
        report.add_warning(
            f"Config expected {expected} customers, but data has {actual}. "
            "Solver will use the actual count."
        )
    else:
        report.stats["customer_count_match"] = True


def _check_integer_columns(nodes: pd.DataFrame, report: ValidationReport) -> None:
    """检查 node_id 与 demand 是否为整数值。"""
    for col in ("node_id", "demand"):
        if col not in nodes.columns:
            continue
        vals = nodes[col].dropna()
        non_int = vals[vals != vals.round()]
        if not non_int.empty:
            report.add_warning(
                f"Column '{col}' contains non-integer values "
                f"(e.g., {non_int.iloc[0]}). Will be cast to int."
            )

