"""
src/io/validate_data.py
------------------------
Robust data validation for the loaded problem instance.

Checks performed:
  - Missing values in node attributes
  - Duplicate node IDs
  - Travel-time matrix dimensions (must be square, N+1 × N+1)
  - Diagonal elements are zero
  - All travel times are non-negative
  - Integer consistency (node IDs, demands)
  - Time-window logical ordering (e_i <= l_i)
  - Demand non-negativity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation result container
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Holds results of a single validation run.

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
        """Record a critical error and mark report as failed."""
        self.errors.append(msg)
        self.passed = False
        logger.error("Validation ERROR: %s", msg)

    def add_warning(self, msg: str) -> None:
        """Record a non-critical warning."""
        self.warnings.append(msg)
        logger.warning("Validation WARNING: %s", msg)

    def summary(self) -> str:
        """Return a human-readable summary string."""
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
# Main validator
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

    # --- Basic stats -------------------------------------------------------
    report.stats["n_nodes_total"] = n_nodes
    report.stats["n_customers"] = n_customers
    report.stats["matrix_shape"] = travel_time.shape

    # 1. Missing values ------------------------------------------------------
    _check_missing_values(nodes, report)

    # 2. Duplicate node IDs --------------------------------------------------
    _check_duplicate_ids(nodes, report)

    # 3. Correct node ID range (0..N) ----------------------------------------
    _check_node_id_range(nodes, n_nodes, report)

    # 4. Matrix dimensions ---------------------------------------------------
    _check_matrix_dimensions(travel_time, n_nodes, report)

    # 5. Diagonal zeros ------------------------------------------------------
    _check_diagonal_zeros(travel_time, report)

    # 6. Non-negativity of travel times --------------------------------------
    _check_non_negative(travel_time, report)

    # 7. Time-window ordering (e_i <= l_i) -----------------------------------
    _check_time_window_order(nodes, report)

    # 8. Demand non-negativity -----------------------------------------------
    _check_demand_non_negative(nodes, report)

    # 9. Expected customer count ---------------------------------------------
    if expected_n_customers is not None:
        _check_customer_count(n_customers, expected_n_customers, report)

    # 10. Integer consistency for node_id and demand -------------------------
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
# Individual checks
# ---------------------------------------------------------------------------


def _check_missing_values(nodes: pd.DataFrame, report: ValidationReport) -> None:
    """Detect NaN / None values in the node DataFrame."""
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
    """Ensure node_id values are unique."""
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
    """Verify node IDs span exactly {0, 1, …, N}."""
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
    """Assert matrix is square and matches node count."""
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
    """Verify all diagonal entries are zero (no self-travel time)."""
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
    """Ensure no travel time is negative."""
    neg_mask = travel_time < 0
    if neg_mask.any():
        neg_count = int(neg_mask.sum())
        # Provide a few examples
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
    """Verify e_i <= l_i for every node that has time windows."""
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
    """Ensure demand values are non-negative."""
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
    """Verify the number of customers matches the config expectation."""
    if actual != expected:
        report.add_warning(
            f"Config expected {expected} customers, but data has {actual}. "
            "Solver will use the actual count."
        )
    else:
        report.stats["customer_count_match"] = True


def _check_integer_columns(nodes: pd.DataFrame, report: ValidationReport) -> None:
    """Check that node_id and demand are integer-valued (no fractional parts)."""
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
