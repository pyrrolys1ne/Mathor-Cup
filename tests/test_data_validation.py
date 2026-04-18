"""
tests/test_data_validation.py
-------------------------------
Tests for src/io/validate_data.py and src/io/load_excel.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.io.validate_data import (
    ValidationReport,
    validate_instance,
    validate_or_raise,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_nodes(n_customers: int = 5) -> pd.DataFrame:
    """Create a minimal valid nodes DataFrame with n_customers + 1 rows."""
    rows = []
    n = n_customers + 1
    for i in range(n):
        rows.append(
            {
                "node_id": i,
                "x": float(i),
                "y": float(i),
                "e": 0.0,
                "l": 100.0,
                "service_time": 5.0,
                "demand": 10.0 if i > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _make_travel_time(n: int) -> np.ndarray:
    """Create a valid n×n travel-time matrix."""
    rng = np.random.default_rng(0)
    tt = rng.uniform(1, 20, size=(n, n)).astype(float)
    np.fill_diagonal(tt, 0.0)
    return tt


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestValidateInstanceHappy:
    def test_valid_instance_passes(self):
        nodes = _make_nodes(5)
        tt = _make_travel_time(6)
        report = validate_instance(nodes, tt, expected_n_customers=5)
        assert report.passed is True
        assert len(report.errors) == 0

    def test_stats_populated(self):
        nodes = _make_nodes(3)
        tt = _make_travel_time(4)
        report = validate_instance(nodes, tt)
        assert "n_nodes_total" in report.stats
        assert report.stats["n_nodes_total"] == 4
        assert report.stats["n_customers"] == 3

    def test_validate_or_raise_no_exception(self):
        nodes = _make_nodes(5)
        tt = _make_travel_time(6)
        report = validate_or_raise(nodes, tt)
        assert report.passed


# ---------------------------------------------------------------------------
# Error-detection tests
# ---------------------------------------------------------------------------


class TestValidateInstanceErrors:
    def test_missing_values(self):
        nodes = _make_nodes(3)
        nodes.loc[1, "x"] = np.nan
        tt = _make_travel_time(4)
        report = validate_instance(nodes, tt)
        assert not report.passed
        assert any("missing" in e.lower() for e in report.errors)

    def test_duplicate_ids(self):
        nodes = _make_nodes(3)
        nodes.loc[2, "node_id"] = 1  # duplicate
        tt = _make_travel_time(4)
        report = validate_instance(nodes, tt)
        assert not report.passed
        assert any("duplicate" in e.lower() for e in report.errors)

    def test_non_square_matrix(self):
        nodes = _make_nodes(3)
        tt = np.zeros((3, 4))  # non-square
        report = validate_instance(nodes, tt)
        assert not report.passed
        assert any("square" in e.lower() for e in report.errors)

    def test_wrong_matrix_size(self):
        nodes = _make_nodes(5)
        tt = _make_travel_time(4)  # wrong size (should be 6)
        report = validate_instance(nodes, tt)
        assert not report.passed
        assert any("size" in e.lower() or "match" in e.lower() for e in report.errors)

    def test_non_zero_diagonal(self):
        nodes = _make_nodes(3)
        tt = _make_travel_time(4)
        tt[1, 1] = 5.0
        report = validate_instance(nodes, tt)
        assert not report.passed
        assert any("diagonal" in e.lower() for e in report.errors)

    def test_negative_travel_time(self):
        nodes = _make_nodes(3)
        tt = _make_travel_time(4)
        tt[0, 2] = -1.0
        report = validate_instance(nodes, tt)
        assert not report.passed
        assert any("negative" in e.lower() for e in report.errors)

    def test_time_window_order(self):
        nodes = _make_nodes(3)
        nodes.loc[1, "e"] = 50.0
        nodes.loc[1, "l"] = 20.0  # e > l → violation
        tt = _make_travel_time(4)
        report = validate_instance(nodes, tt)
        assert not report.passed
        assert any("time-window" in e.lower() or "e > l" in e.lower() for e in report.errors)

    def test_negative_demand(self):
        nodes = _make_nodes(3)
        nodes.loc[2, "demand"] = -5.0
        tt = _make_travel_time(4)
        report = validate_instance(nodes, tt)
        assert not report.passed
        assert any("demand" in e.lower() or "negative" in e.lower() for e in report.errors)

    def test_validate_or_raise_raises(self):
        nodes = _make_nodes(3)
        nodes.loc[1, "x"] = np.nan
        tt = _make_travel_time(4)
        with pytest.raises(ValueError, match="validation failed"):
            validate_or_raise(nodes, tt)


# ---------------------------------------------------------------------------
# Warning tests
# ---------------------------------------------------------------------------


class TestValidateInstanceWarnings:
    def test_customer_count_mismatch_is_warning(self):
        nodes = _make_nodes(5)
        tt = _make_travel_time(6)
        # Expected 3 but actual is 5 → warning, not error
        report = validate_instance(nodes, tt, expected_n_customers=3)
        assert report.passed  # only a warning
        assert len(report.warnings) > 0

    def test_summary_string(self):
        nodes = _make_nodes(5)
        tt = _make_travel_time(6)
        report = validate_instance(nodes, tt)
        summary = report.summary()
        assert "PASSED" in summary


# ---------------------------------------------------------------------------
# ValidationReport unit tests
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_add_error_sets_passed_false(self):
        report = ValidationReport()
        assert report.passed is True
        report.add_error("something wrong")
        assert report.passed is False
        assert len(report.errors) == 1

    def test_add_warning_does_not_fail(self):
        report = ValidationReport()
        report.add_warning("something suspicious")
        assert report.passed is True
        assert len(report.warnings) == 1
