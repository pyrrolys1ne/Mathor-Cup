"""
src/io/load_excel.py
---------------------
Load and pre-process the reference_case.xlsx input file.

Sheet layout expected:
  Sheet1 – Node attributes (ID, x, y, e_i, l_i, service_time, demand)
  Sheet2 – 51×51 travel-time matrix (rows = origins, cols = destinations)

Node 0 is the depot; nodes 1..N are customers.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Expected column names (case-insensitive) that must appear in Sheet1.
REQUIRED_NODE_COLUMNS: list[str] = ["node_id", "x", "y"]
#: Optional columns – filled with defaults when absent.
OPTIONAL_NODE_COLUMNS: dict[str, Any] = {
    "e": 0,            # earliest time window
    "l": 1e9,          # latest time window
    "service_time": 0, # service duration at node
    "demand": 0,       # demand / load
}

#: Column aliases: maps common variant names → canonical name.
COLUMN_ALIASES: dict[str, str] = {
    "id": "node_id",
    "node": "node_id",
    "earliest": "e",
    "latest": "l",
    "service": "service_time",
    "srv": "service_time",
    "srv_time": "service_time",
    "tw_early": "e",
    "tw_late": "l",
    "load": "demand",
    "qty": "demand",
}


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_instance(
    excel_path: str | Path,
    processed_dir: str | Path | None = None,
    force_reload: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load problem instance from Excel and optionally cache results.

    Parameters
    ----------
    excel_path : str | Path
        Path to ``reference_case.xlsx``.
    processed_dir : str | Path | None
        If given, load from / save to pickle caches in this directory.
    force_reload : bool
        Bypass cache even when pickle files exist.

    Returns
    -------
    nodes : pd.DataFrame
        DataFrame with columns ``[node_id, x, y, e, l, service_time, demand]``.
        Row 0 is the depot; rows 1..N are customers.
    travel_time : np.ndarray
        Shape ``(N+1, N+1)`` float64 matrix where entry [i, j] is the
        travel time from node i to node j.  Diagonal must be 0.

    Raises
    ------
    FileNotFoundError
        If *excel_path* does not exist.
    ValueError
        If the Excel structure is incompatible with the expected schema.

    Complexity
    ----------
    O(N²) for matrix load; N ≤ 51 in this problem.
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Try loading from cache -------------------------------------------------
    if processed_dir is not None and not force_reload:
        processed_dir = Path(processed_dir)
        nodes_pkl = processed_dir / "nodes.pkl"
        tt_pkl = processed_dir / "travel_time.pkl"
        if nodes_pkl.exists() and tt_pkl.exists():
            logger.info("Loading cached instance from %s", processed_dir)
            with open(nodes_pkl, "rb") as f:
                nodes = pickle.load(f)
            with open(tt_pkl, "rb") as f:
                travel_time = pickle.load(f)
            logger.debug(
                "Cache hit: nodes shape=%s, travel_time shape=%s",
                nodes.shape,
                travel_time.shape,
            )
            return nodes, travel_time

    # Load from Excel --------------------------------------------------------
    logger.info("Loading Excel instance from %s", excel_path)
    nodes = _load_nodes(excel_path)
    travel_time = _load_travel_time(excel_path, n_nodes=len(nodes))

    # Persist cache ----------------------------------------------------------
    if processed_dir is not None:
        processed_dir = Path(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        with open(processed_dir / "nodes.pkl", "wb") as f:
            pickle.dump(nodes, f)
        with open(processed_dir / "travel_time.pkl", "wb") as f:
            pickle.dump(travel_time, f)
        logger.info("Cached processed data to %s", processed_dir)

    return nodes, travel_time


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and alias column names to canonical form.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame read from Sheet1.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalised column names.

    Complexity
    ----------
    O(C) where C is the number of columns.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)
    return df


def _load_nodes(excel_path: Path) -> pd.DataFrame:
    """Read Sheet1 and return a clean node-attribute DataFrame.

    Parameters
    ----------
    excel_path : Path
        Path to the Excel file.

    Returns
    -------
    pd.DataFrame
        Sorted by ``node_id`` with all required/optional columns present.

    Raises
    ------
    ValueError
        If required columns are missing after alias normalisation.
    """
    raw = pd.read_excel(excel_path, sheet_name=0, header=0)
    df = _normalise_columns(raw)

    # Check required columns
    missing = [c for c in REQUIRED_NODE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Sheet1 is missing required columns after normalisation: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Fill optional columns with defaults
    for col, default in OPTIONAL_NODE_COLUMNS.items():
        if col not in df.columns:
            logger.warning("Column '%s' not found in Sheet1; using default %s", col, default)
            df[col] = default

    # Select and order columns
    canonical_cols = REQUIRED_NODE_COLUMNS + list(OPTIONAL_NODE_COLUMNS.keys())
    df = df[canonical_cols].copy()

    # Ensure node_id is integer and sort
    df["node_id"] = df["node_id"].astype(int)
    df = df.sort_values("node_id").reset_index(drop=True)

    logger.debug("Loaded %d nodes (depot + customers)", len(df))
    return df


def _load_travel_time(excel_path: Path, n_nodes: int) -> np.ndarray:
    """Read Sheet2 and return the travel-time matrix as a numpy array.

    Parameters
    ----------
    excel_path : Path
        Path to the Excel file.
    n_nodes : int
        Expected number of nodes (N+1, including depot).

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, n_nodes)`` float64 array.

    Raises
    ------
    ValueError
        If the sheet cannot be read or dimensions don't match expectations.
    """
    try:
        raw = pd.read_excel(excel_path, sheet_name=1, header=None)
    except Exception as exc:
        raise ValueError(f"Cannot read Sheet2 (travel-time matrix): {exc}") from exc

    # Drop any all-NaN rows/columns (sometimes Excel has a header row)
    raw = raw.dropna(how="all").dropna(axis=1, how="all")

    # If first row/column looks like node indices, drop them
    first_col = raw.iloc[:, 0]
    if pd.api.types.is_numeric_dtype(first_col):
        try:
            indices = first_col.astype(int).tolist()
            if indices == list(range(len(indices))):
                raw = raw.iloc[:, 1:]
                logger.debug("Dropped index column from Sheet2")
        except (ValueError, TypeError):
            pass

    first_row = raw.iloc[0]
    if pd.api.types.is_numeric_dtype(first_row):
        try:
            indices = first_row.astype(int).tolist()
            if indices == list(range(len(indices))):
                raw = raw.iloc[1:, :]
                logger.debug("Dropped index row from Sheet2")
        except (ValueError, TypeError):
            pass

    matrix = raw.values.astype(float)
    logger.debug("Travel-time matrix shape after parsing: %s", matrix.shape)

    # Validate dimensions: accept either n_nodes×n_nodes or smaller (take top-left block)
    if matrix.shape[0] < n_nodes or matrix.shape[1] < n_nodes:
        raise ValueError(
            f"Travel-time matrix shape {matrix.shape} is smaller than expected "
            f"({n_nodes}×{n_nodes}). Check Sheet2."
        )
    if matrix.shape != (n_nodes, n_nodes):
        logger.warning(
            "Travel-time matrix shape %s; taking top-left %dx%d block",
            matrix.shape,
            n_nodes,
            n_nodes,
        )
        matrix = matrix[:n_nodes, :n_nodes]

    return matrix
