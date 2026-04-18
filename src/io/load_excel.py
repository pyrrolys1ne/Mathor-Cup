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
REQUIRED_NODE_COLUMNS: list[str] = ["node_id"]
#: Optional columns – filled with defaults when absent.
OPTIONAL_NODE_COLUMNS: dict[str, Any] = {
    "x": 0.0,          # x coordinate (fallback when absent)
    "y": 0.0,          # y coordinate (fallback when absent)
    "e": 0,            # earliest time window
    "l": 1e9,          # latest time window
    "service_time": 0, # service duration at node
    "demand": 0,       # demand / load
}

#: Column aliases: maps common variant names → canonical name.
COLUMN_ALIASES: dict[str, str] = {
    "id": "node_id",
    "node": "node_id",
    "节点id": "node_id",
    "节点_id": "node_id",
    "节点编号": "node_id",
    "节点": "node_id",
    "坐标x": "x",
    "坐标y": "y",
    "节点坐标x": "x",
    "节点坐标y": "y",
    "x坐标": "x",
    "y坐标": "y",
    "earliest": "e",
    "latest": "l",
    "开始服务时间下界": "e",
    "开始服务时间上界": "l",
    "最早开始时间": "e",
    "最晚开始时间": "l",
    "service": "service_time",
    "srv": "service_time",
    "srv_time": "service_time",
    "服务时间": "service_time",
    "tw_early": "e",
    "tw_late": "l",
    "load": "demand",
    "qty": "demand",
    "需求量": "demand",
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
            if _is_cache_valid(nodes, travel_time):
                logger.debug(
                    "Cache hit: nodes shape=%s, travel_time shape=%s",
                    nodes.shape,
                    travel_time.shape,
                )
                return nodes, travel_time
            logger.warning(
                "Cached matrix appears invalid (shape/diagonal/NaN). Reloading from Excel."
            )

    # Load from Excel --------------------------------------------------------
    logger.info("Loading Excel instance from %s", excel_path)
    nodes = _load_nodes(excel_path)
    travel_time = _load_travel_time(excel_path, n_nodes=len(nodes))
    nodes = _ensure_plot_coords(nodes, travel_time)

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
    raw_num = raw.apply(pd.to_numeric, errors="coerce")

    # Common format: top-left NaN, first row/col are node index labels.
    # Example:
    #   NaN  0  1  2 ...
    #    0   0  2  2 ...
    if raw_num.shape[0] >= n_nodes + 1 and raw_num.shape[1] >= n_nodes + 1:
        top_left_nan = not np.isfinite(raw_num.iat[0, 0])
        row_labels = raw_num.iloc[0, 1 : n_nodes + 1]
        col_labels = raw_num.iloc[1 : n_nodes + 1, 0]
        if top_left_nan and _is_index_sequence(row_labels) and _is_index_sequence(col_labels):
            raw_num = raw_num.iloc[1:, 1:]
            logger.debug("Dropped index row/column with top-left empty cell from Sheet2")

    # If first row/column looks like node indices, drop them
    if _is_index_sequence(raw_num.iloc[:, 0]):
        raw_num = raw_num.iloc[:, 1:]
        logger.debug("Dropped index column from Sheet2")

    if _is_index_sequence(raw_num.iloc[0, :]):
        raw_num = raw_num.iloc[1:, :]
        logger.debug("Dropped index row from Sheet2")

    matrix = raw_num.values.astype(float)
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


def _is_index_sequence(values: pd.Series) -> bool:
    """Return True if values look like index labels: 0..N-1 or 1..N."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    arr_int = arr.astype(int)
    if not np.allclose(arr, arr_int):
        return False
    seq0 = np.arange(arr_int.size)
    seq1 = np.arange(1, arr_int.size + 1)
    return np.array_equal(arr_int, seq0) or np.array_equal(arr_int, seq1)


def _is_cache_valid(nodes: pd.DataFrame, travel_time: np.ndarray) -> bool:
    """Basic cache integrity checks to avoid persistent bad cache reuse."""
    if travel_time.ndim != 2 or travel_time.shape[0] != travel_time.shape[1]:
        return False
    n_nodes = len(nodes)
    if travel_time.shape != (n_nodes, n_nodes):
        return False
    if not np.isfinite(travel_time).all():
        return False
    if not np.allclose(np.diag(travel_time), 0.0):
        return False

    # Reject cache if plotting coordinates are missing/degenerate.
    if "x" in nodes.columns and "y" in nodes.columns:
        xv = pd.to_numeric(nodes["x"], errors="coerce").to_numpy(dtype=float)
        yv = pd.to_numeric(nodes["y"], errors="coerce").to_numpy(dtype=float)
        pts = {
            (float(a), float(b))
            for a, b in zip(xv, yv)
            if np.isfinite(a) and np.isfinite(b)
        }
        if len(pts) <= 1:
            return False

    return True


def _ensure_plot_coords(nodes: pd.DataFrame, travel_time: np.ndarray) -> pd.DataFrame:
    """Ensure nodes have usable 2D coordinates for plotting.

    When source data has no x/y columns (or all points collapse to one place),
    derive pseudo-coordinates from the travel-time matrix via classical MDS.
    This affects visualisation only, not optimisation logic.
    """
    out = nodes.copy()

    if "x" not in out.columns or "y" not in out.columns:
        out["x"] = np.nan
        out["y"] = np.nan

    x = pd.to_numeric(out["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(out["y"], errors="coerce").to_numpy(dtype=float)

    all_missing = np.isnan(x).all() or np.isnan(y).all()
    unique_points = {
        (float(a), float(b))
        for a, b in zip(x, y)
        if np.isfinite(a) and np.isfinite(b)
    }
    degenerate = len(unique_points) <= 1

    if not all_missing and not degenerate:
        return out

    coords = _coords_from_distance_matrix(travel_time)
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    logger.info("x/y missing or degenerate; generated fallback coordinates from travel-time matrix for plotting.")
    return out


def _coords_from_distance_matrix(dist: np.ndarray) -> np.ndarray:
    """Classical MDS: derive 2D coordinates from a distance-like matrix."""
    n = dist.shape[0]
    d = np.asarray(dist, dtype=float)

    # Guard against tiny negative values from numerical noise
    d = np.maximum(d, 0.0)
    d2 = d ** 2

    j = np.eye(n) - np.ones((n, n), dtype=float) / n
    b = -0.5 * j @ d2 @ j

    evals, evecs = np.linalg.eigh(b)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    pos = evals > 1e-12
    if np.count_nonzero(pos) >= 2:
        lam = np.sqrt(evals[:2])
        return evecs[:, :2] * lam

    # Fallback: simple circle layout if matrix has insufficient rank
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))
