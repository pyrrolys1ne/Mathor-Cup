"""
src/solvers/kaiwu_solver.py
----------------------------
Kaiwu SDK solver adapter.

This module wraps the Kaiwu quantum annealing / quantum-inspired solver.
Because the public Kaiwu API may change or may not be available in all
environments, all calls are guarded by a ``try/except`` and the module
degrades gracefully to a warning when the SDK is not installed.

The interface contract below is preserved so callers can switch between
SA and Kaiwu backends without changing upstream code.

Interface contract:
  - ``solve_qubo_kaiwu(Q, cfg)`` accepts a numpy QUBO matrix and returns
    a binary solution vector (np.ndarray of shape (n,)).
  - Raises ``KaiwuUnavailableError`` if the SDK is not installed or the
    endpoint is unreachable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import Kaiwu SDK
# ---------------------------------------------------------------------------

try:
    import kaiwu.license as kaiwu_license
    from kaiwu.qubo import qubo_matrix_to_qubo_model
    from kaiwu.sampler import SimulatedAnnealingSampler
    from kaiwu.solver import SimpleSolver

    _KAIWU_AVAILABLE = True
except ImportError:
    _KAIWU_AVAILABLE = False
    logger.info("Kaiwu SDK not found. Kaiwu backend unavailable.")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KaiwuUnavailableError(RuntimeError):
    """Raised when Kaiwu SDK is not installed or endpoint is unreachable."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class KaiwuConfig:
    """Configuration for the Kaiwu solver backend.

    Attributes
    ----------
    user_id : str
        Kaiwu license user id. If empty, environment variable KAIWU_USER_ID is used.
    sdk_code : str
        Kaiwu license code. If empty, environment variable KAIWU_SDK_CODE is used.
    num_reads : int
        Number of solution reads (samples) to request.
    annealing_time : int
        Annealing duration in microseconds (device-specific).
    seed : int | None
        Random seed for sampler. None means SDK default randomness.
    quiet : bool
        If True, suppress verbose Kaiwu internal logs when possible.

    Notes
    -----
    The legacy fields ``endpoint`` and ``token`` are kept for backward
    compatibility with older config files, but they are not used by the
    currently validated local Kaiwu SDK license flow.
    """

    user_id: str = ""
    sdk_code: str = ""
    endpoint: str = ""  # legacy compatibility
    token: str = ""     # legacy compatibility
    num_reads: int = 100
    annealing_time: int = 20
    seed: int | None = None
    quiet: bool = True


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------


def solve_qubo_kaiwu(
    Q: np.ndarray,
    cfg: KaiwuConfig | None = None,
) -> np.ndarray:
    """Submit a QUBO to the Kaiwu backend and return the best binary solution.

    Parameters
    ----------
    Q : np.ndarray
        Shape (n, n) upper-triangular QUBO matrix.
    cfg : KaiwuConfig | None
        Kaiwu connection and solver parameters.

    Returns
    -------
    np.ndarray
        Best binary solution vector found, shape (n,).

    Raises
    ------
    KaiwuUnavailableError
        If the Kaiwu SDK is not installed or the endpoint is unreachable.

    Complexity
    ----------
    Depends on the Kaiwu backend; typically O(1) from the caller's view
    (the complexity is on the quantum device side).

    """
    if not _KAIWU_AVAILABLE:
        raise KaiwuUnavailableError(
            "Kaiwu SDK is not installed. Install it with: pip install kaiwu\n"
            "Or switch to the SA backend via the 'solver.backend: sa' config option."
        )

    if cfg is None:
        cfg = KaiwuConfig()

    user_id = cfg.user_id or os.getenv("KAIWU_USER_ID", "")
    sdk_code = cfg.sdk_code or os.getenv("KAIWU_SDK_CODE", "")

    if not user_id or not sdk_code:
        raise KaiwuUnavailableError(
            "Kaiwu license credentials are missing. "
            "Set 'kaiwu.user_id' and 'kaiwu.sdk_code' in config, "
            "or environment variables KAIWU_USER_ID / KAIWU_SDK_CODE."
        )

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be a square 2D matrix, got shape={Q.shape}")

    try:
        kaiwu_license.init(user_id=user_id, sdk_code=sdk_code)
        qubo_model = qubo_matrix_to_qubo_model(Q)

        if cfg.quiet:
            logging.getLogger("kaiwu").setLevel(logging.WARNING)

        sampler = SimulatedAnnealingSampler(
            iterations_per_t=max(1, int(cfg.annealing_time)),
            size_limit=max(1, int(cfg.num_reads)),
            rand_seed=cfg.seed,
        )
        solver = SimpleSolver(sampler)
        sample, _energy = solver.solve_qubo(qubo_model)
        x = _sample_to_array(sample, Q.shape[0])
        return x
    except Exception as exc:
        raise KaiwuUnavailableError(f"Kaiwu solve failed: {exc}") from exc


def is_available() -> bool:
    """Return True if the Kaiwu SDK is installed and importable.

    Returns
    -------
    bool
    """
    return _KAIWU_AVAILABLE


# ---------------------------------------------------------------------------
# Helper: convert dense Q matrix to QUBO dict
# ---------------------------------------------------------------------------


def _matrix_to_qubo_dict(Q: np.ndarray) -> dict[tuple[int, int], float]:
    """Convert upper-triangular QUBO matrix to dict format.

    Parameters
    ----------
    Q : np.ndarray
        Shape (n, n) upper-triangular QUBO matrix.

    Returns
    -------
    dict[tuple[int, int], float]
        Sparse QUBO representation: {(i, j): coefficient}.

    Complexity
    ----------
    O(N²)
    """
    n = Q.shape[0]
    qubo: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i, n):
            val = float(Q[i, j])
            if val != 0.0:
                qubo[(i, j)] = val
    return qubo


def _sample_to_array(sample: object, n: int) -> np.ndarray:
    """Convert Kaiwu sample output to a dense binary numpy vector.

    Supported forms:
    - dict with keys like "b[0]", "x0", or integer-like strings
    - sequence with length n
    """
    if isinstance(sample, dict):
        x = np.zeros(n, dtype=np.float64)
        for k, v in sample.items():
            idx = _parse_var_index(str(k))
            if idx is not None and 0 <= idx < n:
                x[idx] = 1.0 if float(v) >= 0.5 else 0.0
        return x

    arr = np.asarray(sample, dtype=np.float64).reshape(-1)
    if arr.size != n:
        raise ValueError(f"Unexpected sample size: expected {n}, got {arr.size}")
    return (arr >= 0.5).astype(np.float64)


def _parse_var_index(name: str) -> int | None:
    """Extract integer index from variable name (e.g. b[12], x12, 12)."""
    if name.isdigit():
        return int(name)
    if "[" in name and "]" in name:
        try:
            return int(name[name.index("[") + 1 : name.index("]")])
        except ValueError:
            return None
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else None
