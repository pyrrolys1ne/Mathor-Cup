"""
src/solvers/kaiwu_solver.py
----------------------------
Kaiwu SDK solver adapter.

This module wraps the Kaiwu quantum annealing / quantum-inspired solver.
Because the public Kaiwu API may change or may not be available in all
environments, all calls are guarded by a ``try/except`` and the module
degrades gracefully to a warning when the SDK is not installed.

TODO: Fill in the actual Kaiwu API calls once SDK documentation is confirmed.
      The interface contract below must be preserved regardless of SDK version.

Interface contract:
  - ``solve_qubo_kaiwu(Q, cfg)`` accepts a numpy QUBO matrix and returns
    a binary solution vector (np.ndarray of shape (n,)).
  - Raises ``KaiwuUnavailableError`` if the SDK is not installed or the
    endpoint is unreachable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import Kaiwu SDK
# ---------------------------------------------------------------------------

try:
    # TODO: Replace with the actual import path once SDK is confirmed.
    # import kaiwu  # noqa: F401
    _KAIWU_AVAILABLE = False  # Set to True when SDK is installed
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
    endpoint : str
        API endpoint URL.
    token : str
        Authentication token / API key.
    num_reads : int
        Number of solution reads (samples) to request.
    annealing_time : int
        Annealing duration in microseconds (device-specific).
    """

    endpoint: str = ""
    token: str = ""
    num_reads: int = 100
    annealing_time: int = 20


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

    TODO
    ----
    1. Import the actual kaiwu SDK.
    2. Convert Q to the SDK's native format (e.g., QUBO dict or Ising Hamiltonian).
    3. Instantiate the sampler with endpoint and token.
    4. Call sampler.sample_qubo() or equivalent.
    5. Extract the best sample from the SampleSet.
    6. Convert to numpy binary array and return.
    """
    if not _KAIWU_AVAILABLE:
        raise KaiwuUnavailableError(
            "Kaiwu SDK is not installed. Install it with: pip install kaiwu\n"
            "Or switch to the SA backend via the 'solver.backend: sa' config option."
        )

    if cfg is None:
        cfg = KaiwuConfig()

    if not cfg.endpoint:
        raise KaiwuUnavailableError(
            "Kaiwu endpoint is not configured. "
            "Set 'kaiwu.endpoint' and 'kaiwu.token' in your config file."
        )

    # TODO: Implement actual Kaiwu API call.
    # Example (pseudocode):
    #   import kaiwu
    #   sampler = kaiwu.QASampler(endpoint=cfg.endpoint, token=cfg.token)
    #   qubo_dict = _matrix_to_qubo_dict(Q)
    #   response = sampler.sample_qubo(
    #       qubo_dict,
    #       num_reads=cfg.num_reads,
    #       annealing_time=cfg.annealing_time,
    #   )
    #   best_sample = response.first.sample
    #   n = Q.shape[0]
    #   x = np.array([best_sample[i] for i in range(n)], dtype=np.float64)
    #   return x

    raise NotImplementedError("Kaiwu API call not yet implemented. See TODO above.")


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
