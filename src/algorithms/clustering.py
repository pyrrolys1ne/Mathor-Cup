"""
src/algorithms/clustering.py
------------------------------
Customer clustering for the decomposition strategy (Q3/Q4).

Supports:
  - K-Means clustering on (x, y) coordinates.
  - DBSCAN clustering (auto-determines number of clusters).

Returns cluster labels (np.ndarray, shape N) and cluster centres.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from src.core.graph_model import ProblemGraph

logger = logging.getLogger(__name__)


def cluster_customers(
    graph: ProblemGraph,
    customer_ids: list[int],
    method: str = "kmeans",
    n_clusters: int = 5,
    seed: int = 42,
    dbscan_eps: float = 10.0,
    dbscan_min_samples: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster customers using geographic coordinates.

    Parameters
    ----------
    graph : ProblemGraph
    customer_ids : list[int]
        Customer node IDs to cluster (depot excluded).
    method : str
        ``'kmeans'`` or ``'dbscan'``.
    n_clusters : int
        Number of clusters for K-Means.
    seed : int
        Random seed.
    dbscan_eps : float
        DBSCAN neighbourhood radius.
    dbscan_min_samples : int
        DBSCAN minimum samples per neighbourhood.

    Returns
    -------
    labels : np.ndarray shape (N,)
        Cluster label per customer (int).  DBSCAN outliers labelled -1 are
        reassigned to the nearest cluster centre.
    centres : np.ndarray shape (K, 2)
        Cluster centres in (x, y) space.

    Raises
    ------
    ValueError
        If ``method`` is not recognised.

    Complexity
    ----------
    O(N * K * iter) for K-Means; O(N²) for DBSCAN.
    """
    # Build coordinate matrix
    coords = np.array([graph.coords(cid) for cid in customer_ids], dtype=float)

    if method == "kmeans":
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        km.fit(coords)
        labels = km.labels_.astype(int)
        centres = km.cluster_centers_
        logger.info(
            "K-Means clustering: K=%d, sizes=%s",
            n_clusters,
            _label_sizes(labels),
        )
    elif method == "dbscan":
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        db.fit(coords)
        labels = db.labels_.astype(int)
        # Compute centres as mean of assigned points
        unique_labels = [l for l in np.unique(labels) if l != -1]
        centres = np.array(
            [coords[labels == lbl].mean(axis=0) for lbl in unique_labels]
        )
        # Reassign outliers (-1) to nearest centre
        if -1 in labels and len(centres) > 0:
            for idx in np.where(labels == -1)[0]:
                dists = np.linalg.norm(centres - coords[idx], axis=1)
                labels[idx] = unique_labels[int(np.argmin(dists))]
        logger.info(
            "DBSCAN clustering: clusters=%d, sizes=%s",
            len(unique_labels),
            _label_sizes(labels),
        )
    else:
        raise ValueError(f"Unknown clustering method: '{method}'. Use 'kmeans' or 'dbscan'.")

    return labels, centres


def _label_sizes(labels: np.ndarray) -> dict[int, int]:
    """Return cluster size mapping {label: count}."""
    unique, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}
