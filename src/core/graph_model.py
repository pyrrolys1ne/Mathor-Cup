"""
src/core/graph_model.py
------------------------
Graph-based representation of the logistics problem instance.

The ``ProblemGraph`` class wraps the raw node DataFrame and travel-time
matrix, providing accessor methods used by solvers, QUBO builders and
evaluation modules.

Node 0  = depot (配送中心)
Nodes 1..N = customers (客户)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core data class
# ---------------------------------------------------------------------------


@dataclass
class ProblemGraph:
    """Encapsulates all instance data for the logistics optimisation problem.

    Attributes
    ----------
    nodes : pd.DataFrame
        Node attributes with columns:
        ``[node_id, x, y, e, l, service_time, demand]``.
    travel_time : np.ndarray
        Shape ``(N+1, N+1)`` float64 symmetric (or asymmetric) matrix.
        ``travel_time[i, j]`` = travel time from node ``i`` to node ``j``.
    n_customers : int
        Number of customer nodes (excludes the depot).
    """

    nodes: pd.DataFrame
    travel_time: np.ndarray
    n_customers: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_customers = len(self.nodes) - 1  # depot at index 0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def depot_id(self) -> int:
        """Integer ID of the depot node (always 0)."""
        return 0

    @property
    def customer_ids(self) -> list[int]:
        """Sorted list of customer node IDs (1..N).

        Complexity: O(N)
        """
        return sorted(self.nodes.loc[self.nodes["node_id"] != 0, "node_id"].tolist())

    @property
    def all_ids(self) -> list[int]:
        """All node IDs including depot, sorted ascending.

        Complexity: O(N)
        """
        return sorted(self.nodes["node_id"].tolist())

    def travel(self, i: int, j: int) -> float:
        """Travel time from node ``i`` to node ``j``.

        Parameters
        ----------
        i, j : int
            Node IDs (0 = depot).

        Returns
        -------
        float
            Travel time (non-negative).

        Complexity: O(1)
        """
        return float(self.travel_time[i, j])

    def node_attr(self, node_id: int) -> pd.Series:
        """Return attribute row for a given node.

        Parameters
        ----------
        node_id : int

        Returns
        -------
        pd.Series
            Row from ``self.nodes`` for the requested node.

        Raises
        ------
        KeyError
            If ``node_id`` is not found.

        Complexity: O(N) scan — use sparingly inside tight loops.
        """
        row = self.nodes[self.nodes["node_id"] == node_id]
        if row.empty:
            raise KeyError(f"node_id={node_id} not found in nodes DataFrame")
        return row.iloc[0]

    def coords(self, node_id: int) -> tuple[float, float]:
        """Return (x, y) coordinates of a node.

        Parameters
        ----------
        node_id : int

        Returns
        -------
        tuple[float, float]

        Complexity: O(N)
        """
        attr = self.node_attr(node_id)
        return float(attr["x"]), float(attr["y"])

    def time_window(self, node_id: int) -> tuple[float, float]:
        """Return the (e_i, l_i) time window for a node.

        Parameters
        ----------
        node_id : int

        Returns
        -------
        tuple[float, float]
            ``(earliest, latest)`` service start times.

        Complexity: O(N)
        """
        attr = self.node_attr(node_id)
        return float(attr["e"]), float(attr["l"])

    def service_time(self, node_id: int) -> float:
        """Service duration at a node.

        Parameters
        ----------
        node_id : int

        Returns
        -------
        float

        Complexity: O(N)
        """
        return float(self.node_attr(node_id)["service_time"])

    def demand(self, node_id: int) -> float:
        """Demand (load) at a node.

        Parameters
        ----------
        node_id : int

        Returns
        -------
        float

        Complexity: O(N)
        """
        return float(self.node_attr(node_id)["demand"])

    def route_travel_time(self, route: Sequence[int]) -> float:
        """Compute total travel time for a given node sequence.

        The route is assumed to start and end at the depot (node 0).
        If the first element is not 0, the depot is prepended; if the
        last element is not 0, the depot is appended.

        Parameters
        ----------
        route : Sequence[int]
            Ordered list of node IDs including depot at start/end.

        Returns
        -------
        float
            Sum of travel times along the full route.

        Complexity: O(|route|)
        """
        full = list(route)
        if not full:
            return 0.0
        if full[0] != self.depot_id:
            full = [self.depot_id] + full
        if full[-1] != self.depot_id:
            full = full + [self.depot_id]
        return sum(self.travel(full[k], full[k + 1]) for k in range(len(full) - 1))

    # ------------------------------------------------------------------
    # NetworkX conversion (for visualisation)
    # ------------------------------------------------------------------

    def to_networkx(self) -> nx.DiGraph:
        """Convert the complete graph to a NetworkX DiGraph.

        Edge weights correspond to travel times.

        Returns
        -------
        nx.DiGraph
            Directed graph with all O(N²) edges.

        Complexity: O(N²)
        """
        g = nx.DiGraph()
        for _, row in self.nodes.iterrows():
            nid = int(row["node_id"])
            g.add_node(nid, x=float(row["x"]), y=float(row["y"]))
        n = len(self.nodes)
        for i in range(n):
            for j in range(n):
                if i != j:
                    g.add_edge(i, j, weight=float(self.travel_time[i, j]))
        return g

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ProblemGraph(n_customers={self.n_customers}, "
            f"matrix_shape={self.travel_time.shape})"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def build_graph(
    nodes: pd.DataFrame,
    travel_time: np.ndarray,
) -> ProblemGraph:
    """Construct a ``ProblemGraph`` from pre-validated data.

    Parameters
    ----------
    nodes : pd.DataFrame
        Validated node attributes (from ``load_excel.load_instance``).
    travel_time : np.ndarray
        Validated travel-time matrix.

    Returns
    -------
    ProblemGraph

    Complexity
    ----------
    O(1) — just wraps existing objects.
    """
    return ProblemGraph(nodes=nodes, travel_time=travel_time)


def subgraph(graph: ProblemGraph, node_ids: list[int]) -> ProblemGraph:
    """Extract a sub-problem containing the depot and a subset of customers.

    Parameters
    ----------
    graph : ProblemGraph
        Parent graph.
    node_ids : list[int]
        Customer IDs to include (depot is always added automatically).

    Returns
    -------
    ProblemGraph
        Sub-graph with re-indexed node IDs *preserved* (not re-mapped).

    Complexity
    ----------
    O(K²) where K = len(node_ids).
    """
    ids = sorted(set([graph.depot_id] + node_ids))
    sub_nodes = graph.nodes[graph.nodes["node_id"].isin(ids)].reset_index(drop=True)
    sub_tt = graph.travel_time[np.ix_(ids, ids)]
    return ProblemGraph(nodes=sub_nodes, travel_time=sub_tt)
