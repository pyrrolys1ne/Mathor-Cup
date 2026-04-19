"""
src/main.py
------------
Command-line entry point for the MathorCup 2026 logistics optimisation project.

Usage:
    python -m src.main --config configs/q1.yaml
    python -m src.main --config configs/q2.yaml
    python -m src.main --config configs/q3.yaml
    python -m src.main --config configs/q4.yaml [--sensitivity]
    python -m src.main --config configs/q4.yaml --phase data

All configuration is driven by the YAML file; CLI flags supplement the config.
"""

from __future__ import annotations

import logging
import json
import sys
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Logging setup (must be first before importing project modules)
# ---------------------------------------------------------------------------


def _setup_logging(log_dir: str, log_level: str, problem: str) -> None:
    """Configure root logger with file + console handlers."""
    log_path = Path(log_dir) / f"{problem}_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


def _as_float(value: Any, default: float) -> float:
    """Safely coerce config values to float."""
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid float config value=%r, fallback to %s", value, default)
        return float(default)


def _as_int(value: Any, default: int) -> int:
    """Safely coerce config values to int."""
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid int config value=%r, fallback to %s", value, default)
        return int(default)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path : str | Path

    Returns
    -------
    dict[str, Any]

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


def run_data_phase(cfg: dict[str, Any]) -> tuple:
    """Load, validate and cache the problem instance.

    Parameters
    ----------
    cfg : dict

    Returns
    -------
    tuple[ProblemGraph, ValidationReport]
    """
    from src.core.graph_model import build_graph
    from src.io.load_excel import load_instance
    from src.io.validate_data import validate_or_raise

    data_cfg = cfg["data"]
    nodes, travel_time = load_instance(
        excel_path=data_cfg["raw_excel"],
        processed_dir=data_cfg.get("processed_dir"),
    )
    n_customers = data_cfg.get("num_customers")

    # Keep experiment scale consistent with config by truncating customers when
    # the raw file contains a larger instance (e.g., Q1/Q2 config=15, raw=50).
    if n_customers is not None:
        expected = int(n_customers)
        actual = len(nodes) - 1
        if 0 < expected < actual:
            customer_ids = sorted(
                nodes.loc[nodes["node_id"] != 0, "node_id"].astype(int).tolist()
            )
            kept_customers = customer_ids[:expected]
            keep_ids = [0] + kept_customers
            nodes = (
                nodes[nodes["node_id"].isin(keep_ids)]
                .sort_values("node_id")
                .reset_index(drop=True)
            )
            travel_time = travel_time[np.ix_(keep_ids, keep_ids)]
            logger.info(
                "Down-sampled instance by config: customers %d -> %d (kept node IDs 1..%d)",
                actual,
                expected,
                expected,
            )

    report = validate_or_raise(nodes, travel_time, expected_n_customers=n_customers)
    logger.info("Data validation passed.\n%s", report.summary())

    graph = build_graph(nodes, travel_time)
    return graph, report


def _result_paths(output_cfg: dict[str, Any], problem: str, source: str) -> tuple[Path, Path]:
    """Build source-specific output paths to avoid overwriting by different solvers."""
    result_dir = Path(output_cfg.get("result_dir", "outputs/results"))
    fig_dir = Path(output_cfg["figure_dir"])
    return (
        result_dir / f"{problem}_result_{source}.csv",
        fig_dir / f"{problem}_route_{source}.png",
    )


def _append_single_result_summary_csv(result_csv: Path, metrics: dict[str, object]) -> None:
    """Append a human-readable summary block to the bottom of single-route result CSV."""
    route = metrics.get("route", [])
    travel = float(metrics.get("total_travel_time", 0.0))
    penalty = float(metrics.get("total_penalty", 0.0))
    objective = float(metrics.get("objective", 0.0))
    n_customers = int(metrics.get("n_customers", 0))

    lines = [
        "",
        "# ==================================================",
        "# Summary",
        "# ==================================================",
        f"# Route,{route}",
        f"# Travel time,{travel:.4f}",
        f"# TW penalty,{penalty:.4f}",
        f"# Objective,{objective:.4f}",
        f"# Customers served,{n_customers}",
        "# ==================================================",
    ]
    with result_csv.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_q1(cfg: dict[str, Any], graph) -> None:
    """Run Problem 1 solver and output results."""
    from src.algorithms.local_search import two_opt
    from src.eval.metrics import save_metrics_csv, single_route_metrics
    from src.qubo.q1_qubo import build_q1_qubo, decode_q1_solution
    from src.solvers.sa_solver import SAConfig, solve_qubo_sa, solve_route_sa
    from src.viz.plot_routes import plot_single_route

    output_cfg = cfg["output"]
    solver_cfg = cfg["solver"]
    qubo_cfg = cfg.get("qubo", {})
    sa_cfg_dict = cfg.get("sa", {})

    sa_cfg = SAConfig(
        initial_temp=_as_float(sa_cfg_dict.get("initial_temp"), 1000.0),
        cooling_rate=_as_float(sa_cfg_dict.get("cooling_rate"), 0.995),
        min_temp=_as_float(sa_cfg_dict.get("min_temp"), 1e-4),
        n_iter_per_temp=_as_int(sa_cfg_dict.get("n_iter_per_temp"), 200),
        seed=_as_int(sa_cfg_dict.get("seed"), 42),
    )

    backend = solver_cfg.get("backend", "sa")
    logger.info("Q1: backend=%s, n_customers=%d", backend, graph.n_customers)

    if backend == "kaiwu":
        from src.qubo.q1_qubo import build_q1_qubo

        qr = build_q1_qubo(
            graph,
            penalty_visit=qubo_cfg.get("penalty_visit", 500),
            penalty_position=qubo_cfg.get("penalty_position", 500),
        )
        try:
            from src.solvers.kaiwu_solver import KaiwuConfig, solve_qubo_kaiwu

            kw_cfg = cfg.get("kaiwu", {})
            kw = KaiwuConfig(**{k: v for k, v in kw_cfg.items() if k in KaiwuConfig.__dataclass_fields__})
            n_runs = max(1, _as_int(kw_cfg.get("n_runs"), 1))
            post_two_opt = bool(kw_cfg.get("postprocess_two_opt", True))
            base_seed = kw.seed

            best_route: list[int] | None = None
            best_cost = float("inf")

            for run_idx in range(n_runs):
                run_no = run_idx + 1
                run_t0 = time.perf_counter()

                # For reproducibility, use base seed + run index (no cumulative drift).
                if base_seed is not None:
                    kw.seed = int(base_seed) + run_idx

                logger.info(
                    "Q1 Kaiwu run %d/%d start (seed=%s, num_reads=%d, annealing_time=%d)",
                    run_no,
                    n_runs,
                    kw.seed,
                    kw.num_reads,
                    kw.annealing_time,
                )

                x = solve_qubo_kaiwu(qr.Q, kw)
                cand_route = decode_q1_solution(x, qr.n_nodes, qr.var_idx)

                if post_two_opt:
                    customers = [n for n in cand_route if n != graph.depot_id]
                    customers = two_opt(customers, graph)
                    cand_route = [graph.depot_id] + customers + [graph.depot_id]

                cand_cost = graph.route_travel_time(cand_route)
                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best_route = cand_route

                logger.info(
                    "Q1 Kaiwu run %d/%d done: travel=%.4f, best_so_far=%.4f, elapsed=%.2fs",
                    run_no,
                    n_runs,
                    cand_cost,
                    best_cost,
                    time.perf_counter() - run_t0,
                )

            if best_route is None:
                raise RuntimeError("Kaiwu produced no valid route.")

            route = best_route
            logger.info(
                "Q1 Kaiwu search complete: runs=%d, post_two_opt=%s, best_travel=%.4f",
                n_runs,
                post_two_opt,
                best_cost,
            )
        except Exception as exc:
            logger.warning("Kaiwu failed (%s); falling back to SA.", exc)
            backend = "sa"

    if backend == "sa":
        # Permutation-space SA (more efficient for small instances)
        def cost_fn(perm: list[int]) -> float:
            return graph.route_travel_time(perm)

        sa_result = solve_route_sa(graph.customer_ids, cost_fn, sa_cfg)
        route = [graph.depot_id] + sa_result.best_solution + [graph.depot_id]
        # Local search refinement
        improved = two_opt(sa_result.best_solution, graph)
        route = [graph.depot_id] + improved + [graph.depot_id]

    metrics = single_route_metrics(route, graph, alpha=0.0, beta=0.0)
    logger.info(
        "Q1 result: route=%s, total_travel_time=%.4f",
        metrics["route"],
        metrics["total_travel_time"],
    )

    source = "kaiwu" if backend == "kaiwu" else "sa"
    result_csv, route_fig = _result_paths(output_cfg, "q1", source)
    save_metrics_csv(metrics, result_csv)
    _append_single_result_summary_csv(result_csv, metrics)
    plot_single_route(
        route,
        graph,
        title=f"Q1 Route (travel={metrics['total_travel_time']:.2f})",
        save_path=route_fig,
    )
    _print_single_result(metrics, "Q1")


def run_q2(cfg: dict[str, Any], graph) -> None:
    """Run Problem 2 solver and output results."""
    from src.algorithms.local_search import two_opt
    from src.eval.metrics import save_metrics_csv, single_route_metrics
    from src.qubo.q1_qubo import decode_q1_solution
    from src.qubo.q2_qubo import build_q2_qubo
    from src.solvers.sa_solver import SAConfig, solve_route_sa
    from src.viz.plot_routes import plot_single_route

    output_cfg = cfg["output"]
    solver_cfg = cfg.get("solver", {})
    qubo_cfg = cfg.get("qubo", {})
    tw_cfg = cfg.get("time_window", {})
    sa_cfg_dict = cfg.get("sa", {})
    alpha = tw_cfg.get("alpha", 10.0)
    beta = tw_cfg.get("beta", 20.0)

    sa_cfg = SAConfig(
        initial_temp=_as_float(sa_cfg_dict.get("initial_temp"), 2000.0),
        cooling_rate=_as_float(sa_cfg_dict.get("cooling_rate"), 0.995),
        min_temp=_as_float(sa_cfg_dict.get("min_temp"), 1e-4),
        n_iter_per_temp=_as_int(sa_cfg_dict.get("n_iter_per_temp"), 300),
        seed=_as_int(sa_cfg_dict.get("seed"), 42),
    )

    from src.core.time_window import simulate_route_timing

    def cost_fn(perm: list[int]) -> float:
        timing = simulate_route_timing(perm, graph, alpha, beta)
        return timing.total_travel_time + timing.total_penalty

    backend = solver_cfg.get("backend", "sa")
    logger.info("Q2: backend=%s, n_customers=%d", backend, graph.n_customers)

    if backend == "kaiwu":
        qr = build_q2_qubo(
            graph,
            penalty_visit=qubo_cfg.get("penalty_visit", 500),
            penalty_position=qubo_cfg.get("penalty_position", 500),
        )
        try:
            from src.solvers.kaiwu_solver import KaiwuConfig, solve_qubo_kaiwu

            kw_cfg = cfg.get("kaiwu", {})
            kw = KaiwuConfig(**{k: v for k, v in kw_cfg.items() if k in KaiwuConfig.__dataclass_fields__})
            n_restarts = max(1, _as_int(kw_cfg.get("n_restarts"), 8))
            base_seed = kw.seed if kw.seed is not None else _as_int(solver_cfg.get("seed"), 42)

            best_route: list[int] | None = None
            best_obj = float("inf")
            success = 0
            for i in range(n_restarts):
                kw_try = KaiwuConfig(**kw.__dict__)
                kw_try.seed = int(base_seed) + i
                try:
                    x = solve_qubo_kaiwu(qr.Q, kw_try)
                    cand = decode_q1_solution(x, qr.n_nodes, qr.var_idx)
                    customers = [n for n in cand if n != graph.depot_id]
                    customers = two_opt(customers, graph)
                    cand = [graph.depot_id] + customers + [graph.depot_id]
                    m = single_route_metrics(cand, graph, alpha=alpha, beta=beta)
                    success += 1
                    if m["objective"] < best_obj:
                        best_obj = float(m["objective"])
                        best_route = cand
                except Exception as run_exc:
                    logger.warning("Q2 Kaiwu restart %d/%d failed: %s", i + 1, n_restarts, run_exc)

            if best_route is None:
                raise RuntimeError("All Kaiwu restarts failed.")

            route = best_route
            logger.info(
                "Q2 Kaiwu candidate scan complete: success=%d/%d, best_obj=%.4f",
                success,
                n_restarts,
                best_obj,
            )
        except Exception as exc:
            logger.warning("Kaiwu failed (%s); falling back to SA.", exc)
            backend = "sa"

    if backend == "sa":
        sa_result = solve_route_sa(graph.customer_ids, cost_fn, sa_cfg)
        improved = two_opt(sa_result.best_solution, graph)
        route = [graph.depot_id] + improved + [graph.depot_id]

    metrics = single_route_metrics(route, graph, alpha=alpha, beta=beta)
    logger.info(
        "Q2 result: obj=%.4f (travel=%.4f, penalty=%.4f)",
        metrics["objective"],
        metrics["total_travel_time"],
        metrics["total_penalty"],
    )

    source = "kaiwu" if backend == "kaiwu" else "sa"
    result_csv, route_fig = _result_paths(output_cfg, "q2", source)
    save_metrics_csv(metrics, result_csv)
    _append_single_result_summary_csv(result_csv, metrics)
    plot_single_route(
        route,
        graph,
        title=f"Q2 Route (obj={metrics['objective']:.2f})",
        save_path=route_fig,
    )
    _print_single_result(metrics, "Q2")


def run_q3(cfg: dict[str, Any], graph) -> None:
    """Run Problem 3 hybrid large-scale solver and output results."""
    from src.eval.metrics import save_metrics_csv, single_route_metrics
    from src.solvers.hybrid_large_scale import HybridConfig, solve_hybrid
    from src.solvers.sa_solver import SAConfig
    from src.viz.plot_routes import plot_clusters, plot_single_route

    output_cfg = cfg["output"]
    solver_cfg = cfg.get("solver", {})
    hybrid_cfg_dict = cfg.get("hybrid", {})
    sa_cfg_dict = cfg.get("sa", {})
    tw_cfg = cfg.get("time_window", {})
    alpha = tw_cfg.get("alpha", 10.0)
    beta = tw_cfg.get("beta", 20.0)

    backend = solver_cfg.get("backend", "hybrid").lower()
    if backend in {"sa", "kaiwu"}:
        sub_solver = backend
    else:
        sub_solver = hybrid_cfg_dict.get("sub_solver", "sa")
    logger.info("Q3: backend=%s, sub_solver=%s, n_customers=%d", backend, sub_solver, graph.n_customers)

    sa_cfg = SAConfig(
        initial_temp=_as_float(sa_cfg_dict.get("initial_temp"), 3000.0),
        cooling_rate=_as_float(sa_cfg_dict.get("cooling_rate"), 0.997),
        min_temp=_as_float(sa_cfg_dict.get("min_temp"), 1e-4),
        n_iter_per_temp=_as_int(sa_cfg_dict.get("n_iter_per_temp"), 500),
        seed=_as_int(sa_cfg_dict.get("seed"), 42),
    )
    h_cfg = HybridConfig(
        cluster_method=hybrid_cfg_dict.get("cluster_method", "kmeans"),
        n_clusters=hybrid_cfg_dict.get("n_clusters", 5),
        sub_solver=sub_solver,
        local_search_iter=hybrid_cfg_dict.get("local_search_iter", 500),
        seed=hybrid_cfg_dict.get("seed", 42),
        sa_cfg=sa_cfg,
    )

    from src.core.time_window import simulate_route_timing

    def cost_fn(perm: list[int]) -> float:
        timing = simulate_route_timing(perm, graph, alpha, beta)
        return timing.total_travel_time + timing.total_penalty

    result = solve_hybrid(graph, cost_fn, h_cfg)
    route = result.global_route

    metrics = single_route_metrics(route, graph, alpha=alpha, beta=beta)
    logger.info(
        "Q3 result: obj=%.4f (travel=%.4f, penalty=%.4f)",
        metrics["objective"],
        metrics["total_travel_time"],
        metrics["total_penalty"],
    )

    result_dir = Path(output_cfg.get("result_dir", "outputs/results"))
    result_csv = result_dir / "q3_result.csv"
    save_metrics_csv(metrics, result_csv)
    _append_single_result_summary_csv(result_csv, metrics)
    plot_single_route(
        route,
        graph,
        title=f"Q3 Route (obj={metrics['objective']:.2f})",
        save_path=Path(output_cfg["figure_dir"]) / "q3_route.png",
    )
    plot_clusters(
        graph,
        result.cluster_labels,
        graph.customer_ids,
        title="Q3 Customer Clusters",
        save_path=Path(output_cfg["figure_dir"]) / "q3_clusters.png",
    )
    _print_single_result(metrics, "Q3")


def run_q4(cfg: dict[str, Any], graph, sensitivity: bool = False) -> None:
    """Run Problem 4 multi-vehicle solver and optional sensitivity analysis."""
    from src.algorithms.vehicle_assignment import assign_customers_to_vehicles
    from src.algorithms.clustering import cluster_customers
    from src.algorithms.local_search import two_opt
    from src.eval.metrics import multi_vehicle_metrics, save_metrics_csv
    from src.eval.sensitivity import run_vehicle_sensitivity
    from src.qubo.q4_qubo import evaluate_q4_solution
    from src.solvers.sa_solver import SAConfig, solve_route_sa
    from src.viz.plot_routes import plot_multi_vehicle_routes
    from src.viz.plot_tradeoff import plot_sensitivity_curve, plot_stacked_cost

    output_cfg = cfg["output"]
    vehicle_cfg = cfg.get("vehicle", {})
    hybrid_cfg_dict = cfg.get("hybrid", {})
    sa_cfg_dict = cfg.get("sa", {})
    tw_cfg = cfg.get("time_window", {})
    alpha = tw_cfg.get("alpha", 10.0)
    beta = tw_cfg.get("beta", 20.0)
    vehicle_capacity = vehicle_cfg.get("capacity", 100.0)

    sa_cfg = SAConfig(
        initial_temp=_as_float(sa_cfg_dict.get("initial_temp"), 5000.0),
        cooling_rate=_as_float(sa_cfg_dict.get("cooling_rate"), 0.997),
        min_temp=_as_float(sa_cfg_dict.get("min_temp"), 1e-4),
        n_iter_per_temp=_as_int(sa_cfg_dict.get("n_iter_per_temp"), 500),
        seed=_as_int(sa_cfg_dict.get("seed"), 42),
    )

    # Cluster and assign vehicles
    n_clusters = hybrid_cfg_dict.get("n_clusters", 5)
    labels, _ = cluster_customers(
        graph,
        graph.customer_ids,
        method=hybrid_cfg_dict.get("cluster_method", "kmeans"),
        n_clusters=n_clusters,
        seed=hybrid_cfg_dict.get("seed", 42),
    )

    customer_groups = assign_customers_to_vehicles(
        graph.customer_ids, graph, vehicle_capacity, cluster_labels=labels
    )

    # Solve each vehicle sub-route
    from src.core.time_window import simulate_route_timing

    def _solve_group(cids: list[int]) -> list[int]:
        def cost_fn(perm: list[int]) -> float:
            timing = simulate_route_timing(perm, graph, alpha, beta)
            return timing.total_travel_time + timing.total_penalty

        result = solve_route_sa(cids, cost_fn, sa_cfg)
        improved = two_opt(result.best_solution, graph)
        return [graph.depot_id] + improved + [graph.depot_id]

    vehicle_routes = [_solve_group(group) for group in customer_groups]

    q4_result = evaluate_q4_solution(vehicle_routes, graph, vehicle_capacity, alpha, beta)
    metrics = multi_vehicle_metrics(vehicle_routes, graph, vehicle_capacity, alpha, beta)

    logger.info(
        "Q4 result: K=%d, obj=%.4f (travel=%.4f, penalty=%.4f), capacity_ok=%s",
        metrics["n_vehicles"],
        metrics["objective"],
        metrics["total_travel_time"],
        metrics["total_penalty"],
        q4_result.capacity_report.feasible if q4_result.capacity_report else "N/A",
    )

    result_dir = Path(output_cfg.get("result_dir", "outputs/results"))
    save_metrics_csv(metrics, result_dir / "q4_result.csv")
    plot_multi_vehicle_routes(
        vehicle_routes,
        graph,
        title=f"Q4 Routes (K={metrics['n_vehicles']}, obj={metrics['objective']:.2f})",
        save_path=Path(output_cfg["figure_dir"]) / "q4_routes.png",
    )
    plot_stacked_cost(
        metrics["vehicles"],
        save_path=Path(output_cfg["figure_dir"]) / "q4_cost_breakdown.png",
    )

    _print_multi_result(metrics, "Q4")

    # Sensitivity analysis
    if sensitivity:
        sens_cfg = cfg.get("sensitivity", {})
        min_k = sens_cfg.get("min_vehicles", 2)
        max_k = sens_cfg.get("max_vehicles", 10)
        step = sens_cfg.get("step", 1)

        def solve_fn_for_k(g, k: int, cap: float) -> list[list[int]]:
            lbl, _ = cluster_customers(g, g.customer_ids, n_clusters=k, seed=42)
            groups = assign_customers_to_vehicles(g.customer_ids, g, cap, cluster_labels=lbl)
            return [_solve_group(grp) for grp in groups]

        sens_result = run_vehicle_sensitivity(
            graph,
            solve_fn_for_k,
            vehicle_capacity,
            alpha=alpha,
            beta=beta,
            min_vehicles=min_k,
            max_vehicles=max_k,
            step=step,
        )
        prescreen_dir = Path(output_cfg.get("prescreen_dir", "outputs/prescreen"))
        sens_result.save_csv(prescreen_dir / "q4_sensitivity.csv")
        plot_sensitivity_curve(
            sens_result.to_dataframe(),
            save_path=Path(output_cfg["figure_dir"]) / "q4_sensitivity.png",
        )
        logger.info("Sensitivity analysis complete. Best K=%d", sens_result.best_k)


def export_qubo_phase(cfg: dict[str, Any], graph) -> Path:
    """Export QUBO matrix to CSV for external/real-machine solving.

    Supported problems: q1, q2, q3, q4.
    For q3/q4, exports decomposed sub-problem matrices and returns a manifest path.
    """
    problem = str(cfg.get("problem", "q1")).lower()
    qubo_cfg = cfg.get("qubo", {})
    output_cfg = cfg.get("output", {})
    qubo_dir = Path(output_cfg.get("qubo_dir", "outputs/qubo_ising"))
    qubo_dir.mkdir(parents=True, exist_ok=True)

    export_cfg = cfg.get("qubo_export", {})
    output_model = str(export_cfg.get("output_model", "qubo")).lower()
    if output_model not in {"qubo", "ising"}:
        logger.warning("Unknown output_model=%s, fallback to 'qubo'.", output_model)
        output_model = "qubo"

    def _export_single_matrix(
        qr,
        problem_tag: str,
        out_name: str,
        raw_name: str,
        meta_name: str,
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out_path = qubo_dir / out_name
        raw_path = qubo_dir / raw_name
        meta_path = qubo_dir / meta_name

        np.savetxt(raw_path, qr.Q, delimiter=",", fmt="%.12g")
        q_export, meta = _adapt_qubo_for_8bit(qr.Q, export_cfg)
        np.savetxt(out_path, q_export, delimiter=",", fmt="%d")

        meta_payload: dict[str, Any] = {
            "problem": problem,
            "tag": problem_tag,
            "output_model": meta.get("output_model", output_model),
            "method": meta.get("method", "none"),
            "n_vars_original": int(qr.Q.shape[0]),
            "n_vars_exported": int(q_export.shape[0]),
            "max_abs_original": float(np.max(np.abs(qr.Q))) if qr.Q.size else 0.0,
            "max_abs_exported": float(np.max(np.abs(q_export))) if q_export.size else 0.0,
        }
        if extra_meta:
            meta_payload.update(extra_meta)
        if "split_last_idx" in meta:
            meta_payload["split_last_idx"] = list(meta["split_last_idx"])
        if "ising_aux_index" in meta:
            meta_payload["ising_aux_index"] = int(meta["ising_aux_index"])
        if "decode_supported" in meta:
            meta_payload["decode_supported"] = bool(meta["decode_supported"])

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        logger.info(
            "Exported %s %s matrix (%s): raw=%s, adapted=%s, meta=%s, shape=%s -> %s, max|Q| %.3f -> %.3f",
            problem_tag.upper(),
            str(meta_payload["output_model"]).upper(),
            meta_payload.get("method", "none"),
            raw_path,
            out_path,
            meta_path,
            qr.Q.shape,
            q_export.shape,
            meta_payload["max_abs_original"],
            meta_payload["max_abs_exported"],
        )
        return {
            "tag": problem_tag,
            "raw": str(raw_path),
            "adapted": str(out_path),
            "meta": str(meta_path),
            "n_vars_original": meta_payload["n_vars_original"],
            "n_vars_exported": meta_payload["n_vars_exported"],
        }

    if problem == "q1":
        from src.qubo.q1_qubo import build_q1_qubo

        qr = build_q1_qubo(
            graph,
            penalty_visit=qubo_cfg.get("penalty_visit", 500),
            penalty_position=qubo_cfg.get("penalty_position", 500),
        )
        out_name = "q1_ising.csv" if output_model == "ising" else "q1_qubo.csv"
        meta_name = "q1_ising_meta.json" if output_model == "ising" else "q1_qubo_meta.json"
        _export_single_matrix(
            qr,
            problem_tag="q1",
            out_name=out_name,
            raw_name="q1_qubo_raw.csv",
            meta_name=meta_name,
        )
        return qubo_dir / out_name
    elif problem == "q2":
        from src.qubo.q2_qubo import build_q2_qubo

        qr = build_q2_qubo(
            graph,
            penalty_visit=qubo_cfg.get("penalty_visit", 500),
            penalty_position=qubo_cfg.get("penalty_position", 500),
        )
        out_name = "q2_ising.csv" if output_model == "ising" else "q2_qubo.csv"
        meta_name = "q2_ising_meta.json" if output_model == "ising" else "q2_qubo_meta.json"
        _export_single_matrix(
            qr,
            problem_tag="q2",
            out_name=out_name,
            raw_name="q2_qubo_raw.csv",
            meta_name=meta_name,
        )
        return qubo_dir / out_name
    elif problem == "q3":
        from src.algorithms.clustering import cluster_customers
        from src.core.graph_model import subgraph
        from src.qubo.q1_qubo import build_q1_qubo

        hybrid_cfg = cfg.get("hybrid", {})
        labels, _ = cluster_customers(
            graph,
            graph.customer_ids,
            method=hybrid_cfg.get("cluster_method", "kmeans"),
            n_clusters=_as_int(hybrid_cfg.get("n_clusters"), 5),
            seed=_as_int(hybrid_cfg.get("seed"), 42),
        )

        cluster_map: dict[int, list[int]] = {}
        for cid, lbl in zip(graph.customer_ids, labels):
            cluster_map.setdefault(int(lbl), []).append(int(cid))

        entries: list[dict[str, Any]] = []
        for cluster_id, cids in sorted(cluster_map.items()):
            sub = subgraph(graph, cids)
            qr = build_q1_qubo(
                sub,
                penalty_visit=qubo_cfg.get("penalty_visit", 500),
                penalty_position=qubo_cfg.get("penalty_position", 500),
            )
            suffix = "ising" if output_model == "ising" else "qubo"
            tag = f"q3_cluster_{cluster_id:02d}"
            entries.append(
                _export_single_matrix(
                    qr,
                    problem_tag=tag,
                    out_name=f"{tag}_{suffix}.csv",
                    raw_name=f"{tag}_qubo_raw.csv",
                    meta_name=f"{tag}_{suffix}_meta.json",
                    extra_meta={
                        "cluster_id": int(cluster_id),
                        "n_customers_sub": int(len(cids)),
                        "customer_ids": [int(x) for x in cids],
                    },
                )
            )

        manifest = {
            "problem": "q3",
            "type": "decomposed_clusters",
            "output_model": output_model,
            "n_parts": len(entries),
            "entries": entries,
        }
        manifest_path = qubo_dir / "q3_export_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info("Q3 export complete: %d cluster matrices, manifest=%s", len(entries), manifest_path)
        return manifest_path
    elif problem == "q4":
        from src.algorithms.clustering import cluster_customers
        from src.algorithms.vehicle_assignment import assign_customers_to_vehicles
        from src.qubo.q4_qubo import build_vehicle_qubo

        hybrid_cfg = cfg.get("hybrid", {})
        vehicle_cfg = cfg.get("vehicle", {})
        labels, _ = cluster_customers(
            graph,
            graph.customer_ids,
            method=hybrid_cfg.get("cluster_method", "kmeans"),
            n_clusters=_as_int(hybrid_cfg.get("n_clusters"), 5),
            seed=_as_int(hybrid_cfg.get("seed"), 42),
        )
        groups = assign_customers_to_vehicles(
            graph.customer_ids,
            graph,
            _as_float(vehicle_cfg.get("capacity"), 100.0),
            cluster_labels=labels,
            seed=_as_int(hybrid_cfg.get("seed"), 42),
        )

        entries: list[dict[str, Any]] = []
        suffix = "ising" if output_model == "ising" else "qubo"
        for idx, cids in enumerate(groups, start=1):
            qr, _ = build_vehicle_qubo(
                graph,
                cids,
                penalty_visit=qubo_cfg.get("penalty_visit", 500),
                penalty_position=qubo_cfg.get("penalty_position", 500),
            )
            tag = f"q4_vehicle_{idx:02d}"
            entries.append(
                _export_single_matrix(
                    qr,
                    problem_tag=tag,
                    out_name=f"{tag}_{suffix}.csv",
                    raw_name=f"{tag}_qubo_raw.csv",
                    meta_name=f"{tag}_{suffix}_meta.json",
                    extra_meta={
                        "vehicle_index": int(idx),
                        "n_customers_sub": int(len(cids)),
                        "customer_ids": [int(x) for x in cids],
                    },
                )
            )

        manifest = {
            "problem": "q4",
            "type": "decomposed_vehicles",
            "output_model": output_model,
            "n_parts": len(entries),
            "entries": entries,
        }
        manifest_path = qubo_dir / "q4_export_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info("Q4 export complete: %d vehicle matrices, manifest=%s", len(entries), manifest_path)
        return manifest_path
    else:
        raise ValueError("QUBO export supports q1/q2/q3/q4 only.")


def _load_solution_vector(
    solution_path: str | Path,
    n_vars: int,
    meta_path: str | Path | None = None,
) -> np.ndarray:
    """Load external binary solution vector from txt/csv/bitstring files.

    Accepted formats:
    1) comma/space/newline separated 0/1 values
    2) compact bitstring, e.g. 010011...
    """
    p = Path(solution_path)
    raw = p.read_text(encoding="utf-8").strip()

    # Platform log JSON mode: [{"quboValue": ..., "solutionVector": [...]}, ...]
    parsed_from_log = _try_parse_platform_log_solution(raw)
    if parsed_from_log is not None:
        return _normalize_solution_vector(parsed_from_log, n_vars, meta_path)

    # Compact bitstring mode
    compact = "".join(ch for ch in raw if ch in {"0", "1"})
    if compact and len(compact) == n_vars and set(compact) <= {"0", "1"}:
        return np.array([1.0 if ch == "1" else 0.0 for ch in compact], dtype=float)

    # Delimited numeric mode
    tokenized = raw.replace(",", " ").replace("\n", " ").split()
    try:
        arr = np.array([float(t) for t in tokenized], dtype=float)
    except ValueError as exc:
        raise ValueError(f"Cannot parse solution file: {p}") from exc
    return _normalize_solution_vector(arr, n_vars, meta_path)


def _load_solution_vectors_from_log(
    solution_path: str | Path,
    n_vars: int,
    meta_path: str | Path | None = None,
) -> list[np.ndarray] | None:
    """Load all solution vectors from platform JSON log and normalize them.

    Returns None when the input file is not a platform JSON log.
    """
    raw = Path(solution_path).read_text(encoding="utf-8").strip()
    vectors = _try_parse_platform_log_solutions(raw)
    if vectors is None:
        return None

    normalized: list[np.ndarray] = []
    for vec in vectors:
        try:
            normalized.append(_normalize_solution_vector(vec, n_vars, meta_path))
        except ValueError:
            continue
    return normalized if normalized else None


def _normalize_solution_vector(
    arr: np.ndarray,
    n_vars: int,
    meta_path: str | Path | None,
) -> np.ndarray:
    """Normalize external vector to binary QUBO vector with target length n_vars."""
    vec = np.asarray(arr, dtype=float).reshape(-1)

    vec = _maybe_restore_ising_aux_solution(vec, n_vars, meta_path)
    if vec.size != n_vars:
        vec = _maybe_restore_split_solution(vec, n_vars, meta_path)
    if vec.size != n_vars:
        raise ValueError(f"Solution length mismatch: expected {n_vars}, got {vec.size}")

    vec = _ising_spin_to_binary_if_needed(vec)
    return (vec >= 0.5).astype(float)


def _try_parse_platform_log_solution(raw: str) -> np.ndarray | None:
    """Try parse platform log JSON and return one solution vector.

    Selection rule:
    1) prefer non-all-zero solution with minimum quboValue
    2) if all are zero vectors, use global minimum quboValue entry
    """
    if not raw:
        return None
    first = raw[0]
    if first not in {"[", "{"}:
        return None

    try:
        data = json.loads(raw)
    except Exception:
        return None

    vectors = _try_parse_platform_log_solutions(raw)
    if not vectors:
        return None

    # Keep legacy single-vector behavior for callers that expect one vector.
    records: list[dict[str, Any]] = []
    if isinstance(data, list):
        records = [r for r in data if isinstance(r, dict) and "solutionVector" in r]
    elif isinstance(data, dict) and "solutionVector" in data:
        records = [data]

    def _qubo_value(rec: dict[str, Any]) -> float:
        try:
            if "quboValue" in rec:
                return float(rec.get("quboValue", float("inf")))
            if "Hamiltonian" in rec:
                return float(rec.get("Hamiltonian", float("inf")))
            if "hamiltonian" in rec:
                return float(rec.get("hamiltonian", float("inf")))
            return float("inf")
        except Exception:
            return float("inf")

    def _vector(rec: dict[str, Any]) -> np.ndarray:
        v = rec.get("solutionVector", [])
        return np.asarray(v, dtype=float).reshape(-1)

    non_zero = [r for r in records if np.any(_vector(r) >= 0.5)]
    chosen = min(non_zero or records, key=_qubo_value)
    return _vector(chosen)


def _try_parse_platform_log_solutions(raw: str) -> list[np.ndarray] | None:
    """Parse all solution vectors from platform JSON log payload, if present."""
    if not raw:
        return None
    first = raw[0]
    if first not in {"[", "{"}:
        return None

    try:
        data = json.loads(raw)
    except Exception:
        return None

    records: list[dict[str, Any]] = []
    if isinstance(data, list):
        records = [r for r in data if isinstance(r, dict) and "solutionVector" in r]
    elif isinstance(data, dict) and "solutionVector" in data:
        records = [data]
    else:
        return None

    vecs: list[np.ndarray] = []
    for rec in records:
        v = np.asarray(rec.get("solutionVector", []), dtype=float).reshape(-1)
        if v.size > 0:
            vecs.append(v)
    return vecs or None


def _try_parse_platform_log_solutions(raw: str) -> list[np.ndarray] | None:
    """Parse all solution vectors from platform JSON log content."""
    if not raw:
        return None
    first = raw[0]
    if first not in {"[", "{"}:
        return None

    try:
        data = json.loads(raw)
    except Exception:
        return None

    records: list[dict[str, Any]] = []
    if isinstance(data, list):
        records = [r for r in data if isinstance(r, dict) and "solutionVector" in r]
    elif isinstance(data, dict) and "solutionVector" in data:
        records = [data]
    else:
        return None

    if not records:
        return None
    return [np.asarray(r.get("solutionVector", []), dtype=float).reshape(-1) for r in records]


def _ising_spin_to_binary_if_needed(arr: np.ndarray) -> np.ndarray:
    """Convert Ising spins {-1,+1} to binary {0,1} when detected."""
    a = np.asarray(arr, dtype=float).reshape(-1)
    if a.size == 0:
        return a
    uniq = np.unique(a)
    if np.all(np.isin(uniq, [-1.0, 1.0])):
        return (a + 1.0) / 2.0
    return a


def _maybe_restore_ising_aux_solution(
    arr: np.ndarray,
    n_vars: int,
    meta_path: str | Path | None,
) -> np.ndarray:
    """Drop auxiliary Ising variable when platform returns n+1 spins.

    Priority:
    1) use meta.ising_aux_index when provided
    2) heuristic: for size n_vars+1, drop the last bit (our export convention)
    """
    a = np.asarray(arr, dtype=float).reshape(-1)
    if a.size == n_vars:
        return a
    if a.size != n_vars + 1:
        return a

    if meta_path is not None:
        p = Path(meta_path)
        if p.exists():
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
                idx = meta.get("ising_aux_index")
                if idx is not None:
                    idx_i = int(idx)
                    if 0 <= idx_i < a.size:
                        return np.delete(a, idx_i)
            except Exception:
                pass

    # Default auxiliary index convention from export path: appended at tail.
    return a[:-1]


def _maybe_restore_split_solution(
    arr: np.ndarray,
    n_vars: int,
    meta_path: str | Path | None,
) -> np.ndarray:
    """Restore split-adapted solution back to original variable size when possible."""
    if meta_path is None:
        return arr
    p = Path(meta_path)
    if not p.exists():
        return arr

    meta = json.loads(p.read_text(encoding="utf-8"))
    if meta.get("method") != "split":
        return arr
    if int(meta.get("n_vars_exported", -1)) != int(arr.size):
        return arr

    split_last_idx = meta.get("split_last_idx")
    if split_last_idx is None:
        return arr

    try:
        import kaiwu as kw

        restored = kw.preprocess.restore_splitted_solution(arr.astype(float), np.array(split_last_idx, dtype=int))
        restored = np.asarray(restored, dtype=float).reshape(-1)
        if restored.size == n_vars:
            return restored
    except Exception as exc:
        logger.warning("Failed to restore split solution with Kaiwu preprocess: %s", exc)
    return arr


def _adapt_qubo_for_8bit(
    q: np.ndarray,
    export_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Adapt QUBO matrix for 8-bit integer hardware constraints.

    Supported methods:
    - none: no adaptation, only integer clipping
    - truncate: Kaiwu direct precision adjustment
    - mutate: Kaiwu dynamic-range mutation adaptation
    - split: Kaiwu variable split (increases variable count)
    """
    method = str(export_cfg.get("precision_method", "truncate")).lower()
    output_model = str(export_cfg.get("output_model", "qubo")).lower()
    target_bits = _as_int(export_cfg.get("target_bits"), 8)
    int_limit = max(1, 2 ** (target_bits - 1) - 1)
    q_work = np.asarray(q, dtype=float)
    meta: dict[str, Any] = {"method": method, "output_model": output_model}

    try:
        import kaiwu as kw

        def _call_candidates(candidates: list[str], *args):
            """Try candidate dotted call paths in order and return first success."""
            errs: list[str] = []
            for path in candidates:
                try:
                    obj = kw
                    for part in path.split("."):
                        obj = getattr(obj, part)
                    return obj(*args)
                except Exception as exc:  # keep trying next API location
                    errs.append(f"{path}: {exc}")
            if errs:
                raise RuntimeError("; ".join(errs))
            raise RuntimeError("No callable precision-adaptation candidates found.")

        if method == "none":
            q_adapt = q_work.copy()
        elif method == "truncate":
            q_adapt = _call_candidates(
                [
                    "qubo.adjust_qubo_matrix_precision",
                    "preprocess.adjust_qubo_matrix_precision",
                ],
                q_work,
            )
        elif method == "mutate":
            if output_model == "ising":
                # User explicitly wants Ising output for platform upload.
                q_adapt, aux_meta = _adapt_qubo_mutate_via_ising_aux(
                    q_work,
                    kw,
                    return_ising=True,
                )
                meta.update(aux_meta)
                meta["method"] = "mutate-via-ising-aux"
                meta["decode_supported"] = False
            else:
                try:
                    q_adapt = _call_candidates(
                        [
                            "preprocess.perform_precision_adaption_mutate",
                            "qubo.perform_precision_adaption_mutate",
                        ],
                        q_work,
                    )
                except Exception as exc:
                    logger.warning(
                        "Mutate adaptation unavailable for current SDK/matrix (%s). "
                        "Trying QUBO->Ising(aux)->mutate->QUBO path.",
                        exc,
                    )
                    try:
                        q_adapt, aux_meta = _adapt_qubo_mutate_via_ising_aux(
                            q_work,
                            kw,
                            return_ising=False,
                        )
                        meta.update(aux_meta)
                        meta["method"] = "mutate-via-ising-aux"
                    except Exception as aux_exc:
                        logger.warning(
                            "Mutate via Ising(aux) failed (%s). Falling back to Kaiwu truncate.",
                            aux_exc,
                        )
                        q_adapt = _call_candidates(
                            [
                                "qubo.adjust_qubo_matrix_precision",
                                "preprocess.adjust_qubo_matrix_precision",
                            ],
                            q_work,
                        )
                        meta["method"] = "mutate-fallback-truncate"
        elif method == "split":
            split_ret, last_idx = _call_candidates(
                [
                    "preprocess.perform_precision_adaption_split",
                    "qubo.perform_precision_adaption_split",
                ],
                q_work,
                target_bits,
            )
            q_adapt = np.asarray(split_ret, dtype=float)
            meta["split_last_idx"] = np.asarray(last_idx, dtype=int).tolist()
        else:
            raise ValueError(f"Unknown precision_method: {method}")
    except Exception as exc:
        logger.warning(
            "Precision adaptation via Kaiwu failed (%s). Falling back to linear scaling.",
            exc,
        )
        max_abs = float(np.max(np.abs(q_work))) if q_work.size else 0.0
        if max_abs > 0:
            q_adapt = q_work / max_abs * int_limit
        else:
            q_adapt = q_work.copy()
        meta["method"] = "fallback-linear"

    # Final integer projection for hardware upload compatibility.
    q_int = np.rint(q_adapt)
    q_int = np.clip(q_int, -int_limit - 1, int_limit)
    return q_int.astype(int), meta


def _adapt_qubo_mutate_via_ising_aux(
    q: np.ndarray,
    kw: Any,
    return_ising: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply Kaiwu mutate by converting QUBO -> Ising(with auxiliary spin) -> QUBO.

    This path is used when the SDK does not expose `qubo.perform_precision_adaption_mutate`
    but does expose `preprocess.perform_precision_adaption_mutate` for zero-diagonal Ising matrices.
    """
    q_upper = np.triu(np.asarray(q, dtype=float))
    n = q_upper.shape[0]
    if q_upper.ndim != 2 or n != q_upper.shape[1]:
        raise ValueError(f"Q must be square, got {q_upper.shape}")

    # Convert QUBO coefficients to Ising h/J (without constant term).
    j_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            j_mat[i, j] = q_upper[i, j] / 4.0
            j_mat[j, i] = j_mat[i, j]

    h = np.zeros(n, dtype=float)
    for i in range(n):
        h[i] = q_upper[i, i] / 2.0 + 0.25 * (
            float(np.sum(q_upper[i, i + 1 :])) + float(np.sum(q_upper[:i, i]))
        )

    # Build auxiliary Ising matrix with zero diagonal:
    # E = sum_{i<j} J_ij s_i s_j + sum_i h_i s_i z, z in {-1, +1}
    # This removes linear terms for CIM-style zero-diagonal Ising representations.
    k = np.zeros((n + 1, n + 1), dtype=float)
    k[:n, :n] = j_mat
    k[:n, n] = h
    k[n, :n] = h
    np.fill_diagonal(k, 0.0)

    k_mut = kw.preprocess.perform_precision_adaption_mutate(k)
    k_mut = np.asarray(k_mut, dtype=float)
    if k_mut.shape != (n + 1, n + 1):
        raise ValueError(f"Unexpected mutated Ising shape: {k_mut.shape}")

    if return_ising:
        return k_mut, {"output_model": "ising", "ising_aux_index": n}

    j_mut = 0.5 * (k_mut[:n, :n] + k_mut[:n, :n].T)
    np.fill_diagonal(j_mut, 0.0)
    h_mut = 0.5 * (k_mut[:n, n] + k_mut[n, :n])

    # Convert Ising h/J back to upper-triangular QUBO matrix.
    q_back = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            q_back[i, j] = 4.0 * j_mut[i, j]

    for i in range(n):
        q_back[i, i] = 2.0 * h_mut[i] - 2.0 * float(np.sum(j_mut[i, :]))

    return np.triu(q_back), {"output_model": "qubo", "ising_aux_index": n}


def run_q1_from_solution(cfg: dict[str, Any], graph, solution_path: str) -> None:
    """Decode an external Q1 solution vector and run standard evaluation/output."""
    from src.algorithms.local_search import two_opt
    from src.eval.metrics import save_metrics_csv, single_route_metrics
    from src.qubo.q1_qubo import build_q1_qubo, decode_q1_solution
    from src.viz.plot_routes import plot_single_route

    output_cfg = cfg["output"]
    qubo_cfg = cfg.get("qubo", {})
    qr = build_q1_qubo(
        graph,
        penalty_visit=qubo_cfg.get("penalty_visit", 500),
        penalty_position=qubo_cfg.get("penalty_position", 500),
    )
    qubo_dir = Path(output_cfg.get("qubo_dir", "outputs/qubo_ising"))
    meta_path = qubo_dir / "q1_ising_meta.json"
    if not meta_path.exists():
        meta_path = qubo_dir / "q1_qubo_meta.json"
    log_vectors = _load_solution_vectors_from_log(solution_path, qr.n_vars, meta_path)
    if log_vectors:
        best_route: list[int] | None = None
        best_travel = float("inf")
        for x in log_vectors:
            cand = decode_q1_solution(x, qr.n_nodes, qr.var_idx)
            customers = [n for n in cand if n != graph.depot_id]
            customers = two_opt(customers, graph)
            cand = [graph.depot_id] + customers + [graph.depot_id]
            travel = graph.route_travel_time(cand)
            if travel < best_travel:
                best_travel = travel
                best_route = cand
        if best_route is None:
            raise RuntimeError("No decodable candidate vectors found in platform log.")
        route = best_route
        logger.info("Q1 external-solution candidate scan: n=%d, best_travel=%.4f", len(log_vectors), best_travel)
    else:
        p = Path(solution_path)
        raw = p.read_text(encoding="utf-8").strip()
        candidates = _try_parse_platform_log_solutions(raw)

        if candidates:
            best_route: list[int] | None = None
            best_cost = float("inf")
            for vec in candidates:
                x = _normalize_solution_vector(vec, qr.n_vars, meta_path)
                route = decode_q1_solution(x, qr.n_nodes, qr.var_idx)
                customers = [n for n in route if n != graph.depot_id]
                customers = two_opt(customers, graph)
                route = [graph.depot_id] + customers + [graph.depot_id]
                travel = graph.route_travel_time(route)
                if travel < best_cost:
                    best_cost = travel
                    best_route = route
            if best_route is None:
                raise RuntimeError("No valid route decoded from platform log solutions.")
            route = best_route
            logger.info("Q1 external-solution: selected best decoded sample among %d candidates.", len(candidates))
        else:
            x = _load_solution_vector(solution_path, qr.n_vars, meta_path)
            route = decode_q1_solution(x, qr.n_nodes, qr.var_idx)
            customers = [n for n in route if n != graph.depot_id]
            customers = two_opt(customers, graph)
            route = [graph.depot_id] + customers + [graph.depot_id]

    metrics = single_route_metrics(route, graph, alpha=0.0, beta=0.0)
    logger.info(
        "Q1 external-solution result: route=%s, total_travel_time=%.4f",
        metrics["route"],
        metrics["total_travel_time"],
    )
    result_csv, route_fig = _result_paths(output_cfg, "q1", "cpqc550")
    save_metrics_csv(metrics, result_csv)
    _append_single_result_summary_csv(result_csv, metrics)
    plot_single_route(
        route,
        graph,
        title=f"Q1 Route (travel={metrics['total_travel_time']:.2f})",
        save_path=route_fig,
    )
    _print_single_result(metrics, "Q1")


def run_q2_from_solution(cfg: dict[str, Any], graph, solution_path: str) -> None:
    """Decode an external Q2 solution vector and run standard evaluation/output."""
    from src.algorithms.local_search import two_opt
    from src.eval.metrics import save_metrics_csv, single_route_metrics
    from src.qubo.q1_qubo import decode_q1_solution
    from src.qubo.q2_qubo import build_q2_qubo
    from src.viz.plot_routes import plot_single_route

    output_cfg = cfg["output"]
    qubo_cfg = cfg.get("qubo", {})
    tw_cfg = cfg.get("time_window", {})
    alpha = tw_cfg.get("alpha", 10.0)
    beta = tw_cfg.get("beta", 20.0)

    qr = build_q2_qubo(
        graph,
        penalty_visit=qubo_cfg.get("penalty_visit", 500),
        penalty_position=qubo_cfg.get("penalty_position", 500),
    )
    qubo_dir = Path(output_cfg.get("qubo_dir", "outputs/qubo_ising"))
    meta_path = qubo_dir / "q2_ising_meta.json"
    if not meta_path.exists():
        meta_path = qubo_dir / "q2_qubo_meta.json"
    log_vectors = _load_solution_vectors_from_log(solution_path, qr.n_vars, meta_path)
    if log_vectors:
        best_route: list[int] | None = None
        best_obj = float("inf")
        for x in log_vectors:
            cand = decode_q1_solution(x, qr.n_nodes, qr.var_idx)
            customers = [n for n in cand if n != graph.depot_id]
            customers = two_opt(customers, graph)
            cand = [graph.depot_id] + customers + [graph.depot_id]
            m = single_route_metrics(cand, graph, alpha=alpha, beta=beta)
            if m["objective"] < best_obj:
                best_obj = float(m["objective"])
                best_route = cand
        if best_route is None:
            raise RuntimeError("No decodable candidate vectors found in platform log.")
        route = best_route
        logger.info("Q2 external-solution candidate scan: n=%d, best_obj=%.4f", len(log_vectors), best_obj)
    else:
        p = Path(solution_path)
        raw = p.read_text(encoding="utf-8").strip()
        candidates = _try_parse_platform_log_solutions(raw)

        if candidates:
            best_route: list[int] | None = None
            best_obj = float("inf")
            for vec in candidates:
                x = _normalize_solution_vector(vec, qr.n_vars, meta_path)
                route = decode_q1_solution(x, qr.n_nodes, qr.var_idx)
                customers = [n for n in route if n != graph.depot_id]
                customers = two_opt(customers, graph)
                route = [graph.depot_id] + customers + [graph.depot_id]
                m = single_route_metrics(route, graph, alpha=alpha, beta=beta)
                if m["objective"] < best_obj:
                    best_obj = m["objective"]
                    best_route = route
            if best_route is None:
                raise RuntimeError("No valid route decoded from platform log solutions.")
            route = best_route
            logger.info("Q2 external-solution: selected best decoded sample among %d candidates.", len(candidates))
        else:
            x = _load_solution_vector(solution_path, qr.n_vars, meta_path)
            route = decode_q1_solution(x, qr.n_nodes, qr.var_idx)
            customers = [n for n in route if n != graph.depot_id]
            customers = two_opt(customers, graph)
            route = [graph.depot_id] + customers + [graph.depot_id]

    metrics = single_route_metrics(route, graph, alpha=alpha, beta=beta)
    logger.info(
        "Q2 external-solution result: obj=%.4f (travel=%.4f, penalty=%.4f)",
        metrics["objective"],
        metrics["total_travel_time"],
        metrics["total_penalty"],
    )
    result_csv, route_fig = _result_paths(output_cfg, "q2", "cpqc550")
    save_metrics_csv(metrics, result_csv)
    _append_single_result_summary_csv(result_csv, metrics)
    plot_single_route(
        route,
        graph,
        title=f"Q2 Route (obj={metrics['objective']:.2f})",
        save_path=route_fig,
    )
    _print_single_result(metrics, "Q2")


def _collect_q3_solution_files(solution_path: str | Path, n_parts: int) -> list[Path]:
    """Collect Q3 feedback files for decomposed sub-problems.

    Accepts either:
    - a directory containing q3_run_*.log/json/txt/csv files
    - one file inside such a directory (auto-collect siblings when n_parts>1)
    """
    p = Path(solution_path)

    def _glob_runs(folder: Path) -> list[Path]:
        files: list[Path] = []
        for pat in ("q3_run_*.log", "q3_run_*.json", "q3_run_*.txt", "q3_run_*.csv"):
            files.extend(sorted(folder.glob(pat)))
        # de-duplicate while preserving sorted order by name
        uniq: list[Path] = []
        seen: set[Path] = set()
        for fp in sorted(files):
            if fp not in seen:
                uniq.append(fp)
                seen.add(fp)
        return uniq

    if p.is_dir():
        candidates = _glob_runs(p)
    else:
        if n_parts <= 1:
            return [p]
        candidates = _glob_runs(p.parent)
        if p not in candidates:
            candidates = [p] + candidates

    if len(candidates) < n_parts:
        raise RuntimeError(
            f"Insufficient Q3 feedback files: need {n_parts}, found {len(candidates)}. "
            f"Place q3_run_*.log under {p if p.is_dir() else p.parent}."
        )
    if len(candidates) > n_parts:
        logger.warning(
            "Q3 feedback files exceed required parts (%d > %d); using first %d files.",
            len(candidates),
            n_parts,
            n_parts,
        )
    return candidates[:n_parts]


def run_q3_from_solution(cfg: dict[str, Any], graph, solution_path: str) -> None:
    """Decode external Q3 decomposed solutions and evaluate full-route metrics."""
    from src.algorithms.local_search import or_opt, two_opt
    from src.algorithms.route_decode import decode_sub_route
    from src.core.graph_model import subgraph
    from src.core.time_window import simulate_route_timing
    from src.eval.metrics import save_metrics_csv, single_route_metrics
    from src.qubo.q1_qubo import build_q1_qubo
    from src.solvers.hybrid_large_scale import _stitch_routes
    from src.viz.plot_routes import plot_clusters, plot_single_route

    output_cfg = cfg["output"]
    qubo_cfg = cfg.get("qubo", {})
    tw_cfg = cfg.get("time_window", {})
    hybrid_cfg = cfg.get("hybrid", {})
    alpha = tw_cfg.get("alpha", 10.0)
    beta = tw_cfg.get("beta", 20.0)
    local_iter = _as_int(hybrid_cfg.get("local_search_iter"), 500)

    qubo_dir = Path(output_cfg.get("qubo_dir", "outputs/qubo_ising"))
    manifest_path = qubo_dir / "q3_export_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"Q3 manifest not found: {manifest_path}. Run --phase export for q3 first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"Q3 manifest has no entries: {manifest_path}")

    solution_files = _collect_q3_solution_files(solution_path, len(entries))
    logger.info("Q3 external-solution: using %d feedback files.", len(solution_files))

    def _obj(customer_perm: list[int]) -> float:
        route = [graph.depot_id] + list(customer_perm) + [graph.depot_id]
        timing = simulate_route_timing(route, graph, alpha=alpha, beta=beta)
        return timing.total_travel_time + timing.total_penalty

    cluster_routes: list[list[int]] = []
    label_map: dict[int, int] = {}

    for i, (entry, sol_file) in enumerate(zip(entries, solution_files), start=1):
        meta_path = Path(str(entry.get("meta", "")))
        if not meta_path.is_absolute():
            meta_path = (Path.cwd() / meta_path).resolve()
        if not meta_path.exists():
            raise RuntimeError(f"Q3 subproblem meta not found: {meta_path}")

        sub_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        customer_ids = [int(x) for x in sub_meta.get("customer_ids", [])]
        if not customer_ids:
            raise RuntimeError(f"Missing customer_ids in meta: {meta_path}")

        cluster_id = int(sub_meta.get("cluster_id", i - 1))
        for cid in customer_ids:
            label_map[cid] = cluster_id

        sub = subgraph(graph, customer_ids)
        qr = build_q1_qubo(
            sub,
            penalty_visit=qubo_cfg.get("penalty_visit", 500),
            penalty_position=qubo_cfg.get("penalty_position", 500),
        )

        vectors = _load_solution_vectors_from_log(sol_file, qr.n_vars, meta_path)
        if vectors is None:
            vectors = [_load_solution_vector(sol_file, qr.n_vars, meta_path)]

        best_sub: list[int] | None = None
        best_cost = float("inf")
        for vec in vectors:
            cand = decode_sub_route(vec, sub, qr)
            if len(cand) >= 3:
                improved = two_opt(cand, graph, n_iter=max(100, local_iter // 5))
                if _obj(improved) < _obj(cand):
                    cand = improved
            c = _obj(cand)
            if c < best_cost:
                best_cost = c
                best_sub = cand

        if best_sub is None:
            raise RuntimeError(f"No decodable candidate found for subproblem {i} from {sol_file}")
        cluster_routes.append(best_sub)
        logger.info(
            "Q3 subproblem %d/%d decoded from %s, best_sub_obj=%.4f",
            i,
            len(entries),
            sol_file,
            best_cost,
        )

    stitched = _stitch_routes(cluster_routes, graph)
    best_perm = list(stitched)
    best_obj = _obj(best_perm)

    cand = two_opt(best_perm, graph, n_iter=local_iter)
    c = _obj(cand)
    if c < best_obj:
        best_perm, best_obj = cand, c

    cand = or_opt(best_perm, graph, n_iter=max(1, local_iter // 2))
    c = _obj(cand)
    if c < best_obj:
        best_perm, best_obj = cand, c

    route = [graph.depot_id] + best_perm + [graph.depot_id]
    metrics = single_route_metrics(route, graph, alpha=alpha, beta=beta)
    logger.info(
        "Q3 external-solution result: obj=%.4f (travel=%.4f, penalty=%.4f)",
        metrics["objective"],
        metrics["total_travel_time"],
        metrics["total_penalty"],
    )

    result_csv, route_fig = _result_paths(output_cfg, "q3", "cpqc550")
    save_metrics_csv(metrics, result_csv)
    _append_single_result_summary_csv(result_csv, metrics)
    plot_single_route(
        route,
        graph,
        title=f"Q3 Route (cpqc550, obj={metrics['objective']:.2f})",
        save_path=route_fig,
    )

    labels = np.array([label_map.get(int(cid), -1) for cid in graph.customer_ids], dtype=int)
    plot_clusters(
        graph,
        labels,
        graph.customer_ids,
        title="Q3 Clusters (from export manifest)",
        save_path=Path(output_cfg["figure_dir"]) / "q3_clusters_cpqc550.png",
    )
    _print_single_result(metrics, "Q3")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def _print_single_result(metrics: dict, label: str) -> None:
    click.echo(f"\n{'='*50}")
    click.echo(f"  {label} Results")
    click.echo(f"{'='*50}")
    click.echo(f"  Route           : {metrics['route']}")
    click.echo(f"  Travel time     : {metrics['total_travel_time']:.4f}")
    click.echo(f"  TW penalty      : {metrics['total_penalty']:.4f}")
    click.echo(f"  Objective       : {metrics['objective']:.4f}")
    click.echo(f"  Customers served: {metrics['n_customers']}")
    click.echo(f"{'='*50}\n")


def _print_multi_result(metrics: dict, label: str) -> None:
    click.echo(f"\n{'='*50}")
    click.echo(f"  {label} Results")
    click.echo(f"{'='*50}")
    click.echo(f"  Vehicles        : {metrics['n_vehicles']}")
    click.echo(f"  Total travel    : {metrics['total_travel_time']:.4f}")
    click.echo(f"  Total TW penalty: {metrics['total_penalty']:.4f}")
    click.echo(f"  Objective       : {metrics['objective']:.4f}")
    for v in metrics["vehicles"]:
        click.echo(
            f"    V{v['vehicle_id']}: route_len={v['n_customers']}, "
            f"load={v['load']:.1f} ({v['load_ratio']*100:.0f}%), "
            f"travel={v['travel_time']:.2f}, penalty={v['penalty']:.2f}"
        )
    click.echo(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML configuration file (e.g., configs/q1.yaml).",
)
@click.option(
    "--phase",
    type=click.Choice(["data", "export", "solve", "all"], case_sensitive=False),
    default="all",
    help="Execution phase: 'data', 'export' (QUBO CSV), 'solve', or 'all'.",
)
@click.option(
    "--sensitivity",
    is_flag=True,
    default=False,
    help="Run vehicle-count sensitivity analysis (Q4 only).",
)
@click.option(
    "--solution",
    type=click.Path(exists=True),
    default=None,
    help="Path to external solution file (q1/q2 vector; q3 supports decomposed feedback logs).",
)
def cli(config: str, phase: str, sensitivity: bool, solution: str | None) -> None:
    """MathorCup 2026 — Quantum Logistics Optimisation CLI.

    Run a specific problem using:

        python -m src.main --config configs/q1.yaml
    """
    cfg = load_config(config)
    problem = cfg.get("problem", "q1")
    output_cfg = cfg.get("output", {})

    _setup_logging(
        log_dir=output_cfg.get("log_dir", "outputs/logs"),
        log_level=output_cfg.get("log_level", "INFO"),
        problem=problem,
    )

    logger.info("Starting %s | config=%s | phase=%s", problem.upper(), config, phase)
    click.echo(f"Running problem {problem.upper()} …")

    # Phase: data
    graph, _ = run_data_phase(cfg)

    if phase == "data":
        click.echo("Data phase complete.")
        return

    if phase == "export":
        out_path = export_qubo_phase(cfg, graph)
        click.echo(f"QUBO exported: {out_path}")
        return

    if solution is not None:
        if problem.lower() == "q1":
            run_q1_from_solution(cfg, graph, solution)
        elif problem.lower() == "q2":
            run_q2_from_solution(cfg, graph, solution)
        elif problem.lower() == "q3":
            run_q3_from_solution(cfg, graph, solution)
        else:
            click.echo("--solution is currently supported only for q1/q2/q3.", err=True)
            sys.exit(1)
        logger.info("Problem %s complete (external solution).", problem.upper())
        return

    # Phase: solve
    problem_runners = {
        "q1": lambda: run_q1(cfg, graph),
        "q2": lambda: run_q2(cfg, graph),
        "q3": lambda: run_q3(cfg, graph),
        "q4": lambda: run_q4(cfg, graph, sensitivity=sensitivity),
    }
    runner = problem_runners.get(problem.lower())
    if runner is None:
        click.echo(f"Unknown problem '{problem}'. Choose from q1, q2, q3, q4.", err=True)
        sys.exit(1)
    runner()
    logger.info("Problem %s complete.", problem.upper())


if __name__ == "__main__":
    cli()
