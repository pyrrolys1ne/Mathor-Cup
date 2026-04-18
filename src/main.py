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

    save_metrics_csv(metrics, Path(output_cfg["table_dir"]) / "q1_result.csv")
    plot_single_route(
        route,
        graph,
        title=f"Q1 Route (travel={metrics['total_travel_time']:.2f})",
        save_path=Path(output_cfg["figure_dir"]) / "q1_route.png",
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
            x = solve_qubo_kaiwu(qr.Q, kw)
            route = decode_q1_solution(x, qr.n_nodes, qr.var_idx)
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

    save_metrics_csv(metrics, Path(output_cfg["table_dir"]) / "q2_result.csv")
    plot_single_route(
        route,
        graph,
        title=f"Q2 Route (obj={metrics['objective']:.2f})",
        save_path=Path(output_cfg["figure_dir"]) / "q2_route.png",
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

    save_metrics_csv(metrics, Path(output_cfg["table_dir"]) / "q3_result.csv")
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

    save_metrics_csv(metrics, Path(output_cfg["table_dir"]) / "q4_result.csv")
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
        sens_result.save_csv(Path(output_cfg["table_dir"]) / "q4_sensitivity.csv")
        plot_sensitivity_curve(
            sens_result.to_dataframe(),
            save_path=Path(output_cfg["figure_dir"]) / "q4_sensitivity.png",
        )
        logger.info("Sensitivity analysis complete. Best K=%d", sens_result.best_k)


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
    type=click.Choice(["data", "solve", "all"], case_sensitive=False),
    default="all",
    help="Execution phase: 'data' (load & validate only), 'solve', or 'all'.",
)
@click.option(
    "--sensitivity",
    is_flag=True,
    default=False,
    help="Run vehicle-count sensitivity analysis (Q4 only).",
)
def cli(config: str, phase: str, sensitivity: bool) -> None:
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
