from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean

from src.algorithms.local_search import two_opt
from src.main import _adapt_qubo_for_8bit, load_config, run_data_phase
from src.qubo.q1_qubo import build_q1_qubo, decode_q1_solution
from src.solvers.kaiwu_solver import KaiwuConfig, solve_qubo_kaiwu


@dataclass
class CandidateResult:
    penalty_visit: int
    penalty_position: int
    ratio: float
    scale: float
    best_travel: float
    mean_travel: float
    n_vars_exported: int
    max_abs_exported: float


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def run_search(
    config_path: str,
    base: float,
    ratios: list[float],
    scales: list[float],
    repeats: int,
    topk: int,
) -> tuple[list[CandidateResult], Path, Path]:
    cfg = load_config(config_path)
    graph, _ = run_data_phase(cfg)

    output_cfg = cfg["output"]
    table_dir = Path(output_cfg["table_dir"])
    table_dir.mkdir(parents=True, exist_ok=True)

    kaiwu_cfg_dict = cfg.get("kaiwu", {})
    kaiwu_cfg = KaiwuConfig(
        **{k: v for k, v in kaiwu_cfg_dict.items() if k in KaiwuConfig.__dataclass_fields__}
    )
    base_seed = kaiwu_cfg.seed if kaiwu_cfg.seed is not None else 42

    export_cfg = cfg.get("qubo_export", {})

    results: list[CandidateResult] = []

    for scale in scales:
        for ratio in ratios:
            penalty_visit = max(1, int(round(base * scale * ratio)))
            penalty_position = max(1, int(round(base * scale / ratio)))

            qr = build_q1_qubo(
                graph,
                penalty_visit=penalty_visit,
                penalty_position=penalty_position,
            )

            travels: list[float] = []
            for rep in range(repeats):
                run_cfg = KaiwuConfig(**kaiwu_cfg.__dict__)
                run_cfg.seed = int(base_seed) + rep

                x = solve_qubo_kaiwu(qr.Q, run_cfg)
                route = decode_q1_solution(x, qr.n_nodes, qr.var_idx)
                customers = [n for n in route if n != graph.depot_id]
                customers = two_opt(customers, graph)
                route = [graph.depot_id] + customers + [graph.depot_id]
                travels.append(graph.route_travel_time(route))

            q_export, _meta = _adapt_qubo_for_8bit(qr.Q, export_cfg)

            results.append(
                CandidateResult(
                    penalty_visit=penalty_visit,
                    penalty_position=penalty_position,
                    ratio=ratio,
                    scale=scale,
                    best_travel=min(travels),
                    mean_travel=mean(travels),
                    n_vars_exported=int(q_export.shape[0]),
                    max_abs_exported=float(abs(q_export).max()) if q_export.size else 0.0,
                )
            )

    results.sort(key=lambda r: (r.best_travel, r.mean_travel, abs(r.ratio - 1.0)))

    csv_path = table_dir / "q1_penalty_search.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "penalty_visit",
                "penalty_position",
                "ratio",
                "scale",
                "best_travel",
                "mean_travel",
                "n_vars_exported",
                "max_abs_exported",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.penalty_visit,
                    r.penalty_position,
                    f"{r.ratio:.4f}",
                    f"{r.scale:.4f}",
                    f"{r.best_travel:.4f}",
                    f"{r.mean_travel:.4f}",
                    r.n_vars_exported,
                    f"{r.max_abs_exported:.4f}",
                ]
            )

    md_path = table_dir / "q1_penalty_candidates.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Q1 Penalty Candidates For CPQC-550\n\n")
        f.write("Top candidates screened by local Kaiwu (best_travel then mean_travel).\n\n")
        for idx, r in enumerate(results[:topk], start=1):
            f.write(f"## Candidate {idx}\n")
            f.write(f"- penalty_visit: {r.penalty_visit}\n")
            f.write(f"- penalty_position: {r.penalty_position}\n")
            f.write(f"- ratio (visit/position): {r.ratio:.4f}\n")
            f.write(f"- scale: {r.scale:.4f}\n")
            f.write(f"- best_travel: {r.best_travel:.4f}\n")
            f.write(f"- mean_travel: {r.mean_travel:.4f}\n")
            f.write(f"- exported_vars: {r.n_vars_exported}\n")
            f.write(f"- exported_max_abs: {r.max_abs_exported:.4f}\n\n")

    return results, csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Q1 penalty ratio search using local Kaiwu")
    parser.add_argument("--config", default="configs/q1.yaml")
    parser.add_argument(
        "--base",
        type=float,
        default=0.0,
        help="Penalty center. <=0 means auto center from config qubo penalties.",
    )
    parser.add_argument(
        "--ratios",
        default="0.65,0.8,1.0,1.25,1.55",
        help="visit/position ratio candidates",
    )
    parser.add_argument(
        "--scales",
        default="0.55,0.75,1.0,1.35,1.8",
        help="overall penalty scale candidates around base",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    ratios = _parse_float_list(args.ratios)
    scales = _parse_float_list(args.scales)

    cfg = load_config(args.config)
    qcfg = cfg.get("qubo", {})
    cfg_a = float(qcfg.get("penalty_visit", 700.0))
    cfg_b = float(qcfg.get("penalty_position", 700.0))
    base = float(args.base)
    if base <= 0:
        # Geometric mean is stable when A/B ratio is skewed.
        base = sqrt(max(1.0, cfg_a) * max(1.0, cfg_b))

    results, csv_path, md_path = run_search(
        config_path=args.config,
        base=base,
        ratios=ratios,
        scales=scales,
        repeats=max(1, args.repeats),
        topk=max(1, args.topk),
    )

    best = results[0]
    print("Best candidate:")
    print(
        f"Search center(base)={base:.2f}, grid={len(ratios)} ratios x {len(scales)} scales = {len(ratios)*len(scales)} candidates"
    )
    print(
        f"A={best.penalty_visit}, B={best.penalty_position}, "
        f"best={best.best_travel:.4f}, mean={best.mean_travel:.4f}, "
        f"ratio={best.ratio:.4f}, scale={best.scale:.4f}"
    )
    print(f"Saved full table to: {csv_path}")
    print(f"Saved top candidates to: {md_path}")


if __name__ == "__main__":
    main()
