"""问题四回填目录选择与最优批次选择测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.main import _collect_q4_solution_files, run_q4_from_solution


class _DummyGraph:
    depot_id = 0


def _write_log(path: Path) -> None:
    payload = [{"solutionVector": [1], "quboValue": 0.0}]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_manifest(manifest_path: Path, meta_name: str) -> None:
    manifest = {
        "problem": "q4",
        "entries": [
            {
                "meta": f"not_exist/{meta_name}",
                "adapted": "dummy.csv",
                "raw": "dummy_raw.csv",
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def _base_cfg(mode: str) -> dict:
    return {
        "output": {
            "qubo_dir": "outputs/qubo_ising_temp",
            "result_dir": "outputs/results",
            "figure_dir": "outputs/figures",
        },
        "qubo": {"penalty_visit": 100, "penalty_position": 100},
        "time_window": {"alpha": 10.0, "beta": 20.0},
        "objective": {"alpha": 1.0, "beta": 1.0, "gamma": 0.0},
        "hybrid": {"local_search_iter": 10},
        "vehicle": {"optimization_mode": mode, "capacity": 100.0},
        "data": {},
    }


def test_collect_q4_solution_files_prefers_manifest_selected_vk(tmp_path: Path):
    feedback_root = tmp_path / "platform_feedback"
    d1 = feedback_root / "q4_v06_k05"
    d2 = feedback_root / "q4_v07_k08"
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)

    (d1 / "q4_v06_k05_01.log").write_text("x", encoding="utf-8")
    (d1 / "q4_v06_k05_02.log").write_text("x", encoding="utf-8")
    (d2 / "q4_v07_k08_01.log").write_text("x", encoding="utf-8")
    (d2 / "q4_v07_k08_02.log").write_text("x", encoding="utf-8")

    qubo_dir = tmp_path / "qubo"
    qubo_dir.mkdir(parents=True)
    (qubo_dir / "q4_export_manifest.json").write_text(
        json.dumps({"target_vehicles": 7, "selected_k": 8}, ensure_ascii=False),
        encoding="utf-8",
    )

    files = _collect_q4_solution_files(feedback_root, n_parts=2, qubo_dir=qubo_dir)
    assert len(files) == 2
    assert all("q4_v07_k08" in f.name for f in files)


def test_run_q4_from_solution_lexicographic_prefers_fewer_vehicles(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    qubo_dir = tmp_path / "outputs" / "qubo_ising"
    qubo_dir.mkdir(parents=True)
    _write_manifest(qubo_dir / "q4_export_manifest_v06.json", "meta_v06.json")
    _write_manifest(qubo_dir / "q4_export_manifest_v07.json", "meta_v07.json")
    (qubo_dir / "q4_export_manifest.json").write_text("{}", encoding="utf-8")

    (qubo_dir / "meta_v06.json").write_text(
        json.dumps({"customer_ids": [11]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (qubo_dir / "meta_v07.json").write_text(
        json.dumps({"customer_ids": [99]}, ensure_ascii=False),
        encoding="utf-8",
    )

    feedback = tmp_path / "data" / "platform_feedback"
    b6 = feedback / "q4_v06_k05"
    b7 = feedback / "q4_v07_k07"
    b6.mkdir(parents=True)
    b7.mkdir(parents=True)
    _write_log(b6 / "q4_v06_k05_01.log")
    _write_log(b7 / "q4_v07_k07_01.log")

    def fake_build_vehicle_qubo(graph, customer_ids, penalty_visit, penalty_position):
        return SimpleNamespace(n_vars=1), SimpleNamespace(customer_ids=list(customer_ids))

    def fake_decode_sub_route(vec, sub, qr):
        return list(sub.customer_ids)

    def fake_single_route_metrics(route, graph, alpha, beta):
        return {"objective": float(route[1]) if len(route) > 2 else 0.0}

    def fake_multi_vehicle_metrics(vehicle_routes, graph, capacity, alpha, beta):
        first = vehicle_routes[0][1] if vehicle_routes and len(vehicle_routes[0]) > 2 else -1
        if first == 99:
            return {
                "n_vehicles": 1,
                "total_travel_time": 100.0,
                "total_penalty": 0.0,
                "objective": 100.0,
                "vehicles": [],
            }
        return {
            "n_vehicles": 2,
            "total_travel_time": 10.0,
            "total_penalty": 0.0,
            "objective": 10.0,
            "vehicles": [],
        }

    def fake_save_metrics_csv(metrics, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(str(metrics["n_vehicles"]), encoding="utf-8")

    monkeypatch.setattr("src.qubo.q4_qubo.build_vehicle_qubo", fake_build_vehicle_qubo)
    monkeypatch.setattr("src.algorithms.route_decode.decode_sub_route", fake_decode_sub_route)
    monkeypatch.setattr("src.eval.metrics.single_route_metrics", fake_single_route_metrics)
    monkeypatch.setattr("src.eval.metrics.multi_vehicle_metrics", fake_multi_vehicle_metrics)
    monkeypatch.setattr("src.eval.metrics.save_metrics_csv", fake_save_metrics_csv)
    monkeypatch.setattr("src.algorithms.local_search.two_opt", lambda customers, graph, n_iter=0: customers)
    monkeypatch.setattr("src.viz.plot_routes.plot_multi_vehicle_routes", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.viz.plot_tradeoff.plot_stacked_cost", lambda *args, **kwargs: None)

    cfg = _base_cfg("lexicographic")
    run_q4_from_solution(cfg, _DummyGraph(), str(feedback))

    result_path = tmp_path / "outputs" / "results" / "q4_result_cpqc550.csv"
    assert result_path.exists()
    assert result_path.read_text(encoding="utf-8").strip().startswith("1")


def test_run_q4_from_solution_weighted_prefers_lower_weighted_score(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    qubo_dir = tmp_path / "outputs" / "qubo_ising"
    qubo_dir.mkdir(parents=True)
    _write_manifest(qubo_dir / "q4_export_manifest_v06.json", "meta_v06.json")
    _write_manifest(qubo_dir / "q4_export_manifest_v07.json", "meta_v07.json")
    (qubo_dir / "q4_export_manifest.json").write_text("{}", encoding="utf-8")

    (qubo_dir / "meta_v06.json").write_text(
        json.dumps({"customer_ids": [11]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (qubo_dir / "meta_v07.json").write_text(
        json.dumps({"customer_ids": [99]}, ensure_ascii=False),
        encoding="utf-8",
    )

    feedback = tmp_path / "data" / "platform_feedback"
    b6 = feedback / "q4_v06_k05"
    b7 = feedback / "q4_v07_k07"
    b6.mkdir(parents=True)
    b7.mkdir(parents=True)
    _write_log(b6 / "q4_v06_k05_01.log")
    _write_log(b7 / "q4_v07_k07_01.log")

    def fake_build_vehicle_qubo(graph, customer_ids, penalty_visit, penalty_position):
        return SimpleNamespace(n_vars=1), SimpleNamespace(customer_ids=list(customer_ids))

    monkeypatch.setattr("src.qubo.q4_qubo.build_vehicle_qubo", fake_build_vehicle_qubo)
    monkeypatch.setattr("src.algorithms.route_decode.decode_sub_route", lambda vec, sub, qr: list(sub.customer_ids))
    monkeypatch.setattr("src.eval.metrics.single_route_metrics", lambda route, graph, alpha, beta: {"objective": 0.0})

    def fake_multi_vehicle_metrics(vehicle_routes, graph, capacity, alpha, beta):
        first = vehicle_routes[0][1] if vehicle_routes and len(vehicle_routes[0]) > 2 else -1
        if first == 99:
            return {
                "n_vehicles": 1,
                "total_travel_time": 100.0,
                "total_penalty": 0.0,
                "objective": 100.0,
                "vehicles": [],
            }
        return {
            "n_vehicles": 2,
            "total_travel_time": 10.0,
            "total_penalty": 0.0,
            "objective": 10.0,
            "vehicles": [],
        }

    def fake_save_metrics_csv(metrics, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(str(metrics["n_vehicles"]), encoding="utf-8")

    monkeypatch.setattr("src.eval.metrics.multi_vehicle_metrics", fake_multi_vehicle_metrics)
    monkeypatch.setattr("src.eval.metrics.save_metrics_csv", fake_save_metrics_csv)
    monkeypatch.setattr("src.algorithms.local_search.two_opt", lambda customers, graph, n_iter=0: customers)
    monkeypatch.setattr("src.viz.plot_routes.plot_multi_vehicle_routes", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.viz.plot_tradeoff.plot_stacked_cost", lambda *args, **kwargs: None)

    cfg = _base_cfg("weighted")
    cfg["objective"] = {"alpha": 1.0, "beta": 1.0, "gamma": 0.0}
    run_q4_from_solution(cfg, _DummyGraph(), str(feedback))

    result_path = tmp_path / "outputs" / "results" / "q4_result_cpqc550.csv"
    assert result_path.exists()
    assert result_path.read_text(encoding="utf-8").strip().startswith("2")
