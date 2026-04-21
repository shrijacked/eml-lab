import pytest

from eml_lab.benchmarks import BenchmarkResult, benchmark_seed_sensitivity_table


def test_benchmark_seed_sensitivity_table_groups_runs_by_target() -> None:
    result = BenchmarkResult(
        suite="example",
        output_dir="runs/example",
        manifest_path="runs/example/manifest.json",
        runs=(
            {
                "target": "exp",
                "seed": 0,
                "success": True,
                "max_mse": 0.1,
                "elapsed_seconds": 1.2,
            },
            {
                "target": "exp",
                "seed": 1,
                "success": False,
                "max_mse": 0.4,
                "elapsed_seconds": 2.0,
            },
            {
                "target": "ln",
                "seed": 0,
                "success": True,
                "max_mse": 0.01,
                "elapsed_seconds": 0.5,
            },
        ),
    )

    rows = benchmark_seed_sensitivity_table(result)
    exp_row = next(row for row in rows if row["target"] == "exp")
    ln_row = next(row for row in rows if row["target"] == "ln")

    assert exp_row["seed_count"] == 2
    assert exp_row["success_rate"] == 0.5
    assert exp_row["best_max_mse"] == 0.1
    assert exp_row["worst_max_mse"] == 0.4
    assert exp_row["mse_spread"] == pytest.approx(0.3)
    assert exp_row["runtime_spread_seconds"] == pytest.approx(0.8)
    assert ln_row["seed_count"] == 1
    assert ln_row["mse_spread"] == 0.0
