from pathlib import Path

from eml_lab import app
from eml_lab.benchmarks import run_benchmark_suite
from eml_lab.cli import main


def test_app_module_imports() -> None:
    assert callable(app.main)


def test_cli_app_dry_run() -> None:
    assert main(["app", "--dry-run"]) == 0


def test_cli_train_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "train"

    assert (
        main(
            [
                "train",
                "--target",
                "exp",
                "--depth",
                "1",
                "--steps",
                "20",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "loss.csv").exists()
    assert (output_dir / "tree.json").exists()


def test_benchmark_suite_writes_artifacts(tmp_path: Path) -> None:
    result = run_benchmark_suite("shallow", tmp_path)

    assert result.recovery_rate == 1.0
    assert (Path(result.output_dir) / "summary.json").exists()
