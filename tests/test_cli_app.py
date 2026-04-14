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
    assert (output_dir / "manifest.json").exists()


def test_benchmark_suite_writes_artifacts(tmp_path: Path) -> None:
    result = run_benchmark_suite("shallow", tmp_path)

    assert result.recovery_rate == 1.0
    assert (Path(result.output_dir) / "summary.json").exists()
    assert (Path(result.output_dir) / "manifest.json").exists()


def test_cli_compare_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "compare"

    exit_code = main(["compare", "--target", "exp", "--output-dir", str(output_dir)])

    assert exit_code in {0, 3}
    compare_root = output_dir / "compare-exp"
    assert (compare_root / "summary.json").exists()
    assert (compare_root / "manifest.json").exists()
    assert (compare_root / "eml" / "manifest.json").exists()


def test_cli_compare_suite_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "compare-suite"

    exit_code = main(["compare-suite", "--suite", "shallow", "--output-dir", str(output_dir)])

    assert exit_code in {0, 3}
    suite_roots = list(output_dir.glob("compare-suite-shallow-*"))
    assert len(suite_roots) == 1
    assert (suite_roots[0] / "summary.json").exists()
    assert (suite_roots[0] / "manifest.json").exists()


def test_cli_compare_methods_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "compare-methods"

    exit_code = main(["compare-methods", "--target", "exp", "--output-dir", str(output_dir)])

    assert exit_code in {0, 3}
    roots = list(output_dir.glob("method-compare-exp-*"))
    assert len(roots) == 1
    assert (roots[0] / "summary.json").exists()
    assert (roots[0] / "manifest.json").exists()


def test_cli_compare_methods_history_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "compare-methods-history"

    exit_code = main(["compare-methods", "--target", "exp", "--output-dir", str(output_dir)])

    assert exit_code in {0, 3}
    assert main(["compare-methods-history", "--root", str(output_dir)]) == 0


def test_cli_compare_methods_report_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "compare-methods-report"

    exit_code = main(["compare-methods", "--target", "exp", "--output-dir", str(output_dir)])

    assert exit_code in {0, 3}
    assert main(["compare-methods-report", "--root", str(output_dir)]) == 0


def test_cli_compare_methods_export_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "compare-methods-export"
    export_dir = tmp_path / "exports"

    exit_code = main(["compare-methods", "--target", "exp", "--output-dir", str(output_dir)])

    assert exit_code in {0, 3}
    assert (
        main(
            [
                "compare-methods-export",
                "--root",
                str(output_dir),
                "--output-dir",
                str(export_dir),
                "--target",
                "exp",
            ]
        )
        == 0
    )
    export_roots = list(export_dir.glob("method-compare-export-*"))
    assert len(export_roots) == 1
    assert (export_roots[0] / "summary.json").exists()
    assert (export_roots[0] / "runs.csv").exists()
    assert (export_roots[0] / "latest_by_target.csv").exists()
    assert (export_roots[0] / "manifest.json").exists()


def test_cli_compare_methods_snapshot_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "compare-methods-snapshot"
    snapshot_dir = tmp_path / "snapshots"

    exit_code = main(["compare-methods", "--target", "exp", "--output-dir", str(output_dir)])

    assert exit_code in {0, 3}
    assert (
        main(
            [
                "compare-methods-snapshot",
                "--root",
                str(output_dir),
                "--output-dir",
                str(snapshot_dir),
                "--target",
                "exp",
            ]
        )
        == 0
    )
    snapshot_roots = list(snapshot_dir.glob("method-compare-snapshot-*"))
    assert len(snapshot_roots) == 1
    assert (snapshot_roots[0] / "summary.json").exists()
    assert (snapshot_roots[0] / "report.md").exists()
    assert (snapshot_roots[0] / "runs.csv").exists()
    assert (snapshot_roots[0] / "latest_by_target.csv").exists()
    assert (snapshot_roots[0] / "runs_by_target.png").exists()
    assert (snapshot_roots[0] / "manifest.json").exists()


def test_cli_compare_methods_snapshot_history_report_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "compare-methods-snapshot-history"
    snapshot_dir = tmp_path / "snapshots"
    report_dir = tmp_path / "snapshot-reports"

    exit_code = main(["compare-methods", "--target", "exp", "--output-dir", str(output_dir)])

    assert exit_code in {0, 3}
    assert (
        main(
            [
                "compare-methods-snapshot",
                "--root",
                str(output_dir),
                "--output-dir",
                str(snapshot_dir),
                "--target",
                "exp",
            ]
        )
        == 0
    )
    assert main(["compare-methods-snapshot-history", "--root", str(snapshot_dir)]) == 0
    assert (
        main(
            [
                "compare-methods-snapshot-report",
                "--root",
                str(snapshot_dir),
                "--output-dir",
                str(report_dir),
            ]
        )
        == 0
    )
    report_roots = list(report_dir.glob("method-compare-snapshot-history-*"))
    assert len(report_roots) == 1
    assert (report_roots[0] / "summary.json").exists()
    assert (report_roots[0] / "report.md").exists()
    assert (report_roots[0] / "snapshots.csv").exists()
    assert (report_roots[0] / "target_trends.csv").exists()
    assert (report_roots[0] / "required_success_rate_over_time.png").exists()
    assert (report_roots[0] / "manifest.json").exists()


def test_cli_campaign_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "campaign"

    assert main(["campaign", "--suite", "phase2-foundation", "--output-dir", str(output_dir)]) == 0
    campaign_roots = list(output_dir.glob("campaign-phase2-foundation-*"))
    assert len(campaign_roots) == 1
    assert (campaign_roots[0] / "summary.json").exists()
    assert (campaign_roots[0] / "manifest.json").exists()


def test_cli_agentic_campaign_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "campaign-agentic"

    assert main(["campaign", "--suite", "phase2-agentic", "--output-dir", str(output_dir)]) == 0
    campaign_roots = list(output_dir.glob("campaign-phase2-agentic-*"))
    assert len(campaign_roots) == 1
    assert (campaign_roots[0] / "summary.json").exists()
    assert (campaign_roots[0] / "manifest.json").exists()


def test_cli_research_campaign_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "campaign-research"

    assert main(["campaign", "--suite", "phase2-research", "--output-dir", str(output_dir)]) == 0
    campaign_roots = list(output_dir.glob("campaign-phase2-research-*"))
    assert len(campaign_roots) == 1
    assert (campaign_roots[0] / "summary.json").exists()
    assert (campaign_roots[0] / "manifest.json").exists()


def test_cli_orchestrate_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "orchestrate"

    assert (
        main(
            [
                "orchestrate",
                "--target",
                "exp",
                "--budget",
                "12",
                "--beam-width",
                "4",
                "--seed-count",
                "3",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    orchestrate_roots = list(output_dir.glob("orchestrate-exp-*"))
    assert len(orchestrate_roots) == 1
    assert (orchestrate_roots[0] / "summary.json").exists()
    assert (orchestrate_roots[0] / "manifest.json").exists()
    assert (orchestrate_roots[0] / "events.jsonl").exists()
