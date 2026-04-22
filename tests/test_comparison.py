import json
import os
from pathlib import Path
from types import SimpleNamespace

from eml_lab.comparison import (
    ComparisonResult,
    ComparisonSuiteResult,
    MethodComparisonAggregate,
    MethodComparisonExportResult,
    MethodComparisonIndexEntry,
    MethodComparisonResult,
    MethodComparisonSnapshotHistory,
    MethodComparisonSnapshotHistoryReportResult,
    MethodComparisonSnapshotIndexEntry,
    MethodComparisonSnapshotResult,
    PySRStatus,
    _prepare_julia_environment,
    _run_pysr_worker,
    aggregate_method_comparisons,
    detect_pysr_environment,
    export_method_comparisons,
    filter_method_comparisons,
    find_method_comparison_snapshots,
    find_method_comparisons,
    load_method_comparison,
    report_method_comparison_snapshots,
    run_method_comparison,
    run_pysr_compare_suite,
    run_pysr_comparison,
    snapshot_method_comparisons,
    summarize_method_comparison_snapshots,
    summarize_method_comparisons,
)


def _fake_ok_worker(**kwargs) -> dict[str, object]:
    output_directory = Path(kwargs["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "equations.csv").write_text(
        "equation,loss\nexp(x0),0.0\n", encoding="utf-8"
    )
    return {
        "status": "ok",
        "best_equation": "exp(x0)",
        "equations": [{"equation": "exp(x0)", "loss": 0.0}],
        "output_directory": str(output_directory),
    }


def test_detect_pysr_environment_reports_missing_dependencies(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr("shutil.which", lambda name: None)

    status = detect_pysr_environment()

    assert not status.available
    assert not status.pysr_installed
    assert not status.julia_found
    assert "PySR" in status.install_hint


def test_detect_pysr_environment_accepts_managed_julia_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: object() if name in {"pysr", "juliapkg"} else None,
    )
    monkeypatch.setattr("shutil.which", lambda name: None)

    status = detect_pysr_environment()

    assert status.available
    assert status.pysr_installed
    assert not status.julia_found
    assert status.reason is None
    assert "bootstrapped" in status.install_hint


def test_prepare_julia_environment_uses_temp_depot_when_unset(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("JULIA_DEPOT_PATH", raising=False)
    monkeypatch.delenv("JULIAUP_DEPOT_PATH", raising=False)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    depot = _prepare_julia_environment()

    assert depot == str(tmp_path / "eml-lab-julia-depot")
    assert os.environ["JULIA_DEPOT_PATH"] == depot
    assert os.environ["JULIAUP_DEPOT_PATH"] == depot
    assert Path(depot).exists()


def test_prepare_julia_environment_reuses_bootstrapped_local_julia(
    monkeypatch, tmp_path: Path
) -> None:
    local_julia = tmp_path / "julia_env" / "pyjuliapkg" / "install" / "bin" / "julia"
    local_julia.parent.mkdir(parents=True, exist_ok=True)
    local_julia.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr("sys.prefix", str(tmp_path))
    monkeypatch.delenv("PYTHON_JULIAPKG_EXE", raising=False)
    monkeypatch.delenv("JULIA_DEPOT_PATH", raising=False)
    monkeypatch.delenv("JULIAUP_DEPOT_PATH", raising=False)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "tmp"))

    depot = _prepare_julia_environment()

    assert depot == str(tmp_path / "tmp" / "eml-lab-julia-depot")
    assert os.environ["PYTHON_JULIAPKG_EXE"] == str(local_julia)


def test_run_pysr_comparison_writes_unavailable_summary(tmp_path: Path, monkeypatch) -> None:
    status = PySRStatus(
        available=False,
        pysr_installed=False,
        julia_found=False,
        julia_path=None,
        reason="PySR is not installed and Julia is not on PATH.",
        install_hint="Install with `python -m pip install pysr` and ensure `julia` is on PATH.",
    )
    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)

    result = run_pysr_comparison("ln", tmp_path)

    assert isinstance(result, ComparisonResult)
    assert not result.available
    assert result.pysr["status"] == "unavailable"
    assert (tmp_path / "compare-ln" / "summary.json").exists()
    assert (tmp_path / "compare-ln" / "manifest.json").exists()
    assert (tmp_path / "compare-ln" / "eml" / "manifest.json").exists()


def test_run_pysr_comparison_uses_fake_pysr(monkeypatch, tmp_path: Path) -> None:
    status = PySRStatus(
        available=True,
        pysr_installed=True,
        julia_found=True,
        julia_path="/usr/bin/julia",
        reason=None,
        install_hint="ok",
    )

    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)
    monkeypatch.setattr("eml_lab.comparison._run_pysr_worker", _fake_ok_worker)

    result = run_pysr_comparison("exp", tmp_path, niterations=2, maxsize=8)

    assert result.available
    assert result.pysr["status"] == "ok"
    assert result.pysr["best_equation"] == "exp(x0)"
    assert (tmp_path / "compare-exp" / "summary.json").exists()
    assert (tmp_path / "compare-exp" / "manifest.json").exists()
    assert (tmp_path / "compare-exp" / "eml" / "manifest.json").exists()
    assert (tmp_path / "compare-exp" / "pysr" / "equations.csv").exists()


def test_run_pysr_comparison_reports_bootstrap_failure_without_crashing(
    monkeypatch, tmp_path: Path
) -> None:
    status = PySRStatus(
        available=True,
        pysr_installed=True,
        julia_found=False,
        julia_path=None,
        reason=None,
        install_hint="PySR can be bootstrapped into a writable Julia depot on first import.",
    )

    def _unavailable_worker(**kwargs) -> dict[str, object]:
        return {
            "status": "unavailable",
            "reason": "network bootstrap failed",
            "install_hint": "Install Julia.",
        }

    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)
    monkeypatch.setattr("eml_lab.comparison._run_pysr_worker", _unavailable_worker)
    monkeypatch.setattr("eml_lab.comparison._prepare_julia_environment", lambda: "/tmp/julia")

    result = run_pysr_comparison("exp", tmp_path)

    assert not result.available
    assert not result.success
    assert result.pysr["status"] == "unavailable"
    assert "network bootstrap failed" in str(result.pysr["reason"])
    assert result.pysr["julia_depot_path"] == "/tmp/julia"
    assert "install_hint" in result.pysr
    assert (tmp_path / "compare-exp" / "summary.json").exists()


def test_run_pysr_compare_suite_marks_runtime_bootstrap_failure_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    status = PySRStatus(
        available=True,
        pysr_installed=True,
        julia_found=False,
        julia_path=None,
        reason=None,
        install_hint="PySR can be bootstrapped into a writable Julia depot on first import.",
    )

    def _unavailable_worker(**kwargs) -> dict[str, object]:
        return {
            "status": "unavailable",
            "reason": "network bootstrap failed",
            "install_hint": "Install Julia.",
        }

    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)
    monkeypatch.setattr("eml_lab.comparison._run_pysr_worker", _unavailable_worker)
    monkeypatch.setattr("eml_lab.comparison._prepare_julia_environment", lambda: "/tmp/julia")

    result = run_pysr_compare_suite("shallow", tmp_path)

    assert not result.available
    assert result.pysr_success_rate == 0.0
    assert all(not run.available for run in result.runs)
    assert all(run.status == "unavailable" for run in result.runs)


def test_run_pysr_worker_invokes_isolated_module(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(command, check, capture_output, text, env):
        captured["command"] = command
        captured["check"] = check
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["env"] = env
        summary_path = Path(command[command.index("--summary-path") + 1])
        summary_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "best_equation": "exp(x0)",
                    "output_directory": str(tmp_path / "pysr"),
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    result = _run_pysr_worker(
        target="exp",
        domain=(-1.0, 1.0),
        points=8,
        niterations=2,
        maxsize=8,
        seed=0,
        output_directory=tmp_path / "pysr",
    )

    command = captured["command"]
    assert command[1:3] == ["-m", "eml_lab.pysr_worker"]
    assert "torch" not in " ".join(command)
    assert result["status"] == "ok"
    assert result["best_equation"] == "exp(x0)"


def test_run_pysr_compare_suite_writes_summary_when_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    status = PySRStatus(
        available=False,
        pysr_installed=False,
        julia_found=False,
        julia_path=None,
        reason="PySR is not installed and Julia is not on PATH.",
        install_hint="Install with `python -m pip install pysr` and ensure `julia` is on PATH.",
    )
    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)

    result = run_pysr_compare_suite("shallow", tmp_path)

    assert isinstance(result, ComparisonSuiteResult)
    assert not result.available
    assert result.success
    assert result.pysr_success_rate == 0.0
    assert (Path(result.output_dir) / "summary.json").exists()
    assert Path(result.manifest_path).exists()
    assert len(result.runs) >= 3


def test_run_pysr_compare_suite_uses_fake_pysr(monkeypatch, tmp_path: Path) -> None:
    status = PySRStatus(
        available=True,
        pysr_installed=True,
        julia_found=True,
        julia_path="/usr/bin/julia",
        reason=None,
        install_hint="ok",
    )

    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)
    monkeypatch.setattr("eml_lab.comparison._run_pysr_worker", _fake_ok_worker)

    result = run_pysr_compare_suite("shallow", tmp_path, niterations=2, maxsize=8)

    assert result.available
    assert result.success
    assert result.pysr_success_rate == 1.0
    assert (Path(result.output_dir) / "summary.json").exists()
    assert Path(result.manifest_path).exists()
    assert all(run.status == "ok" for run in result.runs)


def test_run_method_comparison_writes_unavailable_summary(
    tmp_path: Path, monkeypatch
) -> None:
    status = PySRStatus(
        available=False,
        pysr_installed=False,
        julia_found=False,
        julia_path=None,
        reason="PySR is not installed and Julia is not on PATH.",
        install_hint="Install with `python -m pip install pysr` and ensure `julia` is on PATH.",
    )
    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)

    result = run_method_comparison("exp", tmp_path)

    assert isinstance(result, MethodComparisonResult)
    assert result.required_success
    assert result.success
    assert not result.available
    assert result.pysr["status"] == "unavailable"
    root = Path(result.output_dir)
    assert (root / "summary.json").exists()
    assert Path(result.manifest_path).exists()
    assert Path(result.gradient["manifest_path"]).exists()
    assert Path(result.agentic["manifest_path"]).exists()


def test_run_method_comparison_uses_fake_pysr(monkeypatch, tmp_path: Path) -> None:
    status = PySRStatus(
        available=True,
        pysr_installed=True,
        julia_found=True,
        julia_path="/usr/bin/julia",
        reason=None,
        install_hint="ok",
    )

    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)
    monkeypatch.setattr("eml_lab.comparison._run_pysr_worker", _fake_ok_worker)

    result = run_method_comparison("exp", tmp_path, niterations=2, maxsize=8)

    assert result.available
    assert result.required_success
    assert result.success
    assert result.pysr["status"] == "ok"
    assert result.pysr["best_equation"] == "exp(x0)"
    root = Path(result.output_dir)
    assert (root / "summary.json").exists()
    assert Path(result.manifest_path).exists()


def test_load_method_comparison_round_trips(tmp_path: Path, monkeypatch) -> None:
    status = PySRStatus(
        available=False,
        pysr_installed=False,
        julia_found=False,
        julia_path=None,
        reason="PySR is not installed and Julia is not on PATH.",
        install_hint="Install with `python -m pip install pysr` and ensure `julia` is on PATH.",
    )
    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)

    result = run_method_comparison("exp", tmp_path)
    loaded = load_method_comparison(result.output_dir)

    assert isinstance(loaded, MethodComparisonResult)
    assert loaded.target == result.target
    assert loaded.gradient["rpn"] == result.gradient["rpn"]
    assert loaded.agentic["best_rpn"] == result.agentic["best_rpn"]
    assert loaded.pysr["status"] == "unavailable"


def test_find_method_comparisons_discovers_latest_first(tmp_path: Path, monkeypatch) -> None:
    status = PySRStatus(
        available=False,
        pysr_installed=False,
        julia_found=False,
        julia_path=None,
        reason="PySR is not installed and Julia is not on PATH.",
        install_hint="Install with `python -m pip install pysr` and ensure `julia` is on PATH.",
    )
    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)

    first = run_method_comparison("exp", tmp_path)
    second = run_method_comparison("ln", tmp_path)

    os.utime(Path(first.output_dir) / "summary.json", (1, 1))
    os.utime(Path(second.output_dir) / "summary.json", (2, 2))

    entries = find_method_comparisons(tmp_path)

    assert isinstance(entries[0], MethodComparisonIndexEntry)
    assert [entry.target for entry in entries[:2]] == ["ln", "exp"]
    assert entries[0].summary_path.endswith("summary.json")
    assert entries[0].seed == 0
    assert entries[0].created_at
    assert entries[0].gradient_max_mse is not None
    assert entries[0].agentic_max_mse is not None


def test_summarize_method_comparisons_reports_target_level_rollups(
    tmp_path: Path, monkeypatch
) -> None:
    status = PySRStatus(
        available=False,
        pysr_installed=False,
        julia_found=False,
        julia_path=None,
        reason="PySR is not installed and Julia is not on PATH.",
        install_hint="Install with `python -m pip install pysr` and ensure `julia` is on PATH.",
    )
    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)

    first = run_method_comparison("exp", tmp_path)
    second = run_method_comparison("ln", tmp_path)
    third = run_method_comparison("exp", tmp_path)

    os.utime(Path(first.output_dir) / "summary.json", (1, 1))
    os.utime(Path(second.output_dir) / "summary.json", (2, 2))
    os.utime(Path(third.output_dir) / "summary.json", (3, 3))

    report = summarize_method_comparisons(tmp_path)

    assert isinstance(report, MethodComparisonAggregate)
    assert report.run_count == 3
    assert report.target_count == 2
    assert report.required_success_rate == 1.0
    assert report.pysr_available_rate == 0.0
    assert report.status_counts == {"unavailable": 3}
    latest_by_target = {row.target: row for row in report.latest_by_target}
    assert latest_by_target["exp"].runs == 2
    assert latest_by_target["exp"].seed_count == 1
    assert latest_by_target["ln"].runs == 1


def test_aggregate_method_comparisons_can_summarize_filtered_subset(
    tmp_path: Path, monkeypatch
) -> None:
    status = PySRStatus(
        available=False,
        pysr_installed=False,
        julia_found=False,
        julia_path=None,
        reason="PySR is not installed and Julia is not on PATH.",
        install_hint="Install with `python -m pip install pysr` and ensure `julia` is on PATH.",
    )
    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)

    run_method_comparison("exp", tmp_path, seed=0)
    run_method_comparison("exp", tmp_path, seed=1)
    run_method_comparison("ln", tmp_path, seed=0)

    entries = find_method_comparisons(tmp_path)
    filtered = tuple(entry for entry in entries if entry.target == "exp")
    report = aggregate_method_comparisons(filtered, root=tmp_path)

    assert report.run_count == 2
    assert report.target_count == 1
    assert report.required_success_rate == 1.0
    latest_row = report.latest_by_target[0]
    assert latest_row.target == "exp"
    assert latest_row.seed_count == 2
    assert latest_row.best_gradient_max_mse == 0.0
    assert latest_row.best_agentic_max_mse == 0.0


def test_filter_method_comparisons_can_filter_by_seed_and_target(
    tmp_path: Path, monkeypatch
) -> None:
    status = PySRStatus(
        available=False,
        pysr_installed=False,
        julia_found=False,
        julia_path=None,
        reason="PySR is not installed and Julia is not on PATH.",
        install_hint="Install with `python -m pip install pysr` and ensure `julia` is on PATH.",
    )
    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)

    run_method_comparison("exp", tmp_path, seed=0)
    run_method_comparison("exp", tmp_path, seed=1)
    run_method_comparison("ln", tmp_path, seed=0)

    entries = find_method_comparisons(tmp_path)
    filtered = filter_method_comparisons(entries, targets=["exp"], seeds=[1])

    assert len(filtered) == 1
    assert filtered[0].target == "exp"
    assert filtered[0].seed == 1


def test_export_method_comparisons_writes_filtered_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    status = PySRStatus(
        available=False,
        pysr_installed=False,
        julia_found=False,
        julia_path=None,
        reason="PySR is not installed and Julia is not on PATH.",
        install_hint="Install with `python -m pip install pysr` and ensure `julia` is on PATH.",
    )
    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)

    run_method_comparison("exp", tmp_path, seed=0)
    run_method_comparison("exp", tmp_path, seed=1)
    run_method_comparison("ln", tmp_path, seed=0)

    export = export_method_comparisons(
        tmp_path,
        tmp_path / "exports",
        targets=["exp"],
        seeds=[1],
    )

    assert isinstance(export, MethodComparisonExportResult)
    assert export.run_count == 1
    assert Path(export.summary_path).exists()
    assert Path(export.runs_csv_path).exists()
    assert Path(export.latest_csv_path).exists()
    assert Path(export.manifest_path).exists()
    summary = Path(export.summary_path).read_text(encoding="utf-8")
    assert '"run_count": 1' in summary


def test_snapshot_method_comparisons_writes_report_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    status = PySRStatus(
        available=False,
        pysr_installed=False,
        julia_found=False,
        julia_path=None,
        reason="PySR is not installed and Julia is not on PATH.",
        install_hint="Install with `python -m pip install pysr` and ensure `julia` is on PATH.",
    )
    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)

    run_method_comparison("exp", tmp_path, seed=0)
    run_method_comparison("exp", tmp_path, seed=1)
    run_method_comparison("ln", tmp_path, seed=0)

    snapshot = snapshot_method_comparisons(
        tmp_path,
        tmp_path / "snapshots",
        targets=["exp"],
        seeds=[1],
    )

    assert isinstance(snapshot, MethodComparisonSnapshotResult)
    assert snapshot.run_count == 1
    assert Path(snapshot.summary_path).exists()
    assert Path(snapshot.report_path).exists()
    assert Path(snapshot.runs_csv_path).exists()
    assert Path(snapshot.latest_csv_path).exists()
    assert Path(snapshot.manifest_path).exists()
    assert snapshot.plot_paths
    assert all(Path(path).exists() for path in snapshot.plot_paths.values())
    report = Path(snapshot.report_path).read_text(encoding="utf-8")
    assert "# Method Comparison Snapshot" in report
    assert "Runs: 1" in report
    assert "## Latest By Target" in report
    assert "## Recent Runs" in report


def test_snapshot_history_report_summarizes_multiple_snapshots(
    tmp_path: Path, monkeypatch
) -> None:
    status = PySRStatus(
        available=False,
        pysr_installed=False,
        julia_found=False,
        julia_path=None,
        reason="PySR is not installed and Julia is not on PATH.",
        install_hint="Install with `python -m pip install pysr` and ensure `julia` is on PATH.",
    )
    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)

    run_method_comparison("exp", tmp_path, seed=0)
    first = snapshot_method_comparisons(tmp_path, tmp_path / "snapshots", targets=["exp"])
    run_method_comparison("ln", tmp_path, seed=0)
    second = snapshot_method_comparisons(tmp_path, tmp_path / "snapshots")

    os.utime(Path(first.summary_path), (1, 1))
    os.utime(Path(second.summary_path), (2, 2))

    snapshots = find_method_comparison_snapshots(tmp_path / "snapshots")
    history = summarize_method_comparison_snapshots(tmp_path / "snapshots")
    report = report_method_comparison_snapshots(
        tmp_path / "snapshots",
        tmp_path / "snapshot-reports",
    )

    assert isinstance(snapshots[0], MethodComparisonSnapshotIndexEntry)
    assert isinstance(history, MethodComparisonSnapshotHistory)
    assert history.snapshot_count == 2
    assert history.total_run_count == 3
    assert history.target_count == 2
    assert history.latest_snapshot_dir == second.output_dir
    assert history.best_required_success_rate == 1.0
    assert {trend.target for trend in history.target_trends} == {"exp", "ln"}
    assert isinstance(report, MethodComparisonSnapshotHistoryReportResult)
    assert report.snapshot_count == 2
    assert Path(report.summary_path).exists()
    assert Path(report.report_path).exists()
    assert Path(report.snapshots_csv_path).exists()
    assert Path(report.target_trends_csv_path).exists()
    assert Path(report.manifest_path).exists()
    assert all(Path(path).exists() for path in report.plot_paths.values())
