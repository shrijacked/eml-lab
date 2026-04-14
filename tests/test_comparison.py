import os
from pathlib import Path

from eml_lab.comparison import (
    ComparisonResult,
    ComparisonSuiteResult,
    MethodComparisonAggregate,
    MethodComparisonExportResult,
    MethodComparisonIndexEntry,
    MethodComparisonResult,
    PySRStatus,
    aggregate_method_comparisons,
    detect_pysr_environment,
    export_method_comparisons,
    filter_method_comparisons,
    find_method_comparisons,
    load_method_comparison,
    run_method_comparison,
    run_pysr_compare_suite,
    run_pysr_comparison,
    summarize_method_comparisons,
)


def test_detect_pysr_environment_reports_missing_dependencies(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr("shutil.which", lambda name: None)

    status = detect_pysr_environment()

    assert not status.available
    assert not status.pysr_installed
    assert not status.julia_found
    assert "PySR" in status.install_hint


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

    class FakeEquations:
        def to_dict(self, orient: str = "records") -> list[dict[str, object]]:
            assert orient == "records"
            return [{"equation": "exp(x0)", "loss": 0.0}]

        def to_csv(self, path: Path, index: bool = False) -> None:
            Path(path).write_text("equation,loss\nexp(x0),0.0\n", encoding="utf-8")
            assert not index

    class FakePySRRegressor:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.equations_ = FakeEquations()
            self.fitted = False

        def fit(self, x, y) -> None:
            assert x.shape[1] == 1
            assert len(y) == x.shape[0]
            self.fitted = True

        def sympy(self) -> str:
            return "exp(x0)"

    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)
    monkeypatch.setattr("eml_lab.comparison._load_pysr_regressor", lambda: FakePySRRegressor)

    result = run_pysr_comparison("exp", tmp_path, niterations=2, maxsize=8)

    assert result.available
    assert result.pysr["status"] == "ok"
    assert result.pysr["best_equation"] == "exp(x0)"
    assert (tmp_path / "compare-exp" / "summary.json").exists()
    assert (tmp_path / "compare-exp" / "manifest.json").exists()
    assert (tmp_path / "compare-exp" / "eml" / "manifest.json").exists()
    assert (tmp_path / "compare-exp" / "pysr" / "equations.csv").exists()


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

    class FakeEquations:
        def to_dict(self, orient: str = "records") -> list[dict[str, object]]:
            assert orient == "records"
            return [{"equation": "exp(x0)", "loss": 0.0}]

        def to_csv(self, path: Path, index: bool = False) -> None:
            Path(path).write_text("equation,loss\nexp(x0),0.0\n", encoding="utf-8")
            assert not index

    class FakePySRRegressor:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.equations_ = FakeEquations()

        def fit(self, x, y) -> None:
            assert x.shape[1] == 1
            assert len(y) == x.shape[0]

        def sympy(self) -> str:
            return "exp(x0)"

    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)
    monkeypatch.setattr("eml_lab.comparison._load_pysr_regressor", lambda: FakePySRRegressor)

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

    class FakeEquations:
        def to_dict(self, orient: str = "records") -> list[dict[str, object]]:
            assert orient == "records"
            return [{"equation": "exp(x0)", "loss": 0.0}]

        def to_csv(self, path: Path, index: bool = False) -> None:
            Path(path).write_text("equation,loss\nexp(x0),0.0\n", encoding="utf-8")
            assert not index

    class FakePySRRegressor:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.equations_ = FakeEquations()

        def fit(self, x, y) -> None:
            assert x.shape[1] == 1
            assert len(y) == x.shape[0]

        def sympy(self) -> str:
            return "exp(x0)"

    monkeypatch.setattr("eml_lab.comparison.detect_pysr_environment", lambda: status)
    monkeypatch.setattr("eml_lab.comparison._load_pysr_regressor", lambda: FakePySRRegressor)

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
