from pathlib import Path

from eml_lab.campaigns import run_campaign
from eml_lab.research_reports import (
    ResearchReportResult,
    ResearchRunEntry,
    aggregate_research_runs,
    find_research_runs,
    write_research_report,
)
from eml_lab.targets import list_targets


def test_find_research_runs_discovers_campaign_train_steps(tmp_path: Path) -> None:
    run_campaign("phase2-research", tmp_path)

    runs = find_research_runs(tmp_path)

    assert len(runs) == 4
    assert isinstance(runs[0], ResearchRunEntry)
    assert {run.target for run in runs} == set(list_targets(tier="research"))
    assert all(run.tier == "research" for run in runs)
    assert all(run.summary_path.endswith("metrics.json") for run in runs)


def test_aggregate_research_runs_includes_unrun_research_targets(tmp_path: Path) -> None:
    report = aggregate_research_runs((), root=tmp_path)

    assert report.run_count == 0
    assert report.target_count == len(list_targets(tier="research"))
    assert {row.target for row in report.targets} == set(list_targets(tier="research"))
    assert all(row.runs == 0 for row in report.targets)
    assert all(row.latest_status == "not_run" for row in report.targets)


def test_write_research_report_creates_report_bundle(tmp_path: Path) -> None:
    run_campaign("phase2-research", tmp_path)

    result = write_research_report(tmp_path, tmp_path / "reports")

    assert isinstance(result, ResearchReportResult)
    assert result.run_count == 4
    assert result.target_count == len(list_targets(tier="research"))
    assert Path(result.summary_path).exists()
    assert Path(result.report_path).exists()
    assert Path(result.runs_csv_path).exists()
    assert Path(result.targets_csv_path).exists()
    assert Path(result.manifest_path).exists()
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert "# Research Target Report" in report
    assert "## Per-Target Outcomes" in report
    assert "## Run Details" in report


def test_research_report_api_exports_from_package_root() -> None:
    from eml_lab import ResearchReportResult, summarize_research_runs, write_research_report

    assert ResearchReportResult is not None
    assert callable(summarize_research_runs)
    assert callable(write_research_report)
