from pathlib import Path

from eml_lab.campaigns import CampaignResult, list_campaigns, run_campaign


def test_list_campaigns_includes_phase2_foundation() -> None:
    assert "phase2-foundation" in list_campaigns()
    assert "phase2-agentic" in list_campaigns()


def test_run_campaign_writes_summary_and_manifest(tmp_path: Path) -> None:
    result = run_campaign("phase2-foundation", tmp_path)

    assert isinstance(result, CampaignResult)
    assert result.success
    assert (Path(result.output_dir) / "summary.json").exists()
    assert Path(result.manifest_path).exists()
    assert any(run.kind == "benchmark" for run in result.runs)
    assert any(run.kind == "comparison" for run in result.runs)
    assert all(Path(run.summary_path).exists() for run in result.runs)


def test_run_agentic_campaign_writes_orchestrator_artifacts(tmp_path: Path) -> None:
    result = run_campaign("phase2-agentic", tmp_path)

    assert result.success
    orchestration_runs = [run for run in result.runs if run.kind == "orchestration"]
    assert len(orchestration_runs) == 2
    assert all(Path(run.summary_path).exists() for run in orchestration_runs)
    assert all(Path(run.manifest_path).exists() for run in orchestration_runs if run.manifest_path)


def test_run_research_campaign_reports_failure_metadata(tmp_path: Path) -> None:
    result = run_campaign("phase2-research", tmp_path)

    assert result.success
    train_runs = [run for run in result.runs if run.kind == "train"]
    assert len(train_runs) == 4
    assert all(run.metrics["target_tier"] == "research" for run in train_runs)
    assert all(len(run.metrics["failure_modes"]) >= 1 for run in train_runs)
    failed = [run for run in train_runs if not run.success]
    assert failed
    assert all(
        run.metrics["verification"]["failure_reason"] is not None
        or run.metrics["failure_reason"] is not None
        for run in failed
    )
