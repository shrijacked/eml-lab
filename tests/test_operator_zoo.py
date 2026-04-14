from pathlib import Path

from eml_lab.operator_zoo import (
    OperatorZooConfig,
    OperatorZooResult,
    candidate_registry,
    run_operator_zoo,
)


def test_operator_zoo_writes_artifacts(tmp_path: Path) -> None:
    result = run_operator_zoo(tmp_path, OperatorZooConfig(grid_points=7))

    assert isinstance(result, OperatorZooResult)
    assert result.entries
    assert result.best is not None
    assert result.best.rank == 1
    assert Path(result.summary_path).exists()
    assert Path(result.candidates_csv_path).exists()
    assert Path(result.report_path).exists()
    assert Path(result.plot_path).exists()
    assert Path(result.manifest_path).exists()
    assert "Operator Zoo Report" in Path(result.report_path).read_text(encoding="utf-8")


def test_operator_zoo_keeps_exact_eml_as_reference(tmp_path: Path) -> None:
    result = run_operator_zoo(tmp_path, OperatorZooConfig(grid_points=5))
    exact_entries = [
        entry for entry in result.entries if entry.candidate.name == "eml_exact"
    ]

    assert len(exact_entries) == 1
    assert exact_entries[0].candidate.exact_paper_operator
    assert exact_entries[0].mse_to_exact == 0.0


def test_candidate_registry_has_stabilized_variants() -> None:
    candidates = [candidate for candidate, _ in candidate_registry()]

    assert any(candidate.exact_paper_operator for candidate in candidates)
    assert any(not candidate.exact_paper_operator for candidate in candidates)
