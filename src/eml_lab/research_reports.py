"""Per-target reporting for research-tier EML campaign artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from eml_lab.artifacts import ArtifactFile, write_artifact_manifest
from eml_lab.targets import get_target, list_targets


@dataclass(frozen=True)
class ResearchRunEntry:
    created_at: str
    campaign_output_dir: str
    target: str
    display_name: str
    tier: str
    status: str
    success: bool
    effective_success: bool
    required: bool
    rpn: str | None
    max_mse: float | None
    final_loss: float | None
    failure_reason: str | None
    expected_depth: int | None
    failure_modes: tuple[str, ...]
    notes: str
    output_dir: str
    summary_path: str
    manifest_path: str | None

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "failure_modes": list(self.failure_modes)}


@dataclass(frozen=True)
class ResearchTargetRow:
    target: str
    display_name: str
    runs: int
    success_count: int
    success_rate: float
    latest_status: str
    latest_failure_reason: str | None
    latest_rpn: str | None
    best_max_mse: float | None
    best_final_loss: float | None
    expected_depth: int | None
    failure_modes: tuple[str, ...]
    notes: str
    latest_output_dir: str | None

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "failure_modes": list(self.failure_modes)}


@dataclass(frozen=True)
class ResearchAggregate:
    root: str
    run_count: int
    target_count: int
    attempted_target_count: int
    success_count: int
    success_rate: float
    status_counts: dict[str, int]
    runs: tuple[ResearchRunEntry, ...]
    targets: tuple[ResearchTargetRow, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "run_count": self.run_count,
            "target_count": self.target_count,
            "attempted_target_count": self.attempted_target_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "status_counts": self.status_counts,
            "runs": [run.to_dict() for run in self.runs],
            "targets": [target.to_dict() for target in self.targets],
        }


@dataclass(frozen=True)
class ResearchReportResult:
    source_root: str
    output_dir: str
    manifest_path: str
    summary_path: str
    report_path: str
    runs_csv_path: str
    targets_csv_path: str
    run_count: int
    target_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def find_research_runs(root: str | Path = "runs") -> tuple[ResearchRunEntry, ...]:
    source = Path(root)
    if not source.exists():
        return ()

    summary_paths = set(source.rglob("campaign-phase2-research-*/summary.json"))
    if source.name.startswith("campaign-phase2-research-") and (source / "summary.json").exists():
        summary_paths.add(source / "summary.json")

    ordered_paths = sorted(
        summary_paths,
        key=lambda path: (path.stat().st_mtime, path.as_posix()),
        reverse=True,
    )
    entries: list[ResearchRunEntry] = []
    for summary_path in ordered_paths:
        try:
            entries.extend(_load_research_runs_from_campaign(summary_path))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return tuple(entries)


def aggregate_research_runs(
    entries: Sequence[ResearchRunEntry],
    *,
    root: str | Path = "runs",
) -> ResearchAggregate:
    runs = tuple(entries)
    by_target: dict[str, list[ResearchRunEntry]] = {}
    status_counts: dict[str, int] = {}
    success_count = 0
    for run in runs:
        by_target.setdefault(run.target, []).append(run)
        status_counts[run.status] = status_counts.get(run.status, 0) + 1
        if run.success:
            success_count += 1

    target_rows: list[ResearchTargetRow] = []
    for target_name in list_targets(tier="research"):
        spec = get_target(target_name)
        target_runs = by_target.get(target_name, [])
        latest = target_runs[0] if target_runs else None
        target_success_count = sum(1 for run in target_runs if run.success)
        max_mses = [run.max_mse for run in target_runs if run.max_mse is not None]
        final_losses = [run.final_loss for run in target_runs if run.final_loss is not None]
        target_rows.append(
            ResearchTargetRow(
                target=target_name,
                display_name=spec.display_name,
                runs=len(target_runs),
                success_count=target_success_count,
                success_rate=target_success_count / len(target_runs) if target_runs else 0.0,
                latest_status=latest.status if latest is not None else "not_run",
                latest_failure_reason=latest.failure_reason if latest is not None else None,
                latest_rpn=latest.rpn if latest is not None else None,
                best_max_mse=min(max_mses) if max_mses else None,
                best_final_loss=min(final_losses) if final_losses else None,
                expected_depth=spec.expected_depth,
                failure_modes=spec.failure_modes,
                notes=spec.notes,
                latest_output_dir=latest.output_dir if latest is not None else None,
            )
        )

    run_count = len(runs)
    return ResearchAggregate(
        root=str(Path(root)),
        run_count=run_count,
        target_count=len(target_rows),
        attempted_target_count=sum(1 for row in target_rows if row.runs > 0),
        success_count=success_count,
        success_rate=success_count / run_count if run_count else 0.0,
        status_counts=status_counts,
        runs=runs,
        targets=tuple(target_rows),
    )


def summarize_research_runs(root: str | Path = "runs") -> ResearchAggregate:
    return aggregate_research_runs(find_research_runs(root), root=root)


def write_research_report(
    root: str | Path = "runs",
    output_dir: str | Path = "runs/research-reports",
) -> ResearchReportResult:
    report = summarize_research_runs(root)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    report_root = Path(output_dir) / f"research-target-report-{timestamp}"
    report_root.mkdir(parents=True, exist_ok=True)

    summary_path = report_root / "summary.json"
    runs_csv_path = report_root / "runs.csv"
    targets_csv_path = report_root / "targets.csv"
    report_path = report_root / "report.md"
    summary_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    _write_csv(runs_csv_path, [run.to_dict() for run in report.runs])
    _write_csv(targets_csv_path, [target.to_dict() for target in report.targets])
    report_path.write_text(_render_research_report(report), encoding="utf-8")

    manifest = write_artifact_manifest(
        report_root,
        files=[
            ArtifactFile(label="summary", path=str(summary_path), kind="json"),
            ArtifactFile(label="runs-csv", path=str(runs_csv_path), kind="csv"),
            ArtifactFile(label="targets-csv", path=str(targets_csv_path), kind="csv"),
            ArtifactFile(label="report", path=str(report_path), kind="markdown"),
        ],
        metadata={
            "kind": "research-target-report",
            "source_root": str(Path(root)),
            "run_count": report.run_count,
            "target_count": report.target_count,
            "attempted_target_count": report.attempted_target_count,
        },
    )
    return ResearchReportResult(
        source_root=str(Path(root)),
        output_dir=str(report_root),
        manifest_path=manifest.manifest_path,
        summary_path=str(summary_path),
        report_path=str(report_path),
        runs_csv_path=str(runs_csv_path),
        targets_csv_path=str(targets_csv_path),
        run_count=report.run_count,
        target_count=report.target_count,
    )


def _load_research_runs_from_campaign(summary_path: Path) -> tuple[ResearchRunEntry, ...]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("suite") != "phase2-research":
        return ()

    created_at = datetime.fromtimestamp(summary_path.stat().st_mtime, UTC).isoformat()
    campaign_output_dir = str(summary_path.parent)
    entries: list[ResearchRunEntry] = []
    for run_payload in payload.get("runs", []):
        if not isinstance(run_payload, dict):
            continue
        metrics = run_payload.get("metrics", {})
        metrics = metrics if isinstance(metrics, dict) else {}
        if run_payload.get("kind") != "train" or metrics.get("target_tier") != "research":
            continue
        target = _target_from_metrics(metrics)
        spec = get_target(target)
        verification = metrics.get("verification", {})
        verification = verification if isinstance(verification, dict) else {}
        entries.append(
            ResearchRunEntry(
                created_at=created_at,
                campaign_output_dir=campaign_output_dir,
                target=target,
                display_name=spec.display_name,
                tier=str(metrics.get("target_tier", spec.tier)),
                status=str(run_payload.get("status", "failed")),
                success=bool(run_payload.get("success", False)),
                effective_success=bool(run_payload.get("effective_success", False)),
                required=bool(run_payload.get("required", False)),
                rpn=_optional_string(metrics.get("rpn")),
                max_mse=_optional_float(verification.get("max_mse")),
                final_loss=_optional_float(metrics.get("final_loss")),
                failure_reason=_failure_reason(metrics, verification),
                expected_depth=_optional_int(metrics.get("expected_depth", spec.expected_depth)),
                failure_modes=_failure_modes(metrics.get("failure_modes", ())),
                notes=str(metrics.get("notes", spec.notes)),
                output_dir=str(run_payload.get("output_dir", "")),
                summary_path=str(run_payload.get("summary_path", "")),
                manifest_path=_optional_string(run_payload.get("manifest_path")),
            )
        )
    return tuple(entries)


def _target_from_metrics(metrics: dict[str, object]) -> str:
    target = metrics.get("target")
    if isinstance(target, str):
        return target
    config = metrics.get("config", {})
    if isinstance(config, dict) and isinstance(config.get("target"), str):
        return str(config["target"])
    raise KeyError("Research run metrics do not include a target")


def _failure_reason(metrics: dict[str, object], verification: dict[str, object]) -> str | None:
    reason = metrics.get("failure_reason")
    if isinstance(reason, str) and reason:
        return reason
    reason = verification.get("failure_reason")
    if isinstance(reason, str) and reason:
        return reason
    return None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _failure_modes(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_research_report(report: ResearchAggregate) -> str:
    generated_at = datetime.now(UTC).isoformat()
    lines = [
        "# Research Target Report",
        "",
        f"Generated at: `{generated_at}`",
        f"Source root: `{report.root}`",
        "",
        "## Scope",
        f"- Research runs: {report.run_count}",
        f"- Research targets: {report.target_count}",
        f"- Attempted targets: {report.attempted_target_count}",
        f"- Success rate: {report.success_rate:.0%}",
        "",
        "## Status Counts",
    ]
    if report.status_counts:
        lines.extend(
            f"- `{status}`: {count}" for status, count in sorted(report.status_counts.items())
        )
    else:
        lines.append("- No saved research runs found.")
    lines.extend(
        [
            "",
            "## Per-Target Outcomes",
            (
                "| target | runs | success_rate | latest_status | best_max_mse | "
                "latest_failure_reason | expected_depth | latest_rpn |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.targets:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.target,
                    str(row.runs),
                    f"{row.success_rate:.2%}",
                    row.latest_status,
                    _markdown_scalar(row.best_max_mse),
                    _markdown_scalar(row.latest_failure_reason),
                    _markdown_scalar(row.expected_depth),
                    f"`{row.latest_rpn}`" if row.latest_rpn else "n/a",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Target Notes",
            "| target | failure_modes | notes |",
            "| --- | --- | --- |",
        ]
    )
    for row in report.targets:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.target,
                    "<br>".join(row.failure_modes) if row.failure_modes else "n/a",
                    row.notes,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Run Details",
            (
                "| created_at | target | status | success | max_mse | final_loss | "
                "failure_reason | output_dir |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    if report.runs:
        for run in report.runs:
            lines.append(
                "| "
                + " | ".join(
                    [
                        run.created_at,
                        run.target,
                        run.status,
                        "yes" if run.success else "no",
                        _markdown_scalar(run.max_mse),
                        _markdown_scalar(run.final_loss),
                        _markdown_scalar(run.failure_reason),
                        f"`{run.output_dir}`",
                    ]
                )
                + " |"
            )
    else:
        lines.append("| _none_ | n/a | n/a | no | n/a | n/a | n/a | `n/a` |")
    return "\n".join(lines) + "\n"


def _markdown_scalar(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3e}"
    return str(value).replace("|", "\\|")
