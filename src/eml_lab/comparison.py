"""Optional baseline comparison against PySR."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from eml_lab.agentic import OrchestratorConfig, run_orchestrator
from eml_lab.artifacts import ArtifactFile, write_artifact_manifest
from eml_lab.targets import TargetSpec, get_target, list_targets, sample_inputs
from eml_lab.training import TrainConfig, train_target, write_train_artifacts


@dataclass(frozen=True)
class PySRStatus:
    available: bool
    pysr_installed: bool
    julia_found: bool
    julia_path: str | None
    reason: str | None
    install_hint: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonResult:
    target: str
    output_dir: str
    manifest_path: str
    pysr_status: PySRStatus
    eml: dict[str, object]
    pysr: dict[str, object]

    @property
    def available(self) -> bool:
        return self.pysr_status.available and self.status != "unavailable"

    @property
    def status(self) -> str:
        return str(self.pysr.get("status", "ok"))

    @property
    def success(self) -> bool:
        return bool(self.eml.get("success", False)) and self.status == "ok"

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "available": self.available,
            "success": self.success,
            "status": self.status,
            "pysr_status": self.pysr_status.to_dict(),
            "eml": self.eml,
            "pysr": self.pysr,
        }


@dataclass(frozen=True)
class ComparisonSuiteEntry:
    target: str
    output_dir: str
    manifest_path: str
    available: bool
    success: bool
    status: str
    eml_success: bool
    eml_rpn: str
    eml_max_mse: float
    pysr_best_equation: str | None
    pysr_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonSuiteResult:
    suite: str
    output_dir: str
    manifest_path: str
    pysr_status: PySRStatus
    runs: tuple[ComparisonSuiteEntry, ...]

    @property
    def available(self) -> bool:
        return self.pysr_status.available and all(run.available for run in self.runs)

    @property
    def success(self) -> bool:
        return all(run.eml_success for run in self.runs)

    @property
    def pysr_success_rate(self) -> float:
        if not self.runs:
            return 0.0
        successes = sum(1 for run in self.runs if run.status == "ok")
        return successes / len(self.runs)

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "available": self.available,
            "success": self.success,
            "pysr_success_rate": self.pysr_success_rate,
            "pysr_status": self.pysr_status.to_dict(),
            "runs": [run.to_dict() for run in self.runs],
        }


@dataclass(frozen=True)
class MethodComparisonResult:
    target: str
    output_dir: str
    manifest_path: str
    pysr_status: PySRStatus
    gradient: dict[str, object]
    agentic: dict[str, object]
    pysr: dict[str, object]

    @property
    def available(self) -> bool:
        return self.pysr_status.available and self.status != "unavailable"

    @property
    def status(self) -> str:
        return str(self.pysr.get("status", "ok"))

    @property
    def required_success(self) -> bool:
        return bool(self.gradient.get("success", False)) and bool(
            self.agentic.get("success", False)
        )

    @property
    def success(self) -> bool:
        return self.required_success and self.status in {"ok", "unavailable"}

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "available": self.available,
            "required_success": self.required_success,
            "success": self.success,
            "status": self.status,
            "pysr_status": self.pysr_status.to_dict(),
            "gradient": self.gradient,
            "agentic": self.agentic,
            "pysr": self.pysr,
        }


@dataclass(frozen=True)
class MethodComparisonIndexEntry:
    target: str
    output_dir: str
    summary_path: str
    manifest_path: str
    created_at: str
    seed: int | None
    available: bool
    required_success: bool
    success: bool
    status: str
    gradient_expression: str | None
    agentic_expression: str | None
    pysr_expression: str | None
    gradient_max_mse: float | None
    agentic_max_mse: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MethodComparisonAggregateRow:
    target: str
    runs: int
    seed_count: int
    required_success_rate: float
    latest_status: str
    latest_available: bool
    latest_gradient_expression: str | None
    latest_agentic_expression: str | None
    latest_pysr_expression: str | None
    best_gradient_max_mse: float | None
    best_agentic_max_mse: float | None
    latest_output_dir: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MethodComparisonAggregate:
    root: str
    run_count: int
    target_count: int
    required_success_rate: float
    pysr_available_rate: float
    status_counts: dict[str, int]
    runs: tuple[MethodComparisonIndexEntry, ...]
    latest_by_target: tuple[MethodComparisonAggregateRow, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "run_count": self.run_count,
            "target_count": self.target_count,
            "required_success_rate": self.required_success_rate,
            "pysr_available_rate": self.pysr_available_rate,
            "status_counts": self.status_counts,
            "runs": [entry.to_dict() for entry in self.runs],
            "latest_by_target": [row.to_dict() for row in self.latest_by_target],
        }


@dataclass(frozen=True)
class MethodComparisonExportResult:
    source_root: str
    output_dir: str
    manifest_path: str
    summary_path: str
    runs_csv_path: str
    latest_csv_path: str
    filters: dict[str, object]
    run_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MethodComparisonSnapshotResult:
    source_root: str
    output_dir: str
    manifest_path: str
    summary_path: str
    report_path: str
    runs_csv_path: str
    latest_csv_path: str
    plot_paths: dict[str, str]
    filters: dict[str, object]
    run_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MethodComparisonSnapshotIndexEntry:
    output_dir: str
    summary_path: str
    report_path: str
    manifest_path: str
    created_at: str
    source_root: str
    run_count: int
    target_count: int
    required_success_rate: float
    pysr_available_rate: float
    status_counts: dict[str, int]
    filters: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MethodComparisonSnapshotTargetTrend:
    snapshot_output_dir: str
    snapshot_created_at: str
    target: str
    runs: int
    seed_count: int
    required_success_rate: float
    best_gradient_max_mse: float | None
    best_agentic_max_mse: float | None
    latest_status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MethodComparisonSnapshotHistory:
    root: str
    snapshot_count: int
    total_run_count: int
    target_count: int
    latest_snapshot_dir: str | None
    latest_required_success_rate: float | None
    best_required_success_rate: float | None
    status_counts: dict[str, int]
    snapshots: tuple[MethodComparisonSnapshotIndexEntry, ...]
    target_trends: tuple[MethodComparisonSnapshotTargetTrend, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "snapshot_count": self.snapshot_count,
            "total_run_count": self.total_run_count,
            "target_count": self.target_count,
            "latest_snapshot_dir": self.latest_snapshot_dir,
            "latest_required_success_rate": self.latest_required_success_rate,
            "best_required_success_rate": self.best_required_success_rate,
            "status_counts": self.status_counts,
            "snapshots": [entry.to_dict() for entry in self.snapshots],
            "target_trends": [entry.to_dict() for entry in self.target_trends],
        }


@dataclass(frozen=True)
class MethodComparisonSnapshotHistoryReportResult:
    source_root: str
    output_dir: str
    manifest_path: str
    summary_path: str
    report_path: str
    snapshots_csv_path: str
    target_trends_csv_path: str
    plot_paths: dict[str, str]
    snapshot_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_pysr_environment() -> PySRStatus:
    pysr_installed = importlib.util.find_spec("pysr") is not None
    julia_path = shutil.which("julia")
    julia_found = julia_path is not None
    managed_julia = importlib.util.find_spec("juliapkg") is not None
    reason = None
    if not pysr_installed and not julia_found:
        reason = "PySR is not installed and Julia is not on PATH."
    elif not pysr_installed:
        reason = "PySR is not installed."
    elif not julia_found and not managed_julia:
        reason = "Julia is not on PATH."
    return PySRStatus(
        available=pysr_installed and (julia_found or managed_julia),
        pysr_installed=pysr_installed,
        julia_found=julia_found,
        julia_path=julia_path,
        reason=reason,
        install_hint=(
            "Install with `python -m pip install pysr`. If `julia` is not on PATH, "
            "PySR can be bootstrapped into a writable Julia depot on first import."
        ),
    )


def _prepare_julia_environment() -> str:
    depot_path = os.environ.get("JULIA_DEPOT_PATH")
    if depot_path:
        Path(depot_path).mkdir(parents=True, exist_ok=True)
    else:
        depot = Path(tempfile.gettempdir()) / "eml-lab-julia-depot"
        depot.mkdir(parents=True, exist_ok=True)
        os.environ["JULIA_DEPOT_PATH"] = str(depot)
        os.environ.setdefault("JULIAUP_DEPOT_PATH", str(depot))
        depot_path = str(depot)

    local_julia = (
        Path(sys.prefix) / "julia_env" / "pyjuliapkg" / "install" / "bin" / "julia"
    )
    if local_julia.exists():
        os.environ.setdefault("PYTHON_JULIAPKG_EXE", str(local_julia))

    return depot_path


def _timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")


def run_pysr_comparison(
    target: str = "ln",
    output_dir: str | Path = "runs",
    *,
    points: int = 128,
    niterations: int = 40,
    maxsize: int = 20,
    seed: int = 0,
) -> ComparisonResult:
    spec = get_target(target)
    root = Path(output_dir) / f"compare-{spec.name}"
    root.mkdir(parents=True, exist_ok=True)

    eml_summary = _run_gradient_baseline(spec, root / "eml", seed=seed)

    status = detect_pysr_environment()
    pysr_summary: dict[str, object]
    if not status.available:
        pysr_summary = {
            "status": "unavailable",
            "reason": status.reason,
            "install_hint": status.install_hint,
        }
        result = _finalize_comparison(spec.name, root, status, eml_summary, pysr_summary)
        return result

    pysr_summary = _run_available_pysr(spec, root, points, niterations, maxsize, seed)
    return _finalize_comparison(spec.name, root, status, eml_summary, pysr_summary)


def run_method_comparison(
    target: str = "ln",
    output_dir: str | Path = "runs",
    *,
    train_steps: int | None = None,
    budget: int | None = None,
    beam_width: int = 6,
    seed_count: int = 4,
    max_depth: int | None = None,
    points: int = 128,
    niterations: int = 40,
    maxsize: int = 20,
    seed: int = 0,
) -> MethodComparisonResult:
    spec = get_target(target)
    if spec.known_route is None:
        raise ValueError(
            f"Target {spec.name!r} has no known route; cross-method comparison requires "
            "an orchestratable target."
        )

    timestamp = _timestamp_slug()
    root = Path(output_dir) / f"method-compare-{spec.name}-{timestamp}"
    root.mkdir(parents=True, exist_ok=True)

    gradient_summary = _run_gradient_baseline(spec, root / "gradient", seed=seed, steps=train_steps)
    agentic_summary = _run_agentic_baseline(
        spec,
        root / "agentic",
        seed=seed,
        budget=budget,
        beam_width=beam_width,
        seed_count=seed_count,
        max_depth=max_depth,
    )

    status = detect_pysr_environment()
    if not status.available:
        pysr_summary = {
            "status": "unavailable",
            "reason": status.reason,
            "install_hint": status.install_hint,
        }
    else:
        pysr_summary = _run_available_pysr(spec, root, points, niterations, maxsize, seed)

    return _finalize_method_comparison(
        spec.name,
        root,
        status,
        gradient_summary,
        agentic_summary,
        pysr_summary,
    )


def run_pysr_compare_suite(
    suite: str = "shallow",
    output_dir: str | Path = "runs",
    *,
    points: int = 128,
    niterations: int = 40,
    maxsize: int = 20,
    seed: int = 0,
) -> ComparisonSuiteResult:
    if suite != "shallow":
        raise ValueError("Only the 'shallow' compare suite exists in Phase 2")

    status = detect_pysr_environment()
    timestamp = _timestamp_slug()
    root = Path(output_dir) / f"compare-suite-{suite}-{timestamp}"
    root.mkdir(parents=True, exist_ok=True)

    runs: list[ComparisonSuiteEntry] = []
    files: list[ArtifactFile] = []
    targets = list_targets(tier="stable", comparison_eligible=True)
    for index, target_name in enumerate(targets):
        run_root = root / f"{index:02d}-{target_name}"
        result = run_pysr_comparison(
            target_name,
            run_root,
            points=points,
            niterations=niterations,
            maxsize=maxsize,
            seed=seed,
        )
        entry = ComparisonSuiteEntry(
            target=result.target,
            output_dir=result.output_dir,
            manifest_path=result.manifest_path,
            available=result.available,
            success=result.success,
            status=result.status,
            eml_success=bool(result.eml["success"]),
            eml_rpn=str(result.eml["rpn"]),
            eml_max_mse=float(result.eml["verification"]["max_mse"]),
            pysr_best_equation=result.pysr.get("best_equation"),
            pysr_reason=result.pysr.get("reason"),
        )
        runs.append(entry)
        files.extend(
            [
                ArtifactFile(
                    label=f"{target_name}-summary",
                    path=str(Path(result.output_dir) / "summary.json"),
                    kind="json",
                ),
                ArtifactFile(label=f"{target_name}-root", path=result.output_dir, kind="directory"),
                ArtifactFile(
                    label=f"{target_name}-manifest",
                    path=result.manifest_path,
                    kind="json",
                ),
            ]
        )

    summary_path = root / "summary.json"
    manifest = write_artifact_manifest(
        root,
        files=[
            ArtifactFile(label="summary", path=str(summary_path), kind="json"),
            *files,
        ],
        metadata={
            "kind": "comparison-suite",
            "suite": suite,
            "pysr_available": status.available,
            "target_count": len(runs),
        },
    )
    result = ComparisonSuiteResult(
        suite=suite,
        output_dir=str(root),
        manifest_path=manifest.manifest_path,
        pysr_status=status,
        runs=tuple(runs),
    )
    summary_path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return result


def load_method_comparison(source: str | Path) -> MethodComparisonResult:
    path = Path(source)
    summary_path = path / "summary.json" if path.is_dir() else path
    if not summary_path.exists():
        raise FileNotFoundError(f"Method comparison summary not found: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return MethodComparisonResult(
        target=str(payload["target"]),
        output_dir=str(payload["output_dir"]),
        manifest_path=str(payload["manifest_path"]),
        pysr_status=PySRStatus(**payload["pysr_status"]),
        gradient=dict(payload["gradient"]),
        agentic=dict(payload["agentic"]),
        pysr=dict(payload["pysr"]),
    )


def find_method_comparisons(root: str | Path = "runs") -> tuple[MethodComparisonIndexEntry, ...]:
    source = Path(root)
    if not source.exists():
        return ()

    summary_paths = sorted(
        source.rglob("method-compare-*/summary.json"),
        key=lambda path: (path.stat().st_mtime, path.as_posix()),
        reverse=True,
    )
    entries: list[MethodComparisonIndexEntry] = []
    for summary_path in summary_paths:
        try:
            result = load_method_comparison(summary_path)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        created_at = datetime.fromtimestamp(summary_path.stat().st_mtime, UTC).isoformat()
        gradient_mse = result.gradient.get("verification", {}).get("max_mse")
        agentic_mse = result.agentic.get("max_mse")
        seed = result.gradient.get("config", {}).get("seed")
        entries.append(
            MethodComparisonIndexEntry(
                target=result.target,
                output_dir=result.output_dir,
                summary_path=str(summary_path),
                manifest_path=result.manifest_path,
                created_at=created_at,
                seed=seed if isinstance(seed, int) else None,
                available=result.available,
                required_success=result.required_success,
                success=result.success,
                status=result.status,
                gradient_expression=result.gradient.get("rpn"),
                agentic_expression=result.agentic.get("best_rpn"),
                pysr_expression=result.pysr.get("best_equation"),
                gradient_max_mse=float(gradient_mse) if gradient_mse is not None else None,
                agentic_max_mse=float(agentic_mse) if agentic_mse is not None else None,
            )
        )
    return tuple(entries)


def filter_method_comparisons(
    entries: Sequence[MethodComparisonIndexEntry],
    *,
    targets: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    required_only: bool = False,
) -> tuple[MethodComparisonIndexEntry, ...]:
    target_set = set(targets or [])
    status_set = set(statuses or [])
    seed_set = set(seeds or [])
    filtered: list[MethodComparisonIndexEntry] = []
    for entry in entries:
        if target_set and entry.target not in target_set:
            continue
        if status_set and entry.status not in status_set:
            continue
        if seed_set and entry.seed not in seed_set:
            continue
        if required_only and not entry.required_success:
            continue
        filtered.append(entry)
    return tuple(filtered)


def aggregate_method_comparisons(
    entries: Sequence[MethodComparisonIndexEntry],
    *,
    root: str | Path = "runs",
) -> MethodComparisonAggregate:
    runs = tuple(entries)
    run_count = len(runs)
    if not runs:
        return MethodComparisonAggregate(
            root=str(Path(root)),
            run_count=0,
            target_count=0,
            required_success_rate=0.0,
            pysr_available_rate=0.0,
            status_counts={},
            runs=(),
            latest_by_target=(),
        )

    status_counts: dict[str, int] = {}
    by_target: dict[str, list[MethodComparisonIndexEntry]] = {}
    required_successes = 0
    available_runs = 0
    for entry in runs:
        status_counts[entry.status] = status_counts.get(entry.status, 0) + 1
        by_target.setdefault(entry.target, []).append(entry)
        if entry.required_success:
            required_successes += 1
        if entry.available:
            available_runs += 1

    latest_rows: list[MethodComparisonAggregateRow] = []
    for target in sorted(by_target):
        target_entries = by_target[target]
        latest = target_entries[0]
        success_count = sum(1 for entry in target_entries if entry.required_success)
        seeds = {entry.seed for entry in target_entries if entry.seed is not None}
        gradient_mses = [
            entry.gradient_max_mse for entry in target_entries if entry.gradient_max_mse is not None
        ]
        agentic_mses = [
            entry.agentic_max_mse for entry in target_entries if entry.agentic_max_mse is not None
        ]
        latest_rows.append(
            MethodComparisonAggregateRow(
                target=target,
                runs=len(target_entries),
                seed_count=len(seeds),
                required_success_rate=success_count / len(target_entries),
                latest_status=latest.status,
                latest_available=latest.available,
                latest_gradient_expression=latest.gradient_expression,
                latest_agentic_expression=latest.agentic_expression,
                latest_pysr_expression=latest.pysr_expression,
                best_gradient_max_mse=min(gradient_mses) if gradient_mses else None,
                best_agentic_max_mse=min(agentic_mses) if agentic_mses else None,
                latest_output_dir=latest.output_dir,
            )
        )

    return MethodComparisonAggregate(
        root=str(Path(root)),
        run_count=run_count,
        target_count=len(by_target),
        required_success_rate=required_successes / run_count,
        pysr_available_rate=available_runs / run_count,
        status_counts=status_counts,
        runs=runs,
        latest_by_target=tuple(latest_rows),
    )


def summarize_method_comparisons(root: str | Path = "runs") -> MethodComparisonAggregate:
    return aggregate_method_comparisons(find_method_comparisons(root), root=root)


def load_method_comparison_snapshot(source: str | Path) -> MethodComparisonSnapshotIndexEntry:
    path = Path(source)
    summary_path = path / "summary.json" if path.is_dir() else path
    if not summary_path.exists():
        raise FileNotFoundError(f"Method comparison snapshot summary not found: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    root = summary_path.parent
    manifest_path = root / "manifest.json"
    manifest_payload: dict[str, object] = {}
    if manifest_path.exists():
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest_payload.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    kind = metadata.get("kind")
    if kind is not None and kind != "method-comparison-snapshot":
        raise ValueError(f"Not a method comparison snapshot: {summary_path}")
    status_counts = payload.get("status_counts", {})
    filters = metadata.get("filters", {})
    created_at = datetime.fromtimestamp(summary_path.stat().st_mtime, UTC).isoformat()
    return MethodComparisonSnapshotIndexEntry(
        output_dir=str(root),
        summary_path=str(summary_path),
        report_path=str(root / "report.md"),
        manifest_path=str(manifest_path),
        created_at=created_at,
        source_root=str(metadata.get("source_root", payload.get("root", ""))),
        run_count=int(payload.get("run_count", 0)),
        target_count=int(payload.get("target_count", 0)),
        required_success_rate=float(payload.get("required_success_rate", 0.0)),
        pysr_available_rate=float(payload.get("pysr_available_rate", 0.0)),
        status_counts={
            str(status): int(count)
            for status, count in dict(status_counts).items()
            if isinstance(count, int)
        },
        filters=dict(filters) if isinstance(filters, dict) else {},
    )


def find_method_comparison_snapshots(
    root: str | Path = "runs/snapshots",
) -> tuple[MethodComparisonSnapshotIndexEntry, ...]:
    source = Path(root)
    if not source.exists():
        return ()

    summary_paths = set(source.rglob("method-compare-snapshot-*/summary.json"))
    if source.name.startswith("method-compare-snapshot-") and (source / "summary.json").exists():
        summary_paths.add(source / "summary.json")

    ordered_paths = sorted(
        summary_paths,
        key=lambda path: (path.stat().st_mtime, path.as_posix()),
        reverse=True,
    )
    entries: list[MethodComparisonSnapshotIndexEntry] = []
    for summary_path in ordered_paths:
        try:
            entries.append(load_method_comparison_snapshot(summary_path))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return tuple(entries)


def aggregate_method_comparison_snapshots(
    entries: Sequence[MethodComparisonSnapshotIndexEntry],
    *,
    root: str | Path = "runs/snapshots",
) -> MethodComparisonSnapshotHistory:
    snapshots = tuple(entries)
    if not snapshots:
        return MethodComparisonSnapshotHistory(
            root=str(Path(root)),
            snapshot_count=0,
            total_run_count=0,
            target_count=0,
            latest_snapshot_dir=None,
            latest_required_success_rate=None,
            best_required_success_rate=None,
            status_counts={},
            snapshots=(),
            target_trends=(),
        )

    status_counts: dict[str, int] = {}
    targets: set[str] = set()
    target_trends: list[MethodComparisonSnapshotTargetTrend] = []
    for snapshot in snapshots:
        for status, count in snapshot.status_counts.items():
            status_counts[status] = status_counts.get(status, 0) + count
        for trend in _snapshot_target_trends(snapshot):
            targets.add(trend.target)
            target_trends.append(trend)

    return MethodComparisonSnapshotHistory(
        root=str(Path(root)),
        snapshot_count=len(snapshots),
        total_run_count=sum(snapshot.run_count for snapshot in snapshots),
        target_count=len(targets),
        latest_snapshot_dir=snapshots[0].output_dir,
        latest_required_success_rate=snapshots[0].required_success_rate,
        best_required_success_rate=max(
            snapshot.required_success_rate for snapshot in snapshots
        ),
        status_counts=status_counts,
        snapshots=snapshots,
        target_trends=tuple(target_trends),
    )


def summarize_method_comparison_snapshots(
    root: str | Path = "runs/snapshots",
) -> MethodComparisonSnapshotHistory:
    return aggregate_method_comparison_snapshots(
        find_method_comparison_snapshots(root),
        root=root,
    )


def export_method_comparisons(
    root: str | Path = "runs",
    output_dir: str | Path = "runs/exports",
    *,
    targets: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    required_only: bool = False,
) -> MethodComparisonExportResult:
    entries = find_method_comparisons(root)
    filtered = filter_method_comparisons(
        entries,
        targets=targets,
        statuses=statuses,
        seeds=seeds,
        required_only=required_only,
    )
    filters = _filters_payload(
        targets=targets,
        statuses=statuses,
        seeds=seeds,
        required_only=required_only,
    )
    return write_method_comparison_export(
        filtered,
        output_dir,
        root=root,
        filters=filters,
    )


def write_method_comparison_export(
    entries: Sequence[MethodComparisonIndexEntry],
    output_dir: str | Path,
    *,
    root: str | Path = "runs",
    filters: Mapping[str, object] | None = None,
) -> MethodComparisonExportResult:
    timestamp = _timestamp_slug()
    export_root = Path(output_dir) / f"method-compare-export-{timestamp}"
    export_root.mkdir(parents=True, exist_ok=True)

    report = aggregate_method_comparisons(entries, root=root)
    filters_payload = dict(filters or {})
    summary_path, runs_csv_path, latest_csv_path = _write_method_comparison_bundle_files(
        export_root,
        report,
    )
    manifest = write_artifact_manifest(
        export_root,
        files=[
            ArtifactFile(label="summary", path=str(summary_path), kind="json"),
            ArtifactFile(label="runs-csv", path=str(runs_csv_path), kind="csv"),
            ArtifactFile(label="latest-by-target-csv", path=str(latest_csv_path), kind="csv"),
        ],
        metadata={
            "kind": "method-comparison-export",
            "source_root": str(Path(root)),
            "run_count": report.run_count,
            "target_count": report.target_count,
            "filters": filters_payload,
        },
    )
    return MethodComparisonExportResult(
        source_root=str(Path(root)),
        output_dir=str(export_root),
        manifest_path=manifest.manifest_path,
        summary_path=str(summary_path),
        runs_csv_path=str(runs_csv_path),
        latest_csv_path=str(latest_csv_path),
        filters=filters_payload,
        run_count=report.run_count,
    )


def snapshot_method_comparisons(
    root: str | Path = "runs",
    output_dir: str | Path = "runs/snapshots",
    *,
    targets: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    required_only: bool = False,
) -> MethodComparisonSnapshotResult:
    entries = find_method_comparisons(root)
    filtered = filter_method_comparisons(
        entries,
        targets=targets,
        statuses=statuses,
        seeds=seeds,
        required_only=required_only,
    )
    filters = _filters_payload(
        targets=targets,
        statuses=statuses,
        seeds=seeds,
        required_only=required_only,
    )
    return write_method_comparison_snapshot(
        filtered,
        output_dir,
        root=root,
        filters=filters,
    )


def write_method_comparison_snapshot(
    entries: Sequence[MethodComparisonIndexEntry],
    output_dir: str | Path,
    *,
    root: str | Path = "runs",
    filters: Mapping[str, object] | None = None,
) -> MethodComparisonSnapshotResult:
    timestamp = _timestamp_slug()
    snapshot_root = Path(output_dir) / f"method-compare-snapshot-{timestamp}"
    snapshot_root.mkdir(parents=True, exist_ok=True)

    report = aggregate_method_comparisons(entries, root=root)
    filters_payload = dict(filters or {})
    summary_path, runs_csv_path, latest_csv_path = _write_method_comparison_bundle_files(
        snapshot_root,
        report,
    )
    plot_paths = _write_method_comparison_plots(snapshot_root, report)
    report_path = snapshot_root / "report.md"
    report_path.write_text(
        _render_method_comparison_snapshot_report(
            report,
            filters=filters_payload,
            plot_paths=plot_paths,
        ),
        encoding="utf-8",
    )
    manifest = write_artifact_manifest(
        snapshot_root,
        files=[
            ArtifactFile(label="summary", path=str(summary_path), kind="json"),
            ArtifactFile(label="runs-csv", path=str(runs_csv_path), kind="csv"),
            ArtifactFile(label="latest-by-target-csv", path=str(latest_csv_path), kind="csv"),
            ArtifactFile(label="report", path=str(report_path), kind="markdown"),
            *[
                ArtifactFile(label=f"plot-{label}", path=path, kind="png")
                for label, path in plot_paths.items()
            ],
        ],
        metadata={
            "kind": "method-comparison-snapshot",
            "source_root": str(Path(root)),
            "run_count": report.run_count,
            "target_count": report.target_count,
            "filters": filters_payload,
        },
    )
    return MethodComparisonSnapshotResult(
        source_root=str(Path(root)),
        output_dir=str(snapshot_root),
        manifest_path=manifest.manifest_path,
        summary_path=str(summary_path),
        report_path=str(report_path),
        runs_csv_path=str(runs_csv_path),
        latest_csv_path=str(latest_csv_path),
        plot_paths=plot_paths,
        filters=filters_payload,
        run_count=report.run_count,
    )


def report_method_comparison_snapshots(
    root: str | Path = "runs/snapshots",
    output_dir: str | Path = "runs/snapshot-reports",
) -> MethodComparisonSnapshotHistoryReportResult:
    history = summarize_method_comparison_snapshots(root)
    return write_method_comparison_snapshot_history_report(
        history,
        output_dir,
        root=root,
    )


def write_method_comparison_snapshot_history_report(
    history: MethodComparisonSnapshotHistory,
    output_dir: str | Path,
    *,
    root: str | Path = "runs/snapshots",
) -> MethodComparisonSnapshotHistoryReportResult:
    timestamp = _timestamp_slug()
    report_root = Path(output_dir) / f"method-compare-snapshot-history-{timestamp}"
    report_root.mkdir(parents=True, exist_ok=True)

    summary_path = report_root / "summary.json"
    snapshots_csv_path = report_root / "snapshots.csv"
    target_trends_csv_path = report_root / "target_trends.csv"
    report_path = report_root / "report.md"
    summary_path.write_text(json.dumps(history.to_dict(), indent=2), encoding="utf-8")
    _write_csv(
        snapshots_csv_path,
        [snapshot.to_dict() for snapshot in history.snapshots],
    )
    _write_csv(
        target_trends_csv_path,
        [trend.to_dict() for trend in history.target_trends],
    )
    plot_paths = _write_method_comparison_snapshot_history_plots(report_root, history)
    report_path.write_text(
        _render_method_comparison_snapshot_history_report(history, plot_paths=plot_paths),
        encoding="utf-8",
    )
    manifest = write_artifact_manifest(
        report_root,
        files=[
            ArtifactFile(label="summary", path=str(summary_path), kind="json"),
            ArtifactFile(label="snapshots-csv", path=str(snapshots_csv_path), kind="csv"),
            ArtifactFile(
                label="target-trends-csv",
                path=str(target_trends_csv_path),
                kind="csv",
            ),
            ArtifactFile(label="report", path=str(report_path), kind="markdown"),
            *[
                ArtifactFile(label=f"plot-{label}", path=path, kind="png")
                for label, path in plot_paths.items()
            ],
        ],
        metadata={
            "kind": "method-comparison-snapshot-history",
            "source_root": str(Path(root)),
            "snapshot_count": history.snapshot_count,
            "total_run_count": history.total_run_count,
            "target_count": history.target_count,
        },
    )
    return MethodComparisonSnapshotHistoryReportResult(
        source_root=str(Path(root)),
        output_dir=str(report_root),
        manifest_path=manifest.manifest_path,
        summary_path=str(summary_path),
        report_path=str(report_path),
        snapshots_csv_path=str(snapshots_csv_path),
        target_trends_csv_path=str(target_trends_csv_path),
        plot_paths=plot_paths,
        snapshot_count=history.snapshot_count,
    )


def _run_available_pysr(
    spec: TargetSpec,
    root: Path,
    points: int,
    niterations: int,
    maxsize: int,
    seed: int,
) -> dict[str, object]:
    if len(spec.variables) != 1:
        return {
            "status": "unsupported",
            "reason": "The optional PySR baseline in v1 supports univariate targets only.",
        }

    depot_path = _prepare_julia_environment()
    inputs = sample_inputs(spec, points=points)
    variable_name = spec.variables[0]
    x_tensor = inputs[variable_name]
    y_tensor = spec.function(inputs)
    x_imag_max = float(x_tensor.imag.abs().max().item())
    y_imag_max = float(y_tensor.imag.abs().max().item())
    if y_imag_max > 1e-12 or x_imag_max > 1e-12:
        return {
            "status": "unsupported",
            "reason": "PySR comparison requires real-valued samples for v1 targets.",
            "x_imag_max": x_imag_max,
            "y_imag_max": y_imag_max,
        }

    model_output = root / "pysr"
    domain = spec.train_domain[variable_name]
    summary = _run_pysr_worker(
        target=spec.name,
        domain=domain,
        points=points,
        niterations=niterations,
        maxsize=maxsize,
        seed=seed,
        output_directory=model_output,
    )
    summary["julia_depot_path"] = depot_path
    return summary


def _run_pysr_worker(
    *,
    target: str,
    domain: tuple[float, float],
    points: int,
    niterations: int,
    maxsize: int,
    seed: int,
    output_directory: Path,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "summary.json"
    command = [
        sys.executable,
        "-m",
        "eml_lab.pysr_worker",
        "--target",
        target,
        "--low",
        str(domain[0]),
        "--high",
        str(domain[1]),
        "--points",
        str(points),
        "--niterations",
        str(niterations),
        "--maxsize",
        str(maxsize),
        "--seed",
        str(seed),
        "--output-directory",
        str(output_directory),
        "--summary-path",
        str(summary_path),
    ]
    env = os.environ.copy()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    except OSError as exc:
        return {
            "status": "error",
            "reason": str(exc),
            "worker_command": command,
        }

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            return {
                "status": "error",
                "reason": "PySR worker summary was not a JSON object.",
                "return_code": completed.returncode,
            }
        if completed.returncode != 0:
            summary.setdefault("return_code", completed.returncode)
            if completed.stderr:
                summary.setdefault("stderr", completed.stderr)
        return summary

    status = "unavailable" if completed.returncode == 3 else "error"
    reason = completed.stderr.strip() or completed.stdout.strip() or "PySR worker failed."
    return {
        "status": status,
        "reason": reason,
        "return_code": completed.returncode,
        "worker_command": command,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _filters_payload(
    *,
    targets: Sequence[str] | None,
    statuses: Sequence[str] | None,
    seeds: Sequence[int] | None,
    required_only: bool,
) -> dict[str, object]:
    return {
        "targets": list(targets or []),
        "statuses": list(statuses or []),
        "seeds": list(seeds or []),
        "required_only": required_only,
    }


def _write_method_comparison_bundle_files(
    root: Path,
    report: MethodComparisonAggregate,
) -> tuple[Path, Path, Path]:
    summary_path = root / "summary.json"
    runs_csv_path = root / "runs.csv"
    latest_csv_path = root / "latest_by_target.csv"
    summary_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    _write_csv(runs_csv_path, [entry.to_dict() for entry in report.runs])
    _write_csv(latest_csv_path, [row.to_dict() for row in report.latest_by_target])
    return summary_path, runs_csv_path, latest_csv_path


def _write_method_comparison_plots(
    root: Path,
    report: MethodComparisonAggregate,
) -> dict[str, str]:
    plot_paths = {
        "runs_by_target": str(root / "runs_by_target.png"),
        "required_success_rate_by_target": str(root / "required_success_rate_by_target.png"),
        "runs_by_seed": str(root / "runs_by_seed.png"),
        "status_counts": str(root / "status_counts.png"),
        "error_trend": str(root / "error_trend.png"),
    }
    _save_bar_chart(
        Path(plot_paths["runs_by_target"]),
        {row.target: row.runs for row in report.latest_by_target},
        title="Runs by Target",
        ylabel="Runs",
    )
    _save_bar_chart(
        Path(plot_paths["required_success_rate_by_target"]),
        {row.target: row.required_success_rate for row in report.latest_by_target},
        title="Required Success Rate by Target",
        ylabel="Success Rate",
        ylim=(0.0, 1.0),
    )
    _save_bar_chart(
        Path(plot_paths["runs_by_seed"]),
        _seed_count_values(report.runs),
        title="Runs by Seed",
        ylabel="Runs",
    )
    _save_bar_chart(
        Path(plot_paths["status_counts"]),
        report.status_counts,
        title="Status Counts",
        ylabel="Runs",
    )
    _save_error_trend_chart(Path(plot_paths["error_trend"]), report.runs)
    return plot_paths


def _write_method_comparison_snapshot_history_plots(
    root: Path,
    history: MethodComparisonSnapshotHistory,
) -> dict[str, str]:
    plot_paths = {
        "required_success_rate_over_time": str(root / "required_success_rate_over_time.png"),
        "run_count_over_time": str(root / "run_count_over_time.png"),
        "target_success_rate_over_time": str(root / "target_success_rate_over_time.png"),
        "status_counts": str(root / "status_counts.png"),
    }
    ordered_snapshots = sorted(history.snapshots, key=lambda snapshot: snapshot.created_at)
    _save_line_chart(
        Path(plot_paths["required_success_rate_over_time"]),
        {
            "required_success_rate": [
                snapshot.required_success_rate for snapshot in ordered_snapshots
            ]
        },
        title="Required Success Rate Over Time",
        ylabel="Success Rate",
        ylim=(0.0, 1.0),
    )
    _save_line_chart(
        Path(plot_paths["run_count_over_time"]),
        {"run_count": [snapshot.run_count for snapshot in ordered_snapshots]},
        title="Run Count Over Time",
        ylabel="Runs",
    )
    _save_target_trend_chart(
        Path(plot_paths["target_success_rate_over_time"]),
        history.target_trends,
    )
    _save_bar_chart(
        Path(plot_paths["status_counts"]),
        history.status_counts,
        title="Snapshot History Status Counts",
        ylabel="Runs",
    )
    return plot_paths


def _render_method_comparison_snapshot_report(
    report: MethodComparisonAggregate,
    *,
    filters: Mapping[str, object],
    plot_paths: Mapping[str, str],
) -> str:
    generated_at = datetime.now(UTC).isoformat()
    lines = [
        "# Method Comparison Snapshot",
        "",
        f"Generated at: `{generated_at}`",
        f"Source root: `{report.root}`",
        "",
        "## Scope",
        f"- Runs: {report.run_count}",
        f"- Targets: {report.target_count}",
        f"- Required success rate: {report.required_success_rate:.0%}",
        f"- PySR availability rate: {report.pysr_available_rate:.0%}",
        f"- Filters: `{json.dumps(dict(filters), sort_keys=True)}`",
        "",
        "## Status Counts",
    ]
    if report.status_counts:
        lines.extend(
            f"- `{status}`: {count}" for status, count in sorted(report.status_counts.items())
        )
    else:
        lines.append("- No saved runs matched the filter.")
    lines.extend(
        [
            "",
            "## Latest By Target",
            (
                "| target | runs | seed_count | required_success_rate | latest_status | "
                "best_gradient_max_mse | best_agentic_max_mse | latest_output_dir |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    if report.latest_by_target:
        for row in report.latest_by_target:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.target,
                        str(row.runs),
                        str(row.seed_count),
                        f"{row.required_success_rate:.2%}",
                        row.latest_status,
                        _markdown_scalar(row.best_gradient_max_mse),
                        _markdown_scalar(row.best_agentic_max_mse),
                        f"`{row.latest_output_dir}`",
                    ]
                )
                + " |"
            )
    else:
        lines.append("| _none_ | 0 | 0 | 0.00% | n/a | n/a | n/a | `n/a` |")
    lines.extend(
        [
            "",
            "## Recent Runs",
            (
                "| created_at | target | seed | status | required_success | "
                "gradient_max_mse | agentic_max_mse | output_dir |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    if report.runs:
        for entry in report.runs[:10]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        entry.created_at,
                        entry.target,
                        _markdown_scalar(entry.seed),
                        entry.status,
                        "yes" if entry.required_success else "no",
                        _markdown_scalar(entry.gradient_max_mse),
                        _markdown_scalar(entry.agentic_max_mse),
                        f"`{entry.output_dir}`",
                    ]
                )
                + " |"
            )
    else:
        lines.append("| _none_ | n/a | n/a | n/a | no | n/a | n/a | `n/a` |")
    lines.extend(["", "## Plot Files"])
    lines.extend(f"- `{label}`: `{path}`" for label, path in plot_paths.items())
    return "\n".join(lines) + "\n"


def _render_method_comparison_snapshot_history_report(
    history: MethodComparisonSnapshotHistory,
    *,
    plot_paths: Mapping[str, str],
) -> str:
    generated_at = datetime.now(UTC).isoformat()
    lines = [
        "# Method Comparison Snapshot History",
        "",
        f"Generated at: `{generated_at}`",
        f"Snapshot root: `{history.root}`",
        "",
        "## Scope",
        f"- Snapshots: {history.snapshot_count}",
        f"- Total saved runs represented: {history.total_run_count}",
        f"- Targets observed: {history.target_count}",
        f"- Latest required success rate: {_markdown_scalar(history.latest_required_success_rate)}",
        f"- Best required success rate: {_markdown_scalar(history.best_required_success_rate)}",
        f"- Latest snapshot: `{history.latest_snapshot_dir or 'n/a'}`",
        "",
        "## Status Counts",
    ]
    if history.status_counts:
        lines.extend(
            f"- `{status}`: {count}" for status, count in sorted(history.status_counts.items())
        )
    else:
        lines.append("- No snapshots found.")
    lines.extend(
        [
            "",
            "## Snapshots",
            (
                "| created_at | run_count | target_count | required_success_rate | "
                "pysr_available_rate | output_dir |"
            ),
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    if history.snapshots:
        for snapshot in history.snapshots:
            lines.append(
                "| "
                + " | ".join(
                    [
                        snapshot.created_at,
                        str(snapshot.run_count),
                        str(snapshot.target_count),
                        f"{snapshot.required_success_rate:.2%}",
                        f"{snapshot.pysr_available_rate:.2%}",
                        f"`{snapshot.output_dir}`",
                    ]
                )
                + " |"
            )
    else:
        lines.append("| _none_ | 0 | 0 | 0.00% | 0.00% | `n/a` |")
    lines.extend(
        [
            "",
            "## Target Trends",
            (
                "| snapshot_created_at | target | runs | seed_count | "
                "required_success_rate | best_gradient_max_mse | best_agentic_max_mse |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    if history.target_trends:
        for trend in sorted(
            history.target_trends,
            key=lambda item: (item.target, item.snapshot_created_at),
        ):
            lines.append(
                "| "
                + " | ".join(
                    [
                        trend.snapshot_created_at,
                        trend.target,
                        str(trend.runs),
                        str(trend.seed_count),
                        f"{trend.required_success_rate:.2%}",
                        _markdown_scalar(trend.best_gradient_max_mse),
                        _markdown_scalar(trend.best_agentic_max_mse),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| _none_ | n/a | 0 | 0 | 0.00% | n/a | n/a |")
    lines.extend(["", "## Plot Files"])
    lines.extend(f"- `{label}`: `{path}`" for label, path in plot_paths.items())
    return "\n".join(lines) + "\n"


def _save_bar_chart(
    path: Path,
    values: Mapping[str, float | int],
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4))
    if values:
        labels = list(values.keys())
        numbers = [float(value) for value in values.values()]
        ax.bar(labels, numbers, color="#4C78A8")
        ax.tick_params(axis="x", labelrotation=20)
    else:
        _render_no_data(ax, title)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_line_chart(
    path: Path,
    series: Mapping[str, Sequence[float | int]],
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4))
    if series and any(values for values in series.values()):
        for label, values in series.items():
            x_values = list(range(1, len(values) + 1))
            ax.plot(x_values, [float(value) for value in values], marker="o", label=label)
        ax.legend()
    else:
        _render_no_data(ax, title)
    ax.set_title(title)
    ax.set_xlabel("Snapshot Order")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_target_trend_chart(
    path: Path,
    trends: Sequence[MethodComparisonSnapshotTargetTrend],
) -> None:
    by_target: dict[str, list[MethodComparisonSnapshotTargetTrend]] = {}
    for trend in trends:
        by_target.setdefault(trend.target, []).append(trend)
    series = {
        target: [
            trend.required_success_rate
            for trend in sorted(target_trends, key=lambda item: item.snapshot_created_at)
        ]
        for target, target_trends in sorted(by_target.items())
    }
    _save_line_chart(
        path,
        series,
        title="Target Success Rate Over Time",
        ylabel="Success Rate",
        ylim=(0.0, 1.0),
    )


def _save_error_trend_chart(path: Path, entries: Sequence[MethodComparisonIndexEntry]) -> None:
    ordered = sorted(entries, key=lambda entry: entry.created_at)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4))
    if ordered:
        x_values = list(range(1, len(ordered) + 1))
        gradient_values = [
            entry.gradient_max_mse if entry.gradient_max_mse is not None else math.nan
            for entry in ordered
        ]
        agentic_values = [
            entry.agentic_max_mse if entry.agentic_max_mse is not None else math.nan
            for entry in ordered
        ]
        ax.plot(x_values, gradient_values, marker="o", label="gradient")
        ax.plot(x_values, agentic_values, marker="o", label="agentic")
        ax.legend()
    else:
        _render_no_data(ax, "Error Trend")
    ax.set_title("Error Trend")
    ax.set_xlabel("Run Order")
    ax.set_ylabel("Max MSE")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _render_no_data(ax: object, title: str) -> None:
    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _seed_count_values(entries: Sequence[MethodComparisonIndexEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        label = str(entry.seed) if entry.seed is not None else "unknown"
        counts[label] = counts.get(label, 0) + 1
    return counts


def _markdown_scalar(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _snapshot_target_trends(
    snapshot: MethodComparisonSnapshotIndexEntry,
) -> tuple[MethodComparisonSnapshotTargetTrend, ...]:
    try:
        payload = json.loads(Path(snapshot.summary_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ()
    rows = payload.get("latest_by_target", [])
    if not isinstance(rows, list):
        return ()
    trends: list[MethodComparisonSnapshotTargetTrend] = []
    for row in rows:
        if not isinstance(row, dict) or "target" not in row:
            continue
        trends.append(
            MethodComparisonSnapshotTargetTrend(
                snapshot_output_dir=snapshot.output_dir,
                snapshot_created_at=snapshot.created_at,
                target=str(row["target"]),
                runs=int(row.get("runs", 0)),
                seed_count=int(row.get("seed_count", 0)),
                required_success_rate=float(row.get("required_success_rate", 0.0)),
                best_gradient_max_mse=_optional_float(row.get("best_gradient_max_mse")),
                best_agentic_max_mse=_optional_float(row.get("best_agentic_max_mse")),
                latest_status=str(row.get("latest_status", "")),
            )
        )
    return tuple(trends)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _default_train_steps(spec: TargetSpec) -> int:
    return 180 if spec.name == "ln" else 120


def _default_orchestrator_budget(spec: TargetSpec) -> int:
    return 12 if spec.default_depth <= 1 else 24


def _run_gradient_baseline(
    spec: TargetSpec,
    root: Path,
    *,
    seed: int,
    steps: int | None = None,
) -> dict[str, object]:
    result = train_target(
        TrainConfig(
            target=spec.name,
            depth=spec.default_depth,
            seed=seed,
            steps=steps or _default_train_steps(spec),
        )
    )
    manifest = write_train_artifacts(result, root)
    return {
        "status": "ok" if result.success else "failed",
        **result.to_metrics_dict(),
        "output_dir": str(root),
        "manifest_path": manifest.manifest_path,
    }


def _run_agentic_baseline(
    spec: TargetSpec,
    root: Path,
    *,
    seed: int,
    budget: int | None = None,
    beam_width: int = 6,
    seed_count: int = 4,
    max_depth: int | None = None,
) -> dict[str, object]:
    result = run_orchestrator(
        OrchestratorConfig(
            target=spec.name,
            budget=budget or _default_orchestrator_budget(spec),
            beam_width=beam_width,
            seed_count=seed_count,
            seed=seed,
            max_depth=max_depth,
        ),
        root,
    )
    return {
        "status": "ok" if result.success else "failed",
        "success": result.success,
        "best_rpn": result.best_rpn,
        "initial_best_rpn": result.initial_best_rpn,
        "best_score": result.best_score,
        "max_mse": result.best_score["max_mse"],
        "evaluated_candidates": result.evaluated_candidates,
        "generations": result.generations,
        "output_dir": result.output_dir,
        "manifest_path": result.manifest_path,
        "summary_path": result.summary_path,
        "leaderboard_path": result.leaderboard_path,
        "events_path": result.events_path,
    }


def _write_comparison_summary(result: ComparisonResult, root: Path) -> None:
    (root / "summary.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )


def _finalize_comparison(
    target: str,
    root: Path,
    status: PySRStatus,
    eml_summary: dict[str, object],
    pysr_summary: dict[str, object],
) -> ComparisonResult:
    summary_path = root / "summary.json"
    files = [
        ArtifactFile(label="summary", path=str(summary_path), kind="json"),
        ArtifactFile(label="eml-root", path=str(root / "eml"), kind="directory"),
        ArtifactFile(label="eml-manifest", path=str(eml_summary["manifest_path"]), kind="json"),
    ]
    if "output_directory" in pysr_summary:
        files.append(
            ArtifactFile(
                label="pysr-root",
                path=str(pysr_summary["output_directory"]),
                kind="directory",
            )
        )
        equations_path = Path(str(pysr_summary["output_directory"])) / "equations.csv"
        if equations_path.exists():
            files.append(ArtifactFile(label="pysr-equations", path=str(equations_path), kind="csv"))
    manifest = write_artifact_manifest(
        root,
        files=files,
        metadata={
            "kind": "comparison",
            "target": target,
            "pysr_available": status.available,
            "pysr_status": pysr_summary.get("status"),
        },
    )
    result = ComparisonResult(
        target=target,
        output_dir=str(root),
        manifest_path=manifest.manifest_path,
        pysr_status=status,
        eml=eml_summary,
        pysr=pysr_summary,
    )
    _write_comparison_summary(result, root)
    return result


def _write_method_comparison_summary(result: MethodComparisonResult, root: Path) -> None:
    (root / "summary.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )


def _finalize_method_comparison(
    target: str,
    root: Path,
    status: PySRStatus,
    gradient_summary: dict[str, object],
    agentic_summary: dict[str, object],
    pysr_summary: dict[str, object],
) -> MethodComparisonResult:
    summary_path = root / "summary.json"
    files = [
        ArtifactFile(label="summary", path=str(summary_path), kind="json"),
        ArtifactFile(
            label="gradient-root",
            path=str(gradient_summary["output_dir"]),
            kind="directory",
        ),
        ArtifactFile(
            label="gradient-manifest",
            path=str(gradient_summary["manifest_path"]),
            kind="json",
        ),
        ArtifactFile(
            label="agentic-root",
            path=str(agentic_summary["output_dir"]),
            kind="directory",
        ),
        ArtifactFile(
            label="agentic-manifest",
            path=str(agentic_summary["manifest_path"]),
            kind="json",
        ),
        ArtifactFile(
            label="agentic-events",
            path=str(agentic_summary["events_path"]),
            kind="jsonl",
        ),
        ArtifactFile(
            label="agentic-leaderboard",
            path=str(agentic_summary["leaderboard_path"]),
            kind="json",
        ),
    ]
    if "output_directory" in pysr_summary:
        files.append(
            ArtifactFile(
                label="pysr-root",
                path=str(pysr_summary["output_directory"]),
                kind="directory",
            )
        )
        equations_path = Path(str(pysr_summary["output_directory"])) / "equations.csv"
        if equations_path.exists():
            files.append(ArtifactFile(label="pysr-equations", path=str(equations_path), kind="csv"))

    manifest = write_artifact_manifest(
        root,
        files=files,
        metadata={
            "kind": "method-comparison",
            "target": target,
            "pysr_available": status.available,
            "pysr_status": pysr_summary.get("status"),
            "required_success": bool(gradient_summary.get("success"))
            and bool(agentic_summary.get("success")),
        },
    )
    result = MethodComparisonResult(
        target=target,
        output_dir=str(root),
        manifest_path=manifest.manifest_path,
        pysr_status=status,
        gradient=gradient_summary,
        agentic=agentic_summary,
        pysr=pysr_summary,
    )
    _write_method_comparison_summary(result, root)
    return result
