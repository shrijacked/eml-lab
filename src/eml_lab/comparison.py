"""Optional baseline comparison against PySR."""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

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
        return self.pysr_status.available

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
        return self.pysr_status.available

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
        return self.pysr_status.available

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


def detect_pysr_environment() -> PySRStatus:
    pysr_installed = importlib.util.find_spec("pysr") is not None
    julia_path = shutil.which("julia")
    julia_found = julia_path is not None
    reason = None
    if not pysr_installed and not julia_found:
        reason = "PySR is not installed and Julia is not on PATH."
    elif not pysr_installed:
        reason = "PySR is not installed."
    elif not julia_found:
        reason = "Julia is not on PATH."
    return PySRStatus(
        available=pysr_installed and julia_found,
        pysr_installed=pysr_installed,
        julia_found=julia_found,
        julia_path=julia_path,
        reason=reason,
        install_hint=(
            "Install with `python -m pip install pysr` and ensure `julia` is on PATH. "
            "PySR installs its Julia dependencies at first import."
        ),
    )


def _timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")


def _load_pysr_regressor() -> type:
    from pysr import PySRRegressor

    return PySRRegressor


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
    filters = {
        "targets": list(targets or []),
        "statuses": list(statuses or []),
        "seeds": list(seeds or []),
        "required_only": required_only,
    }
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
    summary_path = export_root / "summary.json"
    runs_csv_path = export_root / "runs.csv"
    latest_csv_path = export_root / "latest_by_target.csv"
    filters_payload = dict(filters or {})

    summary_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    _write_csv(
        runs_csv_path,
        [entry.to_dict() for entry in report.runs],
    )
    _write_csv(
        latest_csv_path,
        [row.to_dict() for row in report.latest_by_target],
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

    PySRRegressor = _load_pysr_regressor()
    inputs = sample_inputs(spec, points=points)
    variable_name = spec.variables[0]
    x_tensor = inputs[variable_name]
    y_tensor = spec.function(inputs)
    x_imag_max = float(torch.max(torch.abs(x_tensor.imag)).item())
    y_imag_max = float(torch.max(torch.abs(y_tensor.imag)).item())
    if y_imag_max > 1e-12 or x_imag_max > 1e-12:
        return {
            "status": "unsupported",
            "reason": "PySR comparison requires real-valued samples for v1 targets.",
            "x_imag_max": x_imag_max,
            "y_imag_max": y_imag_max,
        }

    X = x_tensor.real.detach().cpu().numpy().reshape(-1, 1)
    y = y_tensor.real.detach().cpu().numpy()
    model_output = root / "pysr"
    model_output.mkdir(parents=True, exist_ok=True)

    model = PySRRegressor(
        niterations=niterations,
        maxsize=maxsize,
        model_selection="best",
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log"],
        progress=False,
        precision=64,
        output_directory=str(model_output),
        run_id=f"eml_lab_{spec.name}_{seed}",
    )

    start = time.perf_counter()
    try:
        model.fit(X, y)
    except Exception as exc:  # pragma: no cover - exercised only with real PySR installed
        return {
            "status": "error",
            "reason": str(exc),
            "elapsed_seconds": time.perf_counter() - start,
        }
    elapsed = time.perf_counter() - start

    equations = getattr(model, "equations_", None)
    equations_records: list[dict[str, object]] | None = None
    if equations is not None:
        if hasattr(equations, "to_dict"):
            equations_records = equations.to_dict(orient="records")
        if hasattr(equations, "to_csv"):
            equations.to_csv(model_output / "equations.csv", index=False)

    best_equation = None
    if hasattr(model, "sympy"):
        try:
            best_equation = str(model.sympy())
        except Exception as exc:  # pragma: no cover - depends on PySR runtime details
            best_equation = f"<sympy export failed: {exc}>"

    return {
        "status": "ok",
        "elapsed_seconds": elapsed,
        "best_equation": best_equation,
        "operators": {
            "binary": ["+", "-", "*", "/"],
            "unary": ["exp", "log"],
        },
        "niterations": niterations,
        "maxsize": maxsize,
        "equations": equations_records,
        "output_directory": str(model_output),
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
