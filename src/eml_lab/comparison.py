"""Optional baseline comparison against PySR."""

from __future__ import annotations

import importlib.util
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

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

    eml_result = train_target(
        TrainConfig(
            target=spec.name,
            depth=spec.default_depth,
            seed=seed,
            steps=180 if spec.name == "ln" else 120,
        )
    )
    eml_dir = root / "eml"
    eml_manifest = write_train_artifacts(eml_result, eml_dir)
    eml_summary = {
        **eml_result.to_metrics_dict(),
        "output_dir": str(eml_dir),
        "manifest_path": eml_manifest.manifest_path,
    }

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
    timestamp = time.strftime("%Y%m%d-%H%M%S")
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
