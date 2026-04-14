"""Optional baseline comparison against PySR."""

from __future__ import annotations

import importlib.util
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from eml_lab.targets import TargetSpec, get_target, sample_inputs
from eml_lab.training import TrainConfig, train_target


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
    pysr_status: PySRStatus
    eml: dict[str, object]
    pysr: dict[str, object]

    @property
    def available(self) -> bool:
        return self.pysr_status.available

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "output_dir": self.output_dir,
            "available": self.available,
            "pysr_status": self.pysr_status.to_dict(),
            "eml": self.eml,
            "pysr": self.pysr,
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
    eml_summary = eml_result.to_metrics_dict()

    status = detect_pysr_environment()
    pysr_summary: dict[str, object]
    if not status.available:
        pysr_summary = {
            "status": "unavailable",
            "reason": status.reason,
            "install_hint": status.install_hint,
        }
        result = ComparisonResult(spec.name, str(root), status, eml_summary, pysr_summary)
        _write_comparison_summary(result, root)
        return result

    pysr_summary = _run_available_pysr(spec, root, points, niterations, maxsize, seed)
    result = ComparisonResult(spec.name, str(root), status, eml_summary, pysr_summary)
    _write_comparison_summary(result, root)
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
