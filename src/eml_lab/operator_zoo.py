"""Numerical research harness for EML-like operator variants."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from eml_lab.artifacts import ArtifactFile, write_artifact_manifest
from eml_lab.operators import COMPLEX_DTYPE

OperatorFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


@dataclass(frozen=True)
class OperatorZooCandidate:
    name: str
    expression: str
    description: str
    requires_distinguished_constant: bool
    exact_paper_operator: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorZooConfig:
    grid_points: int = 17
    epsilon: float = 1e-8
    real_span: float = 3.0
    imag_span: float = 1.5

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorZooEntry:
    rank: int
    candidate: OperatorZooCandidate
    sample_count: int
    finite_rate: float
    safe_finite_rate: float
    gradient_finite_rate: float
    mse_to_exact: float | None
    max_abs_error_to_exact: float | None
    max_abs_output: float | None
    stability_score: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate"] = self.candidate.to_dict()
        return payload


@dataclass(frozen=True)
class OperatorZooResult:
    output_dir: str
    manifest_path: str
    summary_path: str
    candidates_csv_path: str
    report_path: str
    plot_path: str
    config: OperatorZooConfig
    entries: tuple[OperatorZooEntry, ...]

    @property
    def best(self) -> OperatorZooEntry | None:
        return self.entries[0] if self.entries else None

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "summary_path": self.summary_path,
            "candidates_csv_path": self.candidates_csv_path,
            "report_path": self.report_path,
            "plot_path": self.plot_path,
            "config": self.config.to_dict(),
            "best": self.best.to_dict() if self.best is not None else None,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def candidate_registry() -> tuple[tuple[OperatorZooCandidate, OperatorFn], ...]:
    return (
        (
            OperatorZooCandidate(
                name="eml_exact",
                expression="exp(x) - log(y)",
                description=(
                    "Faithful paper operator. Kept as the reference, not a stabilized variant."
                ),
                requires_distinguished_constant=True,
                exact_paper_operator=True,
            ),
            _op_eml_exact,
        ),
        (
            OperatorZooCandidate(
                name="abs_log_guard",
                expression="exp(x) - log(abs(y) + eps)",
                description="Real-valued magnitude guard used to avoid log singularities.",
                requires_distinguished_constant=True,
                exact_paper_operator=False,
            ),
            _op_abs_log_guard,
        ),
        (
            OperatorZooCandidate(
                name="shifted_log",
                expression="exp(x) - log(y + 1)",
                description="Shifts the log singularity away from zero.",
                requires_distinguished_constant=True,
                exact_paper_operator=False,
            ),
            _op_shifted_log,
        ),
        (
            OperatorZooCandidate(
                name="softplus_abs_log",
                expression="exp(x) - log(softplus(abs(y)) + eps)",
                description="Smooth positive magnitude guard for the logarithm.",
                requires_distinguished_constant=True,
                exact_paper_operator=False,
            ),
            _op_softplus_abs_log,
        ),
        (
            OperatorZooCandidate(
                name="bounded_exp_abs_log",
                expression="exp(tanh(x.real) + i*tanh(x.imag)) - log(abs(y) + eps)",
                description="Bounded exponential input plus magnitude-guarded logarithm.",
                requires_distinguished_constant=True,
                exact_paper_operator=False,
            ),
            _op_bounded_exp_abs_log,
        ),
    )


def run_operator_zoo(
    output_dir: str | Path = "runs",
    config: OperatorZooConfig | None = None,
) -> OperatorZooResult:
    cfg = config or OperatorZooConfig()
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    root = Path(output_dir) / f"operator-zoo-{timestamp}"
    root.mkdir(parents=True, exist_ok=True)

    x, y, safe_mask = _stress_grid(cfg)
    exact = _op_eml_exact(x, y, cfg.epsilon)
    entries = [
        _evaluate_candidate(candidate, fn, x, y, safe_mask, exact, cfg)
        for candidate, fn in candidate_registry()
    ]
    ranked = tuple(
        entry
        for rank, entry in enumerate(
            sorted(entries, key=lambda item: item.stability_score, reverse=True),
            start=1,
        )
        for entry in (_rerank_entry(entry, rank),)
    )

    summary_path = root / "summary.json"
    candidates_csv_path = root / "candidates.csv"
    report_path = root / "report.md"
    plot_path = root / "stability_scores.png"
    result_without_manifest = OperatorZooResult(
        output_dir=str(root),
        manifest_path=str(root / "manifest.json"),
        summary_path=str(summary_path),
        candidates_csv_path=str(candidates_csv_path),
        report_path=str(report_path),
        plot_path=str(plot_path),
        config=cfg,
        entries=ranked,
    )
    summary_path.write_text(
        json.dumps(result_without_manifest.to_dict(), indent=2),
        encoding="utf-8",
    )
    _write_entries_csv(candidates_csv_path, ranked)
    report_path.write_text(_render_report(result_without_manifest), encoding="utf-8")
    _write_score_plot(plot_path, ranked)
    manifest = write_artifact_manifest(
        root,
        files=[
            ArtifactFile(label="summary", path=str(summary_path), kind="json"),
            ArtifactFile(label="candidates", path=str(candidates_csv_path), kind="csv"),
            ArtifactFile(label="report", path=str(report_path), kind="markdown"),
            ArtifactFile(label="stability-scores", path=str(plot_path), kind="png"),
        ],
        metadata={
            "kind": "operator-zoo",
            "candidate_count": len(ranked),
            "grid_points": cfg.grid_points,
            "epsilon": cfg.epsilon,
        },
    )
    result = OperatorZooResult(
        output_dir=str(root),
        manifest_path=manifest.manifest_path,
        summary_path=str(summary_path),
        candidates_csv_path=str(candidates_csv_path),
        report_path=str(report_path),
        plot_path=str(plot_path),
        config=cfg,
        entries=ranked,
    )
    summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def _evaluate_candidate(
    candidate: OperatorZooCandidate,
    fn: OperatorFn,
    x: torch.Tensor,
    y: torch.Tensor,
    safe_mask: torch.Tensor,
    exact: torch.Tensor,
    config: OperatorZooConfig,
) -> OperatorZooEntry:
    try:
        output = fn(x, y, config.epsilon)
    except Exception:
        return OperatorZooEntry(
            rank=0,
            candidate=candidate,
            sample_count=x.numel(),
            finite_rate=0.0,
            safe_finite_rate=0.0,
            gradient_finite_rate=0.0,
            mse_to_exact=None,
            max_abs_error_to_exact=None,
            max_abs_output=None,
            stability_score=0.0,
        )

    finite_mask = _finite_complex(output)
    exact_finite_mask = _finite_complex(exact)
    safe_finite_mask = safe_mask & finite_mask & exact_finite_mask
    finite_rate = _rate(finite_mask)
    safe_finite_rate = _rate(finite_mask[safe_mask])
    mse_to_exact = None
    max_abs_error_to_exact = None
    if bool(torch.any(safe_finite_mask).item()):
        diff = output[safe_finite_mask] - exact[safe_finite_mask]
        squared = torch.abs(diff) ** 2
        mse_to_exact = float(torch.mean(squared).item())
        max_abs_error_to_exact = float(torch.max(torch.abs(diff)).item())
    finite_values = output[finite_mask]
    max_abs_output = (
        float(torch.max(torch.abs(finite_values)).item()) if finite_values.numel() else None
    )
    gradient_finite_rate = _gradient_finite_rate(fn, x, y, config.epsilon)
    stability_score = _stability_score(
        finite_rate=finite_rate,
        safe_finite_rate=safe_finite_rate,
        gradient_finite_rate=gradient_finite_rate,
        mse_to_exact=mse_to_exact,
    )
    return OperatorZooEntry(
        rank=0,
        candidate=candidate,
        sample_count=x.numel(),
        finite_rate=finite_rate,
        safe_finite_rate=safe_finite_rate,
        gradient_finite_rate=gradient_finite_rate,
        mse_to_exact=mse_to_exact,
        max_abs_error_to_exact=max_abs_error_to_exact,
        max_abs_output=max_abs_output,
        stability_score=stability_score,
    )


def _rerank_entry(entry: OperatorZooEntry, rank: int) -> OperatorZooEntry:
    return OperatorZooEntry(
        rank=rank,
        candidate=entry.candidate,
        sample_count=entry.sample_count,
        finite_rate=entry.finite_rate,
        safe_finite_rate=entry.safe_finite_rate,
        gradient_finite_rate=entry.gradient_finite_rate,
        mse_to_exact=entry.mse_to_exact,
        max_abs_error_to_exact=entry.max_abs_error_to_exact,
        max_abs_output=entry.max_abs_output,
        stability_score=entry.stability_score,
    )


def _stress_grid(config: OperatorZooConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    real = torch.linspace(-config.real_span, config.real_span, config.grid_points)
    imag = torch.linspace(-config.imag_span, config.imag_span, config.grid_points)
    real_mesh, imag_mesh = torch.meshgrid(real, imag, indexing="ij")
    x = torch.complex(real_mesh.reshape(-1), imag_mesh.reshape(-1)).to(COMPLEX_DTYPE)
    y = torch.complex(imag_mesh.reshape(-1), real_mesh.reshape(-1)).to(COMPLEX_DTYPE)
    safe_mask = torch.abs(y) > config.epsilon
    return x, y, safe_mask


def _op_eml_exact(x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    del epsilon
    return torch.exp(x) - torch.log(y)


def _op_abs_log_guard(x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.exp(x) - torch.log(torch.abs(y) + epsilon)


def _op_shifted_log(x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    del epsilon
    return torch.exp(x) - torch.log(y + 1)


def _op_softplus_abs_log(x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.exp(x) - torch.log(torch.nn.functional.softplus(torch.abs(y)) + epsilon)


def _op_bounded_exp_abs_log(x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    bounded_x = torch.complex(torch.tanh(x.real), torch.tanh(x.imag))
    return torch.exp(bounded_x) - torch.log(torch.abs(y) + epsilon)


def _finite_complex(value: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(value.real) & torch.isfinite(value.imag)


def _rate(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(torch.count_nonzero(mask).item() / mask.numel())


def _gradient_finite_rate(
    fn: OperatorFn,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
) -> float:
    x_var = x.detach().clone().requires_grad_(True)
    y_var = y.detach().clone().requires_grad_(True)
    try:
        output = fn(x_var, y_var, epsilon)
        loss_terms = torch.nan_to_num(
            torch.abs(output) ** 2,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        loss = torch.mean(loss_terms)
        loss.backward()
    except Exception:
        return 0.0
    gradients = [gradient for gradient in (x_var.grad, y_var.grad) if gradient is not None]
    if not gradients:
        return 0.0
    finite = torch.cat([_finite_complex(gradient).reshape(-1) for gradient in gradients])
    return _rate(finite)


def _stability_score(
    *,
    finite_rate: float,
    safe_finite_rate: float,
    gradient_finite_rate: float,
    mse_to_exact: float | None,
) -> float:
    if mse_to_exact is None:
        exactness = 0.0
    elif mse_to_exact <= 0:
        exactness = 1.0
    else:
        exactness = 1.0 / (1.0 + max(0.0, math.log10(mse_to_exact + 1e-300) + 12.0))
    return (
        0.35 * finite_rate
        + 0.25 * safe_finite_rate
        + 0.25 * gradient_finite_rate
        + 0.15 * exactness
    )


def _write_entries_csv(path: Path, entries: tuple[OperatorZooEntry, ...]) -> None:
    rows = [
        {
            "rank": entry.rank,
            "name": entry.candidate.name,
            "expression": entry.candidate.expression,
            "exact_paper_operator": entry.candidate.exact_paper_operator,
            "finite_rate": entry.finite_rate,
            "safe_finite_rate": entry.safe_finite_rate,
            "gradient_finite_rate": entry.gradient_finite_rate,
            "mse_to_exact": entry.mse_to_exact,
            "max_abs_error_to_exact": entry.max_abs_error_to_exact,
            "max_abs_output": entry.max_abs_output,
            "stability_score": entry.stability_score,
        }
        for entry in entries
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(result: OperatorZooResult) -> str:
    lines = [
        "# Operator Zoo Report",
        "",
        "This is a numerical research harness for EML-like operator variants. "
        "It does not replace the exact paper operator used by the verifier.",
        "",
        f"Grid points: `{result.config.grid_points}`",
        f"Epsilon: `{result.config.epsilon}`",
        "",
        "## Candidates",
        (
            "| rank | name | exact_paper_operator | finite_rate | safe_finite_rate | "
            "gradient_finite_rate | mse_to_exact | stability_score |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in result.entries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(entry.rank),
                    f"`{entry.candidate.name}`",
                    "yes" if entry.candidate.exact_paper_operator else "no",
                    f"{entry.finite_rate:.2%}",
                    f"{entry.safe_finite_rate:.2%}",
                    f"{entry.gradient_finite_rate:.2%}",
                    _scalar(entry.mse_to_exact),
                    f"{entry.stability_score:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- `eml_exact` is the faithful reference for proofs and verification.",
            "- Stabilized variants are candidates for training heuristics or future experiments.",
            (
                "- Scores combine output finiteness, safe-domain finiteness, gradient "
                "finiteness, and closeness to exact EML on the safe subset."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _write_score_plot(path: Path, entries: tuple[OperatorZooEntry, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [entry.candidate.name for entry in entries]
    scores = [entry.stability_score for entry in entries]
    ax.bar(labels, scores, color="#4C78A8")
    ax.set_title("Operator Zoo Stability Scores")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _scalar(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
