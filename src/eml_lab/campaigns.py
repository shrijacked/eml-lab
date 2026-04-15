"""Campaign suites built on top of shared experiment schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from eml_lab.agentic import OrchestratorConfig, run_orchestrator
from eml_lab.artifacts import ArtifactFile, write_artifact_manifest
from eml_lab.benchmarks import run_benchmark_suite
from eml_lab.comparison import run_pysr_comparison
from eml_lab.experiments import ExperimentRecord
from eml_lab.operator_zoo import OperatorZooConfig, run_operator_zoo
from eml_lab.targets import get_target
from eml_lab.training import TrainConfig, train_target, write_train_artifacts

CampaignStepKind = Literal["benchmark", "comparison", "operator_zoo", "orchestration", "train"]


@dataclass(frozen=True)
class CampaignStep:
    name: str
    kind: CampaignStepKind
    required: bool = True
    suite: str | None = None
    target: str | None = None
    budget: int | None = None
    beam_width: int | None = None
    seed_count: int | None = None
    seed: int | None = None
    max_depth: int | None = None
    depth: int | None = None
    steps: int | None = None
    snap_strategy: str | None = None
    init_strategy: str | None = None
    grid_points: int | None = None
    epsilon: float | None = None


@dataclass(frozen=True)
class CampaignSpec:
    name: str
    description: str
    steps: tuple[CampaignStep, ...]


@dataclass(frozen=True)
class CampaignResult:
    suite: str
    output_dir: str
    manifest_path: str
    runs: tuple[ExperimentRecord, ...]

    @property
    def success(self) -> bool:
        return all(run.effective_success for run in self.runs)

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "success": self.success,
            "runs": [run.to_dict() for run in self.runs],
        }


CAMPAIGNS: dict[str, CampaignSpec] = {
    "phase2": CampaignSpec(
        name="phase2",
        description=(
            "Umbrella Phase 2 smoke covering foundation artifacts, agentic search, "
            "research targets, and operator-zoo checks."
        ),
        steps=(
            CampaignStep(name="shallow-benchmark", kind="benchmark", suite="shallow"),
            CampaignStep(
                name="compare-exp",
                kind="comparison",
                target="exp",
                required=False,
            ),
            CampaignStep(
                name="compare-ln",
                kind="comparison",
                target="ln",
                required=False,
            ),
            CampaignStep(
                name="orchestrate-exp",
                kind="orchestration",
                target="exp",
                budget=12,
                beam_width=4,
                seed_count=3,
                seed=0,
            ),
            CampaignStep(
                name="orchestrate-ln",
                kind="orchestration",
                target="ln",
                budget=24,
                beam_width=6,
                seed_count=4,
                seed=0,
            ),
            CampaignStep(
                name="research-square",
                kind="train",
                target="square",
                required=False,
                depth=4,
                steps=40,
                snap_strategy="logits",
                init_strategy="random",
            ),
            CampaignStep(
                name="research-mul",
                kind="train",
                target="mul",
                required=False,
                depth=4,
                steps=40,
                snap_strategy="logits",
                init_strategy="random",
            ),
            CampaignStep(
                name="research-div",
                kind="train",
                target="div",
                required=False,
                depth=5,
                steps=40,
                snap_strategy="logits",
                init_strategy="random",
            ),
            CampaignStep(
                name="research-sin",
                kind="train",
                target="sin",
                required=False,
                depth=5,
                steps=48,
                snap_strategy="logits",
                init_strategy="random",
            ),
            CampaignStep(
                name="operator-zoo",
                kind="operator_zoo",
                grid_points=17,
                epsilon=1e-8,
            ),
        ),
    ),
    "phase2-foundation": CampaignSpec(
        name="phase2-foundation",
        description="Shared artifact smoke covering benchmarks and optional comparisons.",
        steps=(
            CampaignStep(name="shallow-benchmark", kind="benchmark", suite="shallow"),
            CampaignStep(
                name="compare-exp",
                kind="comparison",
                target="exp",
                required=False,
            ),
            CampaignStep(
                name="compare-ln",
                kind="comparison",
                target="ln",
                required=False,
            ),
        ),
    ),
    "phase2-agentic": CampaignSpec(
        name="phase2-agentic",
        description="Agentic shallow-search suite with leaderboard and trace artifacts.",
        steps=(
            CampaignStep(name="shallow-benchmark", kind="benchmark", suite="shallow"),
            CampaignStep(
                name="orchestrate-exp",
                kind="orchestration",
                target="exp",
                budget=12,
                beam_width=4,
                seed_count=3,
                seed=0,
            ),
            CampaignStep(
                name="orchestrate-ln",
                kind="orchestration",
                target="ln",
                budget=24,
                beam_width=6,
                seed_count=4,
                seed=0,
            ),
        ),
    ),
    "phase2-research": CampaignSpec(
        name="phase2-research",
        description="Research-tier hard targets with explicit failure reporting.",
        steps=(
            CampaignStep(
                name="research-square",
                kind="train",
                target="square",
                required=False,
                depth=4,
                steps=40,
                snap_strategy="logits",
                init_strategy="random",
            ),
            CampaignStep(
                name="research-mul",
                kind="train",
                target="mul",
                required=False,
                depth=4,
                steps=40,
                snap_strategy="logits",
                init_strategy="random",
            ),
            CampaignStep(
                name="research-div",
                kind="train",
                target="div",
                required=False,
                depth=5,
                steps=40,
                snap_strategy="logits",
                init_strategy="random",
            ),
            CampaignStep(
                name="research-sin",
                kind="train",
                target="sin",
                required=False,
                depth=5,
                steps=48,
                snap_strategy="logits",
                init_strategy="random",
            ),
        ),
    ),
    "phase2-operator-zoo": CampaignSpec(
        name="phase2-operator-zoo",
        description="Numerical research suite for EML-like operator variants.",
        steps=(
            CampaignStep(
                name="operator-zoo",
                kind="operator_zoo",
                grid_points=17,
                epsilon=1e-8,
            ),
        ),
    ),
}


def list_campaigns() -> list[str]:
    return sorted(CAMPAIGNS)


def get_campaign(name: str) -> CampaignSpec:
    try:
        return CAMPAIGNS[name]
    except KeyError as exc:
        valid = ", ".join(list_campaigns())
        raise KeyError(f"Unknown campaign {name!r}. Valid campaigns: {valid}") from exc


def run_campaign(
    name: str = "phase2-foundation", output_dir: str | Path = "runs"
) -> CampaignResult:
    spec = get_campaign(name)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    root = Path(output_dir) / f"campaign-{spec.name}-{timestamp}"
    root.mkdir(parents=True, exist_ok=True)

    records: list[ExperimentRecord] = []
    artifact_files: list[ArtifactFile] = []
    for index, step in enumerate(spec.steps):
        if step.kind == "benchmark":
            if step.suite is None:
                raise ValueError(f"Benchmark step {step.name!r} is missing a suite")
            step_root = root / f"{index:02d}-bench-{step.suite}"
            result = run_benchmark_suite(step.suite, step_root)
            summary_path = str(Path(result.output_dir) / "summary.json")
            record = ExperimentRecord(
                name=step.name,
                kind="benchmark",
                status="ok" if result.success else "failed",
                success=result.success,
                required=step.required,
                output_dir=result.output_dir,
                summary_path=summary_path,
                manifest_path=result.manifest_path,
                metrics=result.to_dict(),
            )
        elif step.kind == "comparison":
            if step.target is None:
                raise ValueError(f"Comparison step {step.name!r} is missing a target")
            step_root = root / f"{index:02d}-compare-{step.target}"
            result = run_pysr_comparison(step.target, step_root)
            status = "ok"
            success = result.success
            if not result.available:
                status = "unavailable"
            elif result.pysr.get("status") != "ok":
                status = "failed"
            record = ExperimentRecord(
                name=step.name,
                kind="comparison",
                status=status,
                success=success,
                required=step.required,
                output_dir=result.output_dir,
                summary_path=str(Path(result.output_dir) / "summary.json"),
                manifest_path=result.manifest_path,
                metrics=result.to_dict(),
            )
        elif step.kind == "train":
            if step.target is None:
                raise ValueError(f"Train step {step.name!r} is missing a target")
            target_spec = get_target(step.target)
            result = train_target(
                TrainConfig(
                    target=step.target,
                    depth=step.depth or target_spec.default_depth,
                    seed=step.seed or 0,
                    steps=step.steps or 40,
                    snap_strategy=step.snap_strategy or "logits",
                    init_strategy=step.init_strategy or "random",
                )
            )
            step_root = root / f"{index:02d}-train-{step.target}"
            manifest = write_train_artifacts(result, step_root)
            metrics = {
                **result.to_metrics_dict(),
                "target_tier": target_spec.tier,
                "expected_depth": target_spec.expected_depth,
                "failure_modes": list(target_spec.failure_modes),
                "notes": target_spec.notes,
            }
            record = ExperimentRecord(
                name=step.name,
                kind="train",
                status="ok" if result.success else "failed",
                success=result.success,
                required=step.required,
                output_dir=str(step_root),
                summary_path=str(step_root / "metrics.json"),
                manifest_path=manifest.manifest_path,
                metrics=metrics,
            )
        elif step.kind == "operator_zoo":
            step_root = root / f"{index:02d}-operator-zoo"
            result = run_operator_zoo(
                step_root,
                OperatorZooConfig(
                    grid_points=step.grid_points or 17,
                    epsilon=step.epsilon or 1e-8,
                ),
            )
            record = ExperimentRecord(
                name=step.name,
                kind="operator_zoo",
                status="ok",
                success=True,
                required=step.required,
                output_dir=result.output_dir,
                summary_path=result.summary_path,
                manifest_path=result.manifest_path,
                metrics=result.to_dict(),
            )
        else:
            if step.target is None:
                raise ValueError(f"Orchestration step {step.name!r} is missing a target")
            step_root = root / f"{index:02d}-orchestrate-{step.target}"
            result = run_orchestrator(
                OrchestratorConfig(
                    target=step.target,
                    budget=step.budget or 24,
                    beam_width=step.beam_width or 6,
                    seed_count=step.seed_count or 4,
                    seed=step.seed or 0,
                    max_depth=step.max_depth,
                ),
                step_root,
            )
            record = ExperimentRecord(
                name=step.name,
                kind="orchestration",
                status="ok" if result.success else "failed",
                success=result.success,
                required=step.required,
                output_dir=result.output_dir,
                summary_path=result.summary_path,
                manifest_path=result.manifest_path,
                metrics=result.to_dict(),
            )
        records.append(record)
        artifact_files.extend(
            [
                ArtifactFile(label=f"{step.name}-summary", path=record.summary_path, kind="json"),
                ArtifactFile(label=f"{step.name}-root", path=record.output_dir, kind="directory"),
            ]
        )
        if record.manifest_path is not None:
            artifact_files.append(
                ArtifactFile(label=f"{step.name}-manifest", path=record.manifest_path, kind="json")
            )

    summary_path = root / "summary.json"
    manifest = write_artifact_manifest(
        root,
        files=[
            ArtifactFile(label="summary", path=str(summary_path), kind="json"),
            *artifact_files,
        ],
        metadata={
            "suite": spec.name,
            "success": all(record.effective_success for record in records),
            "description": spec.description,
        },
    )
    result = CampaignResult(
        suite=spec.name,
        output_dir=str(root),
        manifest_path=manifest.manifest_path,
        runs=tuple(records),
    )
    summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
