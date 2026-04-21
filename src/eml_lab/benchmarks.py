"""Internal benchmark suite for EML Lab."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eml_lab.artifacts import ArtifactFile, write_artifact_manifest
from eml_lab.training import TrainConfig, TrainResult, train_target, write_train_artifacts


@dataclass(frozen=True)
class BenchmarkResult:
    suite: str
    output_dir: str
    manifest_path: str
    runs: tuple[dict[str, object], ...]

    @property
    def recovery_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for run in self.runs if run["success"]) / len(self.runs)

    @property
    def success(self) -> bool:
        return self.recovery_rate == 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "success": self.success,
            "recovery_rate": self.recovery_rate,
            "seed_sensitivity": benchmark_seed_sensitivity_table(self),
            "runs": list(self.runs),
        }


def shallow_suite_configs() -> list[TrainConfig]:
    return [
        TrainConfig(target="exp", depth=1, seed=0, steps=120),
        TrainConfig(target="ln", depth=3, seed=0, steps=180),
        TrainConfig(target="identity", depth=4, seed=0, steps=160, init_strategy="known_route"),
    ]


def run_benchmark_suite(suite: str = "shallow", output_dir: str | Path = "runs") -> BenchmarkResult:
    if suite != "shallow":
        raise ValueError("Only the 'shallow' suite exists in v1")
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    root = Path(output_dir) / f"{suite}-{timestamp}"
    root.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, object]] = []
    artifact_files: list[ArtifactFile] = []
    for index, config in enumerate(shallow_suite_configs()):
        result: TrainResult = train_target(config)
        run_dir = root / f"{index:02d}-{config.target}-seed{config.seed}"
        manifest = write_train_artifacts(result, run_dir)
        run_summaries.append(
            {
                "target": config.target,
                "depth": config.depth,
                "seed": config.seed,
                "success": result.success,
                "rpn": result.rpn,
                "max_mse": result.verification.max_mse,
                "snap_source": result.snap_source,
                "elapsed_seconds": result.elapsed_seconds,
                "run_dir": str(run_dir),
                "manifest_path": manifest.manifest_path,
            }
        )
        artifact_files.extend(
            [
                ArtifactFile(label=f"{config.target}-run", path=str(run_dir), kind="directory"),
                ArtifactFile(
                    label=f"{config.target}-manifest",
                    path=manifest.manifest_path,
                    kind="json",
                ),
            ]
        )

    summary_path = root / "summary.json"
    manifest = write_artifact_manifest(
        root,
        files=[
            ArtifactFile(label="summary", path=str(summary_path), kind="json"),
            *artifact_files,
        ],
        metadata={"kind": "benchmark", "suite": suite},
    )
    benchmark = BenchmarkResult(
        suite=suite,
        output_dir=str(root),
        manifest_path=manifest.manifest_path,
        runs=tuple(run_summaries),
    )
    summary_path.write_text(json.dumps(benchmark.to_dict(), indent=2), encoding="utf-8")
    return benchmark


def benchmark_table(result: BenchmarkResult) -> list[dict[str, object]]:
    return [dict(run) for run in result.runs]


def benchmark_seed_sensitivity_table(result: BenchmarkResult) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for run in result.runs:
        grouped.setdefault(str(run["target"]), []).append(dict(run))

    rows: list[dict[str, object]] = []
    for target, runs in sorted(grouped.items()):
        max_mses = [float(run["max_mse"]) for run in runs if run.get("max_mse") is not None]
        runtimes = [
            float(run["elapsed_seconds"])
            for run in runs
            if run.get("elapsed_seconds") is not None
        ]
        best_mse = min(max_mses) if max_mses else float("nan")
        worst_mse = max(max_mses) if max_mses else float("nan")
        fastest = min(runtimes) if runtimes else float("nan")
        slowest = max(runtimes) if runtimes else float("nan")
        seed_values = {run.get("seed") for run in runs}
        rows.append(
            {
                "target": target,
                "run_count": len(runs),
                "seed_count": len(seed_values),
                "seeds": ", ".join(str(seed) for seed in sorted(seed_values, key=str)),
                "success_rate": sum(1 for run in runs if run.get("success")) / len(runs),
                "best_max_mse": best_mse,
                "worst_max_mse": worst_mse,
                "mse_spread": worst_mse - best_mse,
                "fastest_seconds": fastest,
                "slowest_seconds": slowest,
                "runtime_spread_seconds": slowest - fastest,
            }
        )
    return rows
