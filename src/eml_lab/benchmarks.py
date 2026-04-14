"""Internal benchmark suite for EML Lab."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eml_lab.training import TrainConfig, TrainResult, train_target, write_train_artifacts


@dataclass(frozen=True)
class BenchmarkResult:
    suite: str
    output_dir: str
    runs: tuple[dict[str, object], ...]

    @property
    def recovery_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for run in self.runs if run["success"]) / len(self.runs)

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "output_dir": self.output_dir,
            "recovery_rate": self.recovery_rate,
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
    for index, config in enumerate(shallow_suite_configs()):
        result: TrainResult = train_target(config)
        run_dir = root / f"{index:02d}-{config.target}-seed{config.seed}"
        write_train_artifacts(result, run_dir)
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
            }
        )

    benchmark = BenchmarkResult(suite=suite, output_dir=str(root), runs=tuple(run_summaries))
    (root / "summary.json").write_text(json.dumps(benchmark.to_dict(), indent=2), encoding="utf-8")
    return benchmark


def benchmark_table(result: BenchmarkResult) -> list[dict[str, object]]:
    return [dict(run) for run in result.runs]
