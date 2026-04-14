"""Training loop and shallow discrete refinement for EML trees."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch

from eml_lab.artifacts import ArtifactFile, ArtifactManifest, write_artifact_manifest
from eml_lab.operators import StabilityConfig, eml_exact
from eml_lab.soft_tree import SoftEMLTree
from eml_lab.targets import TargetSpec, get_target, sample_inputs
from eml_lab.trees import TreeNode, rpn_string, tree_to_json
from eml_lab.verify import VerificationReport, verify_tree

SnapStrategy = Literal["logits", "best_discrete"]
InitStrategy = Literal["random", "known_route"]


@dataclass(frozen=True)
class TrainConfig:
    target: str = "ln"
    depth: int | None = None
    seed: int = 0
    steps: int = 300
    learning_rate: float = 0.03
    temperature_start: float = 2.0
    temperature_end: float = 0.2
    points: int = 128
    snap_strategy: SnapStrategy = "best_discrete"
    init_strategy: InitStrategy = "random"
    known_route_margin: float = 8.0
    known_route_noise: float = 0.35
    verify_tolerance: float = 1e-20
    stability: StabilityConfig = field(default_factory=StabilityConfig)


@dataclass(frozen=True)
class TrainResult:
    config: TrainConfig
    target: str
    tree: TreeNode
    verification: VerificationReport
    losses: tuple[float, ...]
    final_loss: float
    snap_source: str
    elapsed_seconds: float
    logits_table: tuple[dict[str, float | int | str], ...]

    @property
    def success(self) -> bool:
        return self.verification.passed

    @property
    def rpn(self) -> str:
        return rpn_string(self.tree)

    def to_metrics_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "success": self.success,
            "rpn": self.rpn,
            "final_loss": self.final_loss,
            "snap_source": self.snap_source,
            "elapsed_seconds": self.elapsed_seconds,
            "verification": self.verification.to_dict(),
            "config": asdict(self.config),
        }


def _temperature(config: TrainConfig, step: int) -> float:
    if config.steps <= 1:
        return config.temperature_end
    ratio = step / (config.steps - 1)
    return config.temperature_start * ((config.temperature_end / config.temperature_start) ** ratio)


def _loss(prediction: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    diff = prediction - expected
    return torch.mean(torch.abs(diff) ** 2)


def train_target(config: TrainConfig) -> TrainResult:
    spec = get_target(config.target)
    depth = config.depth or spec.default_depth
    torch.manual_seed(config.seed)
    inputs = sample_inputs(spec, points=config.points)
    expected = spec.function(inputs)

    model = SoftEMLTree(depth, spec.variables)
    if config.init_strategy == "known_route":
        if spec.known_route is None:
            raise ValueError(f"Target {spec.name!r} has no known route")
        model.seed_route(
            spec.known_route,
            margin=config.known_route_margin,
            noise_std=config.known_route_noise,
            seed=config.seed,
        )
    else:
        _seed_random_logits(model, config.seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    start = time.perf_counter()
    for step in range(config.steps):
        optimizer.zero_grad()
        prediction = model(
            inputs, temperature=_temperature(config, step), stability_config=config.stability
        )
        loss = _loss(prediction, expected)
        if not torch.isfinite(loss):
            losses.append(float("inf"))
            break
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    snapped_tree, _ = model.snap()
    snap_source = "logits"
    if config.snap_strategy == "best_discrete":
        logit_report = verify_tree(
            snapped_tree, spec, points=config.points, tolerance=config.verify_tolerance
        )
        if not logit_report.passed:
            candidate_tree, _candidate_loss = best_discrete_tree(spec, depth, points=config.points)
            snapped_tree = candidate_tree
            snap_source = "best_discrete"

    verification = verify_tree(
        snapped_tree,
        spec,
        points=config.points,
        tolerance=config.verify_tolerance,
    )
    elapsed = time.perf_counter() - start
    return TrainResult(
        config=config,
        target=spec.name,
        tree=snapped_tree,
        verification=verification,
        losses=tuple(losses),
        final_loss=losses[-1] if losses else float("inf"),
        snap_source=snap_source,
        elapsed_seconds=elapsed,
        logits_table=tuple(model.logits_table()),
    )


def _seed_random_logits(model: SoftEMLTree, seed: int) -> None:
    generator = torch.Generator()
    generator.manual_seed(seed)
    with torch.no_grad():
        for parameter in [*model.left_logits, *model.right_logits]:
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.2
            )


@dataclass(frozen=True)
class _CandidateState:
    cell_trees: tuple[TreeNode, ...]
    values: tuple[torch.Tensor, ...]


def best_discrete_tree(
    spec: TargetSpec, depth: int, *, points: int = 128
) -> tuple[TreeNode, float]:
    """Exhaustive shallow route refinement for the sequential tree used in v1."""

    inputs = sample_inputs(spec, points=points)
    expected = spec.function(inputs)
    terminal_trees = [TreeNode.one(), *[TreeNode.var(name) for name in spec.variables]]
    terminal_values = tuple(tree.evaluate(inputs) for tree in terminal_trees)
    states = [_CandidateState(cell_trees=(), values=())]
    best_tree = terminal_trees[0]
    best_loss = float("inf")

    for _cell_index in range(depth):
        next_states: list[_CandidateState] = []
        for state in states:
            available_trees = [*terminal_trees, *state.cell_trees]
            available_values = [*terminal_values, *state.values]
            for left_index, left_value in enumerate(available_values):
                for right_index, right_value in enumerate(available_values):
                    tree = TreeNode.eml(available_trees[left_index], available_trees[right_index])
                    try:
                        value = eml_exact(left_value, right_value)
                    except RuntimeError:
                        continue
                    if not torch.isfinite(value).all():
                        continue
                    candidate_loss = float(_loss(value, expected).item())
                    if candidate_loss < best_loss:
                        best_loss = candidate_loss
                        best_tree = tree
                    next_states.append(
                        _CandidateState(
                            cell_trees=(*state.cell_trees, tree),
                            values=(*state.values, value),
                        )
                    )
        states = next_states
    return best_tree, best_loss


def write_train_artifacts(result: TrainResult, output_dir: Path) -> ArtifactManifest:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    tree_path = output_dir / "tree.json"
    loss_path = output_dir / "loss.csv"
    metrics_path.write_text(
        json.dumps(result.to_metrics_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    tree_path.write_text(
        json.dumps(tree_to_json(result.tree), indent=2),
        encoding="utf-8",
    )
    with loss_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss"])
        for step, loss in enumerate(result.losses):
            writer.writerow([step, loss])
    return write_artifact_manifest(
        output_dir,
        files=[
            ArtifactFile(label="metrics", path=str(metrics_path), kind="json"),
            ArtifactFile(label="tree", path=str(tree_path), kind="json"),
            ArtifactFile(label="loss", path=str(loss_path), kind="csv"),
        ],
        metadata={
            "kind": "train",
            "target": result.target,
            "success": result.success,
            "rpn": result.rpn,
        },
    )
