"""Scoring logic for agentic EML route search."""

from __future__ import annotations

from dataclasses import dataclass

from eml_lab.targets import TargetSpec, get_target
from eml_lab.trees import TreeNode
from eml_lab.verify import VerificationReport, verify_tree


@dataclass(frozen=True)
class CandidateScore:
    target: str
    passed: bool
    train_mse: float
    interpolation_mse: float
    extrapolation_mse: float
    max_mse: float
    node_count: int
    complexity_penalty: float
    failure_penalty: float
    total_score: float
    failure_reason: str | None
    verification: VerificationReport

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "passed": self.passed,
            "train_mse": self.train_mse,
            "interpolation_mse": self.interpolation_mse,
            "extrapolation_mse": self.extrapolation_mse,
            "max_mse": self.max_mse,
            "node_count": self.node_count,
            "complexity_penalty": self.complexity_penalty,
            "failure_penalty": self.failure_penalty,
            "total_score": self.total_score,
            "failure_reason": self.failure_reason,
            "verification": self.verification.to_dict(),
        }


def score_tree(
    tree: TreeNode,
    target: TargetSpec | str,
    *,
    points: int = 128,
    tolerance: float = 1e-20,
    complexity_weight: float = 1e-6,
    failure_penalty_value: float = 1e6,
) -> CandidateScore:
    spec = get_target(target) if isinstance(target, str) else target
    verification = verify_tree(tree, spec, points=points, tolerance=tolerance)
    split_map = {split.name: split for split in verification.splits}
    train_mse = split_map["train"].mse
    interpolation_mse = split_map["interpolation"].mse
    extrapolation_mse = split_map["extrapolation"].mse
    complexity_penalty = complexity_weight * tree.node_count()
    failure_penalty = 0.0 if verification.passed else failure_penalty_value
    total_score = (
        train_mse + interpolation_mse + extrapolation_mse + complexity_penalty + failure_penalty
    )
    return CandidateScore(
        target=spec.name,
        passed=verification.passed,
        train_mse=train_mse,
        interpolation_mse=interpolation_mse,
        extrapolation_mse=extrapolation_mse,
        max_mse=verification.max_mse,
        node_count=tree.node_count(),
        complexity_penalty=complexity_penalty,
        failure_penalty=failure_penalty,
        total_score=total_score,
        failure_reason=verification.failure_reason,
        verification=verification,
    )
