"""Exact verification of snapped EML trees."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from eml_lab.targets import TargetSpec, get_target, sample_inputs
from eml_lab.trees import TreeNode


@dataclass(frozen=True)
class VerificationSplit:
    name: str
    mse: float
    max_abs_error: float
    finite: bool
    passed: bool
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationReport:
    target: str
    tolerance: float
    splits: tuple[VerificationSplit, ...]

    @property
    def passed(self) -> bool:
        return all(split.passed for split in self.splits)

    @property
    def max_mse(self) -> float:
        return max(split.mse for split in self.splits)

    @property
    def failure_reason(self) -> str | None:
        for split in self.splits:
            if not split.passed:
                return split.failure_reason
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "max_mse": self.max_mse,
            "failure_reason": self.failure_reason,
            "splits": [split.to_dict() for split in self.splits],
        }


def _split_report(
    name: str,
    prediction: torch.Tensor,
    expected: torch.Tensor,
    tolerance: float,
) -> VerificationSplit:
    finite = bool(torch.isfinite(prediction).all().item() and torch.isfinite(expected).all().item())
    if not finite:
        return VerificationSplit(
            name=name,
            mse=float("inf"),
            max_abs_error=float("inf"),
            finite=False,
            passed=False,
            failure_reason="non-finite value from exact EML evaluation",
        )
    diff = prediction - expected
    mse = float(torch.mean(torch.abs(diff) ** 2).real.item())
    max_abs_error = float(torch.max(torch.abs(diff)).real.item())
    passed = mse <= tolerance
    reason = None if passed else f"high error: mse={mse:.3e} > tolerance={tolerance:.3e}"
    return VerificationSplit(
        name=name,
        mse=mse,
        max_abs_error=max_abs_error,
        finite=True,
        passed=passed,
        failure_reason=reason,
    )


def verify_tree(
    tree: TreeNode,
    target: TargetSpec | str,
    domain: dict[str, tuple[float, float]] | None = None,
    *,
    points: int = 256,
    tolerance: float = 1e-20,
) -> VerificationReport:
    """Verify a snapped tree against train/interpolation/extrapolation grids."""

    spec = get_target(target) if isinstance(target, str) else target
    domains = [
        ("train", domain or spec.train_domain),
        ("interpolation", spec.interpolation_domain),
        ("extrapolation", spec.extrapolation_domain),
    ]
    splits: list[VerificationSplit] = []
    for name, selected_domain in domains:
        inputs = sample_inputs(spec, selected_domain, points=points)
        prediction = tree.evaluate(inputs)
        expected = spec.function(inputs)
        splits.append(_split_report(name, prediction, expected, tolerance))
    return VerificationReport(spec.name, tolerance, tuple(splits))
