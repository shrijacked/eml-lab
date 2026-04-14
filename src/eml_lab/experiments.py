"""Shared experiment result schemas for Phase 2 tooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ExperimentKind = Literal[
    "train",
    "benchmark",
    "comparison",
    "campaign",
    "orchestration",
    "operator_zoo",
]
ExperimentStatus = Literal["ok", "failed", "unavailable"]


@dataclass(frozen=True)
class ExperimentRecord:
    name: str
    kind: ExperimentKind
    status: ExperimentStatus
    success: bool
    required: bool
    output_dir: str
    summary_path: str
    manifest_path: str | None
    metrics: dict[str, object]

    @property
    def effective_success(self) -> bool:
        return self.success or not self.required

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "success": self.success,
            "required": self.required,
            "effective_success": self.effective_success,
            "output_dir": self.output_dir,
            "summary_path": self.summary_path,
            "manifest_path": self.manifest_path,
            "metrics": self.metrics,
        }
