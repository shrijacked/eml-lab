"""EML Lab public API."""

from eml_lab.benchmarks import BenchmarkResult, run_benchmark_suite
from eml_lab.operators import StabilityConfig, StabilityStats, eml_exact, eml_train
from eml_lab.soft_tree import SoftEMLTree, snap_tree
from eml_lab.targets import TargetSpec, get_target, list_targets
from eml_lab.training import TrainConfig, TrainResult, train_target
from eml_lab.trees import TreeNode, from_rpn, to_rpn
from eml_lab.verify import VerificationReport, verify_tree

__all__ = [
    "BenchmarkResult",
    "SoftEMLTree",
    "StabilityConfig",
    "StabilityStats",
    "TargetSpec",
    "TrainConfig",
    "TrainResult",
    "TreeNode",
    "VerificationReport",
    "eml_exact",
    "eml_train",
    "from_rpn",
    "get_target",
    "list_targets",
    "run_benchmark_suite",
    "snap_tree",
    "to_rpn",
    "train_target",
    "verify_tree",
]
