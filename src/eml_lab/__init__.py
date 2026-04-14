"""EML Lab public API."""

from eml_lab.agentic import OrchestratorConfig, OrchestratorResult, run_orchestrator
from eml_lab.artifacts import ArtifactFile, ArtifactManifest
from eml_lab.benchmarks import BenchmarkResult, run_benchmark_suite
from eml_lab.campaigns import CampaignResult, run_campaign
from eml_lab.comparison import (
    ComparisonResult,
    ComparisonSuiteEntry,
    ComparisonSuiteResult,
    MethodComparisonAggregate,
    MethodComparisonAggregateRow,
    MethodComparisonIndexEntry,
    MethodComparisonResult,
    PySRStatus,
    detect_pysr_environment,
    find_method_comparisons,
    load_method_comparison,
    run_method_comparison,
    run_pysr_compare_suite,
    run_pysr_comparison,
    summarize_method_comparisons,
)
from eml_lab.experiments import ExperimentRecord
from eml_lab.operators import StabilityConfig, StabilityStats, eml_exact, eml_train
from eml_lab.soft_tree import SoftEMLTree, snap_tree
from eml_lab.targets import TargetSpec, get_target, list_targets
from eml_lab.training import TrainConfig, TrainResult, train_target
from eml_lab.trees import TreeNode, from_rpn, to_rpn
from eml_lab.verify import VerificationReport, verify_tree

__all__ = [
    "ArtifactFile",
    "ArtifactManifest",
    "BenchmarkResult",
    "CampaignResult",
    "ComparisonResult",
    "ComparisonSuiteEntry",
    "ComparisonSuiteResult",
    "ExperimentRecord",
    "MethodComparisonAggregate",
    "MethodComparisonAggregateRow",
    "MethodComparisonIndexEntry",
    "MethodComparisonResult",
    "OrchestratorConfig",
    "OrchestratorResult",
    "PySRStatus",
    "SoftEMLTree",
    "StabilityConfig",
    "StabilityStats",
    "detect_pysr_environment",
    "TargetSpec",
    "TrainConfig",
    "TrainResult",
    "TreeNode",
    "VerificationReport",
    "eml_exact",
    "eml_train",
    "find_method_comparisons",
    "from_rpn",
    "get_target",
    "list_targets",
    "load_method_comparison",
    "run_orchestrator",
    "run_campaign",
    "run_method_comparison",
    "run_benchmark_suite",
    "run_pysr_compare_suite",
    "run_pysr_comparison",
    "snap_tree",
    "summarize_method_comparisons",
    "to_rpn",
    "train_target",
    "verify_tree",
]
