from eml_lab.targets import exp_tree
from eml_lab.trees import TreeNode
from eml_lab.verify import verify_tree


def test_verify_tree_passes_on_exact_tree() -> None:
    report = verify_tree(exp_tree(), "exp", points=32)

    assert report.passed
    assert report.failure_reason is None


def test_verify_tree_reports_high_error() -> None:
    report = verify_tree(TreeNode.one(), "exp", points=32)

    assert not report.passed
    assert report.failure_reason is not None
    assert "high error" in report.failure_reason
