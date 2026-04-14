from eml_lab.targets import get_target, list_targets


def test_list_targets_splits_stable_and_research() -> None:
    stable = list_targets(tier="stable")
    research = list_targets(tier="research")

    assert "exp" in stable
    assert "ln" in stable
    assert "identity" in stable
    assert "square" not in stable
    assert {"square", "mul", "div", "sin"} <= set(research)


def test_list_targets_comparison_filter_excludes_research() -> None:
    compare_targets = list_targets(comparison_eligible=True)

    assert {"exp", "ln", "identity"} <= set(compare_targets)
    assert "sin" not in compare_targets
    assert "mul" not in compare_targets


def test_research_target_metadata_includes_failure_modes() -> None:
    spec = get_target("sin")

    assert spec.tier == "research"
    assert spec.expected_depth == 5
    assert spec.known_route is None
    assert spec.known_tree is None
    assert len(spec.failure_modes) >= 1
