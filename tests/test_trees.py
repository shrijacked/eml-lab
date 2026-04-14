import torch

from eml_lab.targets import exp_tree, get_target, sample_inputs
from eml_lab.trees import from_rpn, rpn_string, to_networkx


def test_discrete_tree_evaluates_direct_formula() -> None:
    spec = get_target("exp")
    inputs = sample_inputs(spec, points=32)
    predicted = exp_tree().evaluate(inputs)

    assert torch.allclose(predicted, torch.exp(inputs["x"]), atol=1e-12, rtol=1e-12)


def test_rpn_round_trip_preserves_output() -> None:
    tree = exp_tree()
    parsed = from_rpn(rpn_string(tree))
    spec = get_target("exp")
    inputs = sample_inputs(spec, points=16)

    assert rpn_string(parsed) == "x 1 E"
    assert torch.allclose(parsed.evaluate(inputs), tree.evaluate(inputs), atol=1e-12, rtol=1e-12)


def test_networkx_conversion_has_labels() -> None:
    graph = to_networkx(exp_tree())

    labels = {data["label"] for _, data in graph.nodes(data=True)}
    assert {"eml", "x", "1"}.issubset(labels)
