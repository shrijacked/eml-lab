from eml_lab.soft_tree import SoftEMLTree, snap_tree
from eml_lab.targets import get_target
from eml_lab.training import TrainConfig, train_target
from eml_lab.trees import rpn_string


def test_snap_tree_chooses_argmax_deterministically() -> None:
    model = SoftEMLTree(depth=1, variables=("x",))
    model.seed_route([("x", "1")], margin=10.0)

    assert rpn_string(snap_tree(model)) == "x 1 E"


def test_exp_training_recovers_exact_tree() -> None:
    result = train_target(TrainConfig(target="exp", depth=1, seed=0, steps=80))

    assert result.success
    assert result.rpn == "x 1 E"


def test_ln_training_recovers_known_tree_with_shallow_refinement() -> None:
    result = train_target(TrainConfig(target="ln", depth=3, seed=0, steps=100))

    assert result.success
    assert result.rpn == "1 1 x E 1 E E"


def test_perturbed_known_route_recovers_identity() -> None:
    spec = get_target("identity")
    result = train_target(
        TrainConfig(
            target=spec.name,
            depth=spec.default_depth,
            seed=0,
            steps=60,
            init_strategy="known_route",
        )
    )

    assert result.success
    assert result.rpn == "1 1 x E 1 E E 1 E"
