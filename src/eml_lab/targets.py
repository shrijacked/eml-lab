"""Target functions and known shallow EML fixtures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from eml_lab.operators import COMPLEX_DTYPE
from eml_lab.trees import TreeNode

TargetFunction = Callable[[dict[str, torch.Tensor]], torch.Tensor]
Route = list[tuple[str, str]]


@dataclass(frozen=True)
class TargetSpec:
    name: str
    display_name: str
    variables: tuple[str, ...]
    train_domain: dict[str, tuple[float, float]]
    interpolation_domain: dict[str, tuple[float, float]]
    extrapolation_domain: dict[str, tuple[float, float]]
    default_depth: int
    function: TargetFunction
    known_tree: TreeNode
    known_route: Route | None
    notes: str


def _x(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return inputs["x"].to(dtype=COMPLEX_DTYPE)


def _exp(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.exp(_x(inputs))


def _ln(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.log(_x(inputs))


def _identity(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return _x(inputs)


def one_tree() -> TreeNode:
    return TreeNode.one()


def e_tree() -> TreeNode:
    return TreeNode.eml(TreeNode.one(), TreeNode.one())


def exp_tree() -> TreeNode:
    return TreeNode.eml(TreeNode.var("x"), TreeNode.one())


def ln_tree() -> TreeNode:
    inner = TreeNode.eml(TreeNode.one(), TreeNode.var("x"))
    middle = TreeNode.eml(inner, TreeNode.one())
    return TreeNode.eml(TreeNode.one(), middle)


def identity_tree() -> TreeNode:
    return TreeNode.eml(ln_tree(), TreeNode.one())


def x_minus_one_tree() -> TreeNode:
    return TreeNode.eml(ln_tree(), e_tree())


def _zero_tree() -> TreeNode:
    return TreeNode.eml(TreeNode.one(), TreeNode.eml(e_tree(), TreeNode.one()))


def neg_tree() -> TreeNode:
    # Uses extended-real behavior: log(0) = -inf, exp(-inf) = 0.
    ln_zero = TreeNode.eml(
        TreeNode.one(), TreeNode.eml(TreeNode.eml(TreeNode.one(), _zero_tree()), TreeNode.one())
    )
    return TreeNode.eml(ln_zero, exp_tree())


def reciprocal_tree() -> TreeNode:
    # Kept as a paper-explorer fixture only. It relies on the negation fixture above.
    return TreeNode.eml(neg_ln_tree(), TreeNode.one())


def neg_ln_tree() -> TreeNode:
    ln_x = ln_tree()
    zero = _zero_tree()
    ln_zero = TreeNode.eml(
        TreeNode.one(), TreeNode.eml(TreeNode.eml(TreeNode.one(), zero), TreeNode.one())
    )
    exp_lnx = TreeNode.eml(ln_x, TreeNode.one())
    return TreeNode.eml(ln_zero, exp_lnx)


TARGETS: dict[str, TargetSpec] = {
    "exp": TargetSpec(
        name="exp",
        display_name="exp(x)",
        variables=("x",),
        train_domain={"x": (-1.0, 1.0)},
        interpolation_domain={"x": (-0.9, 0.9)},
        extrapolation_domain={"x": (-1.5, 1.5)},
        default_depth=1,
        function=_exp,
        known_tree=exp_tree(),
        known_route=[("x", "1")],
        notes="Depth-1 paper identity: eml(x, 1).",
    ),
    "ln": TargetSpec(
        name="ln",
        display_name="ln(x)",
        variables=("x",),
        train_domain={"x": (0.35, 2.0)},
        interpolation_domain={"x": (0.45, 1.9)},
        extrapolation_domain={"x": (0.2, 2.5)},
        default_depth=3,
        function=_ln,
        known_tree=ln_tree(),
        known_route=[("1", "x"), ("cell0", "1"), ("1", "cell1")],
        notes="Paper identity: eml(1, eml(eml(1, x), 1)).",
    ),
    "identity": TargetSpec(
        name="identity",
        display_name="x",
        variables=("x",),
        train_domain={"x": (0.35, 2.0)},
        interpolation_domain={"x": (0.45, 1.9)},
        extrapolation_domain={"x": (0.2, 2.5)},
        default_depth=4,
        function=_identity,
        known_tree=identity_tree(),
        known_route=[("1", "x"), ("cell0", "1"), ("1", "cell1"), ("cell2", "1")],
        notes="Non-trivial identity route: exp(ln(x)) expressed in EML form.",
    ),
}


PAPER_FIXTURES: dict[str, TreeNode] = {
    "1": one_tree(),
    "e": e_tree(),
    "exp(x)": exp_tree(),
    "ln(x)": ln_tree(),
    "x": identity_tree(),
    "x-1": x_minus_one_tree(),
    "-x": neg_tree(),
    "1/x": reciprocal_tree(),
}


def list_targets() -> list[str]:
    return sorted(TARGETS)


def get_target(name: str) -> TargetSpec:
    try:
        return TARGETS[name]
    except KeyError as exc:
        valid = ", ".join(list_targets())
        raise KeyError(f"Unknown target {name!r}. Valid targets: {valid}") from exc


def sample_inputs(
    spec: TargetSpec,
    domain: dict[str, tuple[float, float]] | None = None,
    *,
    points: int = 128,
) -> dict[str, torch.Tensor]:
    selected = domain or spec.train_domain
    values: dict[str, torch.Tensor] = {}
    for variable in spec.variables:
        low, high = selected[variable]
        values[variable] = torch.linspace(low, high, points, dtype=torch.float64).to(COMPLEX_DTYPE)
    return values
