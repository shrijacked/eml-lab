"""Target functions and known shallow EML fixtures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from eml_lab.operators import COMPLEX_DTYPE
from eml_lab.trees import TreeNode

TargetFunction = Callable[[dict[str, torch.Tensor]], torch.Tensor]
Route = list[tuple[str, str]]
TargetTier = Literal["stable", "research"]


@dataclass(frozen=True)
class TargetSpec:
    name: str
    display_name: str
    variables: tuple[str, ...]
    train_domain: dict[str, tuple[float, float]]
    interpolation_domain: dict[str, tuple[float, float]]
    extrapolation_domain: dict[str, tuple[float, float]]
    default_depth: int
    expected_depth: int | None
    function: TargetFunction
    known_tree: TreeNode | None
    known_route: Route | None
    tier: TargetTier
    comparison_eligible: bool
    failure_modes: tuple[str, ...]
    notes: str


def _x(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return inputs["x"].to(dtype=COMPLEX_DTYPE)


def _exp(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.exp(_x(inputs))


def _ln(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.log(_x(inputs))


def _identity(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return _x(inputs)


def _y(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return inputs["y"].to(dtype=COMPLEX_DTYPE)


def _square(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    x = _x(inputs)
    return x * x


def _mul(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return _x(inputs) * _y(inputs)


def _div(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return _x(inputs) / _y(inputs)


def _sin(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sin(_x(inputs))


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
        expected_depth=1,
        function=_exp,
        known_tree=exp_tree(),
        known_route=[("x", "1")],
        tier="stable",
        comparison_eligible=True,
        failure_modes=(),
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
        expected_depth=3,
        function=_ln,
        known_tree=ln_tree(),
        known_route=[("1", "x"), ("cell0", "1"), ("1", "cell1")],
        tier="stable",
        comparison_eligible=True,
        failure_modes=(),
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
        expected_depth=4,
        function=_identity,
        known_tree=identity_tree(),
        known_route=[("1", "x"), ("cell0", "1"), ("1", "cell1"), ("cell2", "1")],
        tier="stable",
        comparison_eligible=True,
        failure_modes=(),
        notes="Non-trivial identity route: exp(ln(x)) expressed in EML form.",
    ),
    "square": TargetSpec(
        name="square",
        display_name="x^2",
        variables=("x",),
        train_domain={"x": (-1.5, 1.5)},
        interpolation_domain={"x": (-1.2, 1.2)},
        extrapolation_domain={"x": (-2.0, 2.0)},
        default_depth=4,
        expected_depth=4,
        function=_square,
        known_tree=None,
        known_route=None,
        tier="research",
        comparison_eligible=False,
        failure_modes=(
            "may collapse to exp/log surrogates instead of quadratic growth",
            "gradient search can settle on locally smooth but globally wrong trees",
        ),
        notes="Research target. Polynomial recovery is not a shipped claim in v1 or Phase 2.",
    ),
    "mul": TargetSpec(
        name="mul",
        display_name="x*y",
        variables=("x", "y"),
        train_domain={"x": (-1.25, 1.25), "y": (0.4, 1.8)},
        interpolation_domain={"x": (-1.0, 1.0), "y": (0.5, 1.6)},
        extrapolation_domain={"x": (-1.5, 1.5), "y": (0.3, 2.0)},
        default_depth=4,
        expected_depth=4,
        function=_mul,
        known_tree=None,
        known_route=None,
        tier="research",
        comparison_eligible=False,
        failure_modes=(
            "paired-sample training may underconstrain multivariate behavior",
            "exact verifier can expose interpolation fits that fail on extrapolation",
        ),
        notes="Research target. Multivariate multiplication is tracked as an explicit experiment.",
    ),
    "div": TargetSpec(
        name="div",
        display_name="x/y",
        variables=("x", "y"),
        train_domain={"x": (-1.0, 1.0), "y": (0.5, 2.0)},
        interpolation_domain={"x": (-0.8, 0.8), "y": (0.6, 1.8)},
        extrapolation_domain={"x": (-1.25, 1.25), "y": (0.35, 2.2)},
        default_depth=5,
        expected_depth=5,
        function=_div,
        known_tree=None,
        known_route=None,
        tier="research",
        comparison_eligible=False,
        failure_modes=(
            "division is numerically brittle near small denominators",
            "candidate trees can look good on train points but break under extrapolation",
        ),
        notes="Research target. Domains stay away from zero to avoid trivial singularities.",
    ),
    "sin": TargetSpec(
        name="sin",
        display_name="sin(x)",
        variables=("x",),
        train_domain={"x": (-1.5, 1.5)},
        interpolation_domain={"x": (-1.25, 1.25)},
        extrapolation_domain={"x": (-3.0, 3.0)},
        default_depth=5,
        expected_depth=5,
        function=_sin,
        known_tree=None,
        known_route=None,
        tier="research",
        comparison_eligible=False,
        failure_modes=(
            "periodicity is hard to express with shallow EML route search",
            "models may overfit the local Taylor-like region and fail outside it",
        ),
        notes="Research target. Trigonometric recovery remains exploratory, not promised.",
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


def list_targets(
    tier: TargetTier | None = None, *, comparison_eligible: bool | None = None
) -> list[str]:
    names = sorted(TARGETS)
    if tier is not None:
        names = [name for name in names if TARGETS[name].tier == tier]
    if comparison_eligible is not None:
        names = [
            name for name in names if TARGETS[name].comparison_eligible == comparison_eligible
        ]
    return names


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
