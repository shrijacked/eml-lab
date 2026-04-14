"""Discrete EML tree representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import sympy as sp
import torch

from eml_lab.operators import COMPLEX_DTYPE, eml_exact

TreeKind = Literal["const", "var", "eml"]


@dataclass(frozen=True)
class TreeNode:
    """Immutable binary EML expression tree."""

    kind: TreeKind
    value: str | None = None
    left: TreeNode | None = None
    right: TreeNode | None = None

    @staticmethod
    def one() -> TreeNode:
        return TreeNode("const", "1")

    @staticmethod
    def var(name: str) -> TreeNode:
        return TreeNode("var", name)

    @staticmethod
    def eml(left: TreeNode, right: TreeNode) -> TreeNode:
        return TreeNode("eml", None, left, right)

    def evaluate(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.kind == "const":
            return _ones_like_inputs(inputs)
        if self.kind == "var":
            if self.value not in inputs:
                raise KeyError(f"Missing variable {self.value!r}")
            return inputs[self.value].to(dtype=COMPLEX_DTYPE)
        if self.left is None or self.right is None:
            raise ValueError("EML node requires left and right children")
        return eml_exact(self.left.evaluate(inputs), self.right.evaluate(inputs))

    def leaf_count(self) -> int:
        if self.kind in {"const", "var"}:
            return 1
        assert self.left is not None and self.right is not None
        return self.left.leaf_count() + self.right.leaf_count()

    def node_count(self) -> int:
        if self.kind in {"const", "var"}:
            return 1
        assert self.left is not None and self.right is not None
        return 1 + self.left.node_count() + self.right.node_count()

    def to_sympy(self) -> sp.Expr:
        if self.kind == "const":
            return sp.Integer(1)
        if self.kind == "var":
            assert self.value is not None
            return sp.Symbol(self.value)
        assert self.left is not None and self.right is not None
        eml = sp.Function("eml")
        return eml(self.left.to_sympy(), self.right.to_sympy())


def _ones_like_inputs(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if inputs:
        first = next(iter(inputs.values()))
        return torch.ones_like(first, dtype=COMPLEX_DTYPE)
    return torch.ones((), dtype=COMPLEX_DTYPE)


def to_rpn(tree: TreeNode) -> list[str]:
    """Return RPN tokens using E for EML."""

    if tree.kind == "const":
        return ["1"]
    if tree.kind == "var":
        assert tree.value is not None
        return [tree.value]
    assert tree.left is not None and tree.right is not None
    return [*to_rpn(tree.left), *to_rpn(tree.right), "E"]


def rpn_string(tree: TreeNode) -> str:
    return " ".join(to_rpn(tree))


def from_rpn(tokens: list[str] | str) -> TreeNode:
    """Parse RPN tokens using E for EML."""

    if isinstance(tokens, str):
        parts = tokens.split()
        if len(parts) == 1 and parts[0] and "E" in parts[0]:
            parts = list(parts[0])
    else:
        parts = tokens

    stack: list[TreeNode] = []
    for token in parts:
        if token == "E":
            if len(stack) < 2:
                raise ValueError("RPN E token requires two operands")
            right = stack.pop()
            left = stack.pop()
            stack.append(TreeNode.eml(left, right))
        elif token == "1":
            stack.append(TreeNode.one())
        else:
            stack.append(TreeNode.var(token))
    if len(stack) != 1:
        raise ValueError(f"RPN expression left {len(stack)} values on the stack")
    return stack[0]


def to_networkx(tree: TreeNode) -> nx.DiGraph:
    """Convert a tree to a directed graph with labels suitable for visualization."""

    graph = nx.DiGraph()

    def visit(node: TreeNode, prefix: str) -> None:
        label = "eml" if node.kind == "eml" else str(node.value)
        graph.add_node(prefix, label=label)
        if node.left is not None:
            left_id = f"{prefix}L"
            graph.add_edge(prefix, left_id, side="left")
            visit(node.left, left_id)
        if node.right is not None:
            right_id = f"{prefix}R"
            graph.add_edge(prefix, right_id, side="right")
            visit(node.right, right_id)

    visit(tree, "root")
    return graph


def tree_to_json(tree: TreeNode) -> dict[str, object]:
    if tree.kind in {"const", "var"}:
        return {"kind": tree.kind, "value": tree.value}
    assert tree.left is not None and tree.right is not None
    return {
        "kind": "eml",
        "left": tree_to_json(tree.left),
        "right": tree_to_json(tree.right),
    }
