"""Differentiable soft-routed EML trees."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from eml_lab.operators import COMPLEX_DTYPE, StabilityConfig, eml_train
from eml_lab.trees import TreeNode


@dataclass(frozen=True)
class SnapStep:
    cell_index: int
    left_choice: str
    right_choice: str


def _mixture(values: list[torch.Tensor], logits: torch.Tensor, temperature: float) -> torch.Tensor:
    weights = torch.softmax(logits / temperature, dim=0).to(dtype=COMPLEX_DTYPE)
    stacked = torch.stack(values, dim=0)
    shape = (weights.shape[0],) + (1,) * (stacked.ndim - 1)
    return torch.sum(weights.reshape(shape) * stacked, dim=0)


class SoftEMLTree(nn.Module):
    """A sequential soft EML tree.

    Cell i can consume constants, variables, or outputs of cells 0..i-1. Snapping
    expands those references into a plain binary EML tree.
    """

    def __init__(self, depth: int, variables: tuple[str, ...] = ("x",)) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.variables = tuple(variables)
        self.left_logits = nn.ParameterList()
        self.right_logits = nn.ParameterList()
        for cell_index in range(depth):
            option_count = 1 + len(self.variables) + cell_index
            self.left_logits.append(nn.Parameter(torch.zeros(option_count, dtype=torch.float64)))
            self.right_logits.append(nn.Parameter(torch.zeros(option_count, dtype=torch.float64)))

    def option_labels(self, cell_index: int) -> list[str]:
        return ["1", *self.variables, *[f"cell{i}" for i in range(cell_index)]]

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        temperature: float = 1.0,
        stability_config: StabilityConfig | None = None,
    ) -> torch.Tensor:
        available_values: list[torch.Tensor] = [self._one_like(inputs)]
        available_values.extend(inputs[name].to(dtype=COMPLEX_DTYPE) for name in self.variables)
        outputs: list[torch.Tensor] = []
        for cell_index in range(self.depth):
            left = _mixture(available_values, self.left_logits[cell_index], temperature)
            right = _mixture(available_values, self.right_logits[cell_index], temperature)
            output = eml_train(left, right, stability_config)
            assert isinstance(output, torch.Tensor)
            outputs.append(output)
            available_values.append(output)
        return outputs[-1]

    def snap(self) -> tuple[TreeNode, list[SnapStep]]:
        available_trees: list[TreeNode] = [
            TreeNode.one(),
            *[TreeNode.var(name) for name in self.variables],
        ]
        steps: list[SnapStep] = []
        for cell_index in range(self.depth):
            labels = self.option_labels(cell_index)
            left_idx = int(torch.argmax(self.left_logits[cell_index]).item())
            right_idx = int(torch.argmax(self.right_logits[cell_index]).item())
            left_tree = available_trees[left_idx]
            right_tree = available_trees[right_idx]
            node = TreeNode.eml(left_tree, right_tree)
            available_trees.append(node)
            steps.append(SnapStep(cell_index, labels[left_idx], labels[right_idx]))
        return available_trees[-1], steps

    def seed_route(
        self,
        route: list[tuple[str, str]],
        *,
        margin: float = 8.0,
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if len(route) > self.depth:
            raise ValueError("route is deeper than this model")
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        with torch.no_grad():
            for cell_index, (left_choice, right_choice) in enumerate(route):
                labels = self.option_labels(cell_index)
                self.left_logits[cell_index].fill_(-margin)
                self.right_logits[cell_index].fill_(-margin)
                self.left_logits[cell_index][labels.index(left_choice)] = margin
                self.right_logits[cell_index][labels.index(right_choice)] = margin
                if noise_std > 0:
                    self.left_logits[cell_index].add_(
                        torch.randn(self.left_logits[cell_index].shape, generator=generator)
                        * noise_std
                    )
                    self.right_logits[cell_index].add_(
                        torch.randn(self.right_logits[cell_index].shape, generator=generator)
                        * noise_std
                    )

    def logits_table(self) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for cell_index in range(self.depth):
            labels = self.option_labels(cell_index)
            left_weights = torch.softmax(self.left_logits[cell_index], dim=0)
            right_weights = torch.softmax(self.right_logits[cell_index], dim=0)
            for label, left, right in zip(labels, left_weights, right_weights, strict=True):
                rows.append(
                    {
                        "cell": cell_index,
                        "choice": label,
                        "left_probability": float(left.item()),
                        "right_probability": float(right.item()),
                    }
                )
        return rows

    def _one_like(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if inputs:
            first = next(iter(inputs.values()))
            return torch.ones_like(first, dtype=COMPLEX_DTYPE)
        return torch.ones((), dtype=COMPLEX_DTYPE)


def snap_tree(model: SoftEMLTree) -> TreeNode:
    tree, _ = model.snap()
    return tree
