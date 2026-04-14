"""Deterministic route mutations for the local EML orchestrator."""

from __future__ import annotations

import random
from dataclasses import dataclass

from eml_lab.targets import Route
from eml_lab.trees import TreeNode

RouteTuple = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RouteMutation:
    name: str
    route: RouteTuple
    parent_route: RouteTuple | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "route": [list(step) for step in self.route],
            "parent_route": None
            if self.parent_route is None
            else [list(step) for step in self.parent_route],
        }


def as_route_tuple(route: Route | RouteTuple) -> RouteTuple:
    return tuple((left, right) for left, right in route)


def option_labels(cell_index: int, variables: tuple[str, ...]) -> list[str]:
    return ["1", *variables, *[f"cell{i}" for i in range(cell_index)]]


def route_to_tree(route: Route | RouteTuple, variables: tuple[str, ...] = ("x",)) -> TreeNode:
    available_trees: list[TreeNode] = [TreeNode.one(), *[TreeNode.var(name) for name in variables]]
    for cell_index, (left_choice, right_choice) in enumerate(route):
        labels = option_labels(cell_index, variables)
        left_tree = available_trees[labels.index(left_choice)]
        right_tree = available_trees[labels.index(right_choice)]
        available_trees.append(TreeNode.eml(left_tree, right_tree))
    return available_trees[-1]


def enumerate_single_edit_mutations(
    route: Route | RouteTuple, variables: tuple[str, ...] = ("x",)
) -> list[RouteMutation]:
    base_route = as_route_tuple(route)
    mutations: list[RouteMutation] = []
    for cell_index, (left_choice, right_choice) in enumerate(base_route):
        labels = option_labels(cell_index, variables)
        for label in labels:
            if label != left_choice:
                updated = list(base_route)
                updated[cell_index] = (label, right_choice)
                mutations.append(
                    RouteMutation(
                        name=f"cell{cell_index}:left->{label}",
                        route=tuple(updated),
                        parent_route=base_route,
                    )
                )
            if label != right_choice:
                updated = list(base_route)
                updated[cell_index] = (left_choice, label)
                mutations.append(
                    RouteMutation(
                        name=f"cell{cell_index}:right->{label}",
                        route=tuple(updated),
                        parent_route=base_route,
                    )
                )
        if left_choice != right_choice:
            updated = list(base_route)
            updated[cell_index] = (right_choice, left_choice)
            mutations.append(
                RouteMutation(
                    name=f"cell{cell_index}:swap",
                    route=tuple(updated),
                    parent_route=base_route,
                )
            )
    return _dedupe_mutations(mutations)


def enumerate_depth_expansion_mutations(
    route: Route | RouteTuple,
    variables: tuple[str, ...] = ("x",),
    *,
    max_depth: int,
) -> list[RouteMutation]:
    base_route = as_route_tuple(route)
    if len(base_route) >= max_depth:
        return []
    new_cell_index = len(base_route)
    labels = option_labels(new_cell_index, variables)
    preferred_labels = [f"cell{new_cell_index - 1}"] if new_cell_index > 0 else []
    preferred_labels.extend(["1", *variables])
    candidates = [label for label in preferred_labels if label in labels]

    mutations: list[RouteMutation] = []
    for left_label in candidates:
        for right_label in candidates:
            expanded = (*base_route, (left_label, right_label))
            mutations.append(
                RouteMutation(
                    name=f"expand:{left_label},{right_label}",
                    route=expanded,
                    parent_route=base_route,
                )
            )
    return _dedupe_mutations(mutations)


def deterministic_seed_mutations(
    route: Route | RouteTuple,
    variables: tuple[str, ...] = ("x",),
    *,
    seed: int = 0,
    count: int = 4,
) -> list[RouteMutation]:
    mutations = enumerate_single_edit_mutations(route, variables)
    generator = random.Random(seed)
    generator.shuffle(mutations)
    return mutations[:count]


def _dedupe_mutations(mutations: list[RouteMutation]) -> list[RouteMutation]:
    seen: set[RouteTuple] = set()
    unique: list[RouteMutation] = []
    for mutation in mutations:
        if mutation.route in seen:
            continue
        seen.add(mutation.route)
        unique.append(mutation)
    return unique
