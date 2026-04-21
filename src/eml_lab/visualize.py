"""Visualization helpers shared by CLI and Streamlit."""

from __future__ import annotations

import matplotlib
import networkx as nx
import numpy as np

from eml_lab.trees import TreeNode, to_networkx

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def tree_figure(tree: TreeNode):
    graph = to_networkx(tree)
    positions = (
        nx.nx_agraph.graphviz_layout(graph, prog="dot")
        if _has_pygraphviz()
        else nx.spring_layout(graph, seed=0)
    )
    labels = nx.get_node_attributes(graph, "label")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    nx.draw(
        graph,
        positions,
        labels=labels,
        with_labels=True,
        node_color="#f7f7f7",
        edge_color="#777777",
        node_size=1200,
        font_size=10,
        ax=ax,
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def logits_heatmap_figure(logits_table: tuple[dict[str, float | int | str], ...]):
    if not logits_table:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
        for ax, title in zip(axes, ["Left logits", "Right logits"], strict=True):
            ax.set_title(title)
            ax.text(0.5, 0.5, "No logits", ha="center", va="center")
            ax.set_axis_off()
        fig.tight_layout()
        return fig

    labels: list[str] = []
    cells: list[int] = []
    rows_by_key: dict[tuple[int, str], dict[str, float | int | str]] = {}
    for row in logits_table:
        cell = int(row["cell"])
        choice = str(row["choice"])
        rows_by_key[(cell, choice)] = row
        if cell not in cells:
            cells.append(cell)
        if choice not in labels:
            labels.append(choice)

    left = np.full((len(cells), len(labels)), np.nan)
    right = np.full((len(cells), len(labels)), np.nan)
    for cell_index, cell in enumerate(cells):
        for label_index, label in enumerate(labels):
            row = rows_by_key.get((cell, label))
            if row is None:
                continue
            left[cell_index, label_index] = float(row["left_probability"])
            right[cell_index, label_index] = float(row["right_probability"])

    fig, axes = plt.subplots(1, 2, figsize=(10, max(3.5, len(cells) * 1.2)), sharey=True)
    matrices = [np.ma.masked_invalid(left), np.ma.masked_invalid(right)]
    titles = ["Left logits", "Right logits"]
    y_labels = [f"Cell {cell}" for cell in cells]

    for ax, matrix, title in zip(axes, matrices, titles, strict=True):
        ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(cells)), labels=y_labels)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                if np.ma.is_masked(value):
                    continue
                text_color = "white" if float(value) >= 0.55 else "black"
                ax.text(
                    col_index,
                    row_index,
                    f"{float(value):.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
        ax.set_xlabel("Choice")
    axes[0].set_ylabel("Tree cell")
    fig.tight_layout()
    return fig


def _has_pygraphviz() -> bool:
    try:
        import pygraphviz  # noqa: F401
    except ImportError:
        return False
    return True
