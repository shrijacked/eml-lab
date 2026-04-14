"""Visualization helpers shared by CLI and Streamlit."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from eml_lab.trees import TreeNode, to_networkx


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


def _has_pygraphviz() -> bool:
    try:
        import pygraphviz  # noqa: F401
    except ImportError:
        return False
    return True
