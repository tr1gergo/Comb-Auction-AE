"""
Visualization helpers for GRID Auctions.

Provides a simple routine to render grid instances and color items by the
players that obtained them. Cells with multiple winners are subdivided into
equal slices.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def _default_colors(num_players: int) -> List[str]:
    """Generate distinct colors for players."""
    cmap = plt.cm.get_cmap("tab20", max(1, num_players))
    return [cmap(i) for i in range(num_players)]


def draw_grid_solution(
    instance: Dict[str, Any],
    result: Dict[str, Any],
    player_colors: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw the grid, bases, and final allocation with per-player colors.

    Parameters
    ----------
    instance:
        GRID instance dictionary from ``generate_grid_instance``.
    result:
        Output from ``run_grid_heuristic``; expects ``demands`` field.
    player_colors:
        Optional sequence of color specs per player.
    ax:
        Matplotlib Axes to draw on; if None, a new figure/axes is created.
    title:
        Optional plot title.
    """
    n = int(instance["grid_size"])
    items: List[Tuple[int, int]] = instance["items"]
    item_index: Dict[Tuple[int, int], int] = instance["item_index"]
    num_players = int(instance["num_players"])
    bases = [tuple(p["base"]) for p in instance["players"]]

    if player_colors is None:
        player_colors = _default_colors(num_players)

    demands: List[set] = [set(bundle) for bundle in result.get("demands", [])]

    created_axes = ax is None
    if created_axes:
        fig, ax = plt.subplots(figsize=(max(6, n / 2), max(6, n / 2)))
    else:
        fig = ax.figure

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Put (0,0) at top-left to match grid intuition.
    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    ax.grid(True, which="both", color="lightgray", linestyle="--", linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Draw all item cells with a light background.
    for (r, c) in items:
        rect = Rectangle((c, r), 1, 1, facecolor="#f7f7f7", edgecolor="none")
        ax.add_patch(rect)

    # Mark player bases.
    for player_idx, (r, c) in enumerate(bases):
        ax.plot(
            c + 0.5,
            r + 0.5,
            marker="X",
            markersize=10,
            markeredgecolor="black",
            markerfacecolor=player_colors[player_idx % len(player_colors)],
            alpha=0.9,
        )

    # Map items to the set of players that received them.
    item_to_players: Dict[int, List[int]] = {idx: [] for idx in range(len(items))}
    for p_idx, bundle in enumerate(demands):
        for item_idx in bundle:
            item_to_players[item_idx].append(p_idx)

    # Color each allocated item; split cell into slices if multiple players share it.
    for (r, c), item_idx in item_index.items():
        owners = item_to_players.get(item_idx, [])
        if not owners:
            continue
        k = len(owners)
        slice_width = 1.0 / k
        for pos, owner in enumerate(owners):
            rect = Rectangle(
                (c + pos * slice_width, r),
                slice_width,
                1,
                facecolor=player_colors[owner % len(player_colors)],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
            )
            ax.add_patch(rect)

    if title:
        ax.set_title(title)

    # Legend
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=player_colors[i % len(player_colors)])
        for i in range(num_players)
    ]
    labels = [f"Player {i}" for i in range(num_players)]
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    # Reserve room on the right so the legend stays fully outside the axes.
    if created_axes:
        fig.subplots_adjust(right=0.78)

    return fig, ax


__all__ = ["draw_grid_solution"]
