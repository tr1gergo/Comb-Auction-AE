"""
Instance generation and demand logic for GRID Auctions.

This module creates grid-based instances, provides efficient demand computation,
and exposes a budget-respecting IP fallback. It mirrors the style of the ACE
heuristic code while preferring Gurobi when available.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import pulp
except ImportError as exc:  # pragma: no cover - PuLP is required at runtime
    raise ImportError(
        "PuLP is required to run the GRID auction experiments. "
        "Install it with `pip install pulp`."
    ) from exc


# ---------------------------------------------------------------------------
# Utilities


def _seed_all(seed: Optional[int]) -> np.random.Generator:
    """Seed Python's RNG and return a dedicated NumPy generator."""
    if seed is not None:
        random.seed(seed)
    return np.random.default_rng(seed)


def _prepare_solver(requested: Optional[str] = None) -> pulp.LpSolver:
    """
    Instantiate a PuLP solver, preferring GUROBI when available.

    When ``requested`` is None or ``"GUROBI"``, this attempts to create a Gurobi
    solver first and falls back to CBC if Gurobi is unavailable. Other solver
    names are passed through to PuLP if exposed, otherwise a ValueError is
    raised.
    """
    name = "GUROBI" if requested is None else requested.upper()

    if name == "GUROBI":
        gurobi_cls = getattr(pulp, "GUROBI", None)
        if gurobi_cls is not None:
            try:
                return gurobi_cls(msg=False)
            except Exception:
                # Fall back to CBC if Gurobi is not licensed/available.
                pass
        return pulp.PULP_CBC_CMD(msg=False)

    if name == "PULP_CBC_CMD":
        return pulp.PULP_CBC_CMD(msg=False)

    solver_cls = getattr(pulp, name, None)
    if solver_cls is None:
        raise ValueError(f"Unsupported solver '{requested}'.")
    return solver_cls(msg=False)  # type: ignore[call-arg]


def _direction_offsets() -> Dict[str, Tuple[int, int]]:
    """Return the grid step offsets for the eight compass directions."""
    return {
        "N": (-1, 0),
        "S": (1, 0),
        "E": (0, 1),
        "W": (0, -1),
        "NE": (-1, 1),
        "NW": (-1, -1),
        "SE": (1, 1),
        "SW": (1, -1),
    }


# ---------------------------------------------------------------------------
# Instance generation


def generate_grid_instance(
    n: int,
    m: int,
    C: int,
    max_cap: int,
    max_budget: int,
    max_utility: int,
    rng_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate a GRID auction instance on an n x n grid with directional paths.

    Parameters
    ----------
    n:
        Grid dimension (n x n).
    m:
        Number of players.
    C:
        Maximum path length in each direction.
    max_cap:
        Maximum capacity for any item.
    max_budget:
        Upper bound for budgets.
    max_utility:
        Upper bound for per-item utility.
    rng_seed:
        Optional seed for reproducibility.

    Returns
    -------
    dict
        Contains item list, capacities, budgets, and per-player directional paths
        with utilities.
    """
    if n < 1:
        raise ValueError("Grid size n must be positive.")
    if m < 0:
        raise ValueError("Number of players m must be non-negative.")
    if C < 0:
        raise ValueError("Path length C must be non-negative.")

    rng = _seed_all(rng_seed)
    np.random.default_rng(rng_seed)

    # Sample player bases uniformly with replacement.
    bases = [(int(r), int(c)) for r, c in rng.integers(0, n, size=(m, 2))]

    base_counts = np.zeros((n, n), dtype=int)
    for r, c in bases:
        base_counts[r, c] += 1

    # Items are the grid cells with zero players.
    items: List[Tuple[int, int]] = []
    item_index: Dict[Tuple[int, int], int] = {}
    for r in range(n):
        for c in range(n):
            if base_counts[r, c] == 0:
                idx = len(items)
                items.append((r, c))
                item_index[(r, c)] = idx

    num_items = len(items)
    capacities = rng.integers(1, max_cap + 1, size=num_items, dtype=int).astype(float)

    # Bell-shaped budgets via mean of two uniforms, clipped to [1, max_budget].
    if max_budget < 1:
        raise ValueError("max_budget must be at least 1.")
    if m > 0:
        budget_draws = rng.integers(1, max_budget + 1, size=(2, m))
        budgets = np.rint(budget_draws.mean(axis=0)).astype(float)
        budgets = np.clip(budgets, 1, max_budget)
    else:
        budgets = np.array([], dtype=float)

    directions = _direction_offsets()
    players: List[Dict[str, Any]] = []

    for base_idx, (r0, c0) in enumerate(bases):
        path_map: Dict[str, List[Tuple[int, float]]] = {}

        for dir_name, (dr, dc) in directions.items():
            path: List[Tuple[int, float]] = []
            for step in range(1, C + 1):
                r = r0 + dr * step
                c = c0 + dc * step
                if r < 0 or r >= n or c < 0 or c >= n:
                    break  # Hit boundary.
                if base_counts[r, c] > 0:
                    break  # Hit another base; path stops before including it.

                idx = item_index.get((r, c))
                if idx is None:
                    break  # Cell is not an item (occupied by a player).

                utility = float(rng.integers(1, max_utility + 1))
                path.append((idx, utility))
            path_map[dir_name] = path

        players.append({"base": (r0, c0), "paths": path_map, "id": base_idx})

    max_budget_value = float(np.max(budgets)) if budgets.size else 0.0

    return {
        "grid_size": n,
        "num_players": m,
        "max_path_length": C,
        "items": items,
        "num_items": num_items,
        "item_index": item_index,
        "capacities": capacities,
        "budgets": budgets,
        "max_budget": max_budget_value,
        "players": players,
    }


# ---------------------------------------------------------------------------
# Demand logic


def compute_demand_efficient(
    player: Dict[str, Any],
    prices: np.ndarray,
    budget: float,
) -> Optional[Tuple[Set[int], float]]:
    """
    Fast greedy demand computation ignoring budget interactions across directions.

    For each direction, select the length that maximises surplus (utility - price).
    If the aggregated cost across all directions fits within the budget, return
    the resulting bundle and its total utility; otherwise return None to signal
    that the budget binds and IP should be used.
    """
    chosen_items: List[int] = []
    total_cost = 0.0
    total_utility = 0.0

    for direction, path in player["paths"].items():
        best_surplus = 0.0
        best_len = 0
        best_cost = 0.0
        best_util = 0.0

        prefix_cost = 0.0
        prefix_util = 0.0
        for idx, (item_idx, utility) in enumerate(path, start=1):
            prefix_cost += float(prices[item_idx])
            prefix_util += float(utility)
            surplus = prefix_util - prefix_cost
            if surplus > best_surplus + 1e-12:
                best_surplus = surplus
                best_len = idx
                best_cost = prefix_cost
                best_util = prefix_util

        if best_len > 0:
            chosen_items.extend([item for item, _ in path[:best_len]])
            total_cost += best_cost
            total_utility += best_util

    if total_cost <= budget + 1e-9:
        return set(chosen_items), total_utility
    return None


def compute_demand_IP(
    player: Dict[str, Any],
    prices: np.ndarray,
    budget: float,
    nudges: Optional[np.ndarray] = None,
    soft_budget: float = 0.0,
    solver_name: Optional[str] = "GUROBI",
) -> Tuple[Set[int], float]:
    """
    Solve the multiple-choice knapsack per player using IP.

    Parameters
    ----------
    nudges:
        Optional per-item adjustments added to utilities in the objective.
    soft_budget:
        Extra budget allowance used during relaxed steps.
    solver_name:
        Solver preference; defaults to GUROBI with CBC fallback.

    Returns
    -------
    bundle, utility
        The chosen item indices and their true (unnudged) utility.
    """
    nudges_vec = nudges if nudges is not None else np.zeros_like(prices, dtype=float)
    problem = pulp.LpProblem("PlayerDemandMCKP", pulp.LpMaximize)
    solver = _prepare_solver(solver_name)

    x_vars: Dict[Tuple[str, int], pulp.LpVariable] = {}
    option_data: Dict[Tuple[str, int], Tuple[List[int], float, float]] = {}

    for direction, path in player["paths"].items():
        # Option length 0: pick nothing in this direction.
        option_data[(direction, 0)] = ([], 0.0, 0.0)
        x_vars[(direction, 0)] = pulp.LpVariable(f"x_{direction}_0", cat="Binary")

        prefix_cost = 0.0
        prefix_util = 0.0
        for length, (item_idx, utility) in enumerate(path, start=1):
            prefix_cost += float(prices[item_idx])
            prefix_util += float(utility)

            option_key = (direction, length)
            option_data[option_key] = (
                [item for item, _ in path[:length]],
                prefix_cost,
                prefix_util,
            )
            x_vars[option_key] = pulp.LpVariable(f"x_{direction}_{length}", cat="Binary")

        # One length must be selected per direction.
        relevant_vars = [var for (d, _), var in x_vars.items() if d == direction]
        problem += pulp.lpSum(relevant_vars) == 1, f"choose_one_{direction}"

    # Budget constraint.
    total_cost = pulp.lpSum(cost * x_vars[key] for key, (_, cost, _) in option_data.items())
    problem += total_cost <= budget + soft_budget, "budget"

    # Objective with optional nudges.
    def _nudged_surplus(key: Tuple[str, int]) -> float:
        items, cost, base_util = option_data[key]
        nudge = float(np.sum(nudges_vec[items])) if items else 0.0
        return (base_util - cost) + nudge

    problem += pulp.lpSum(_nudged_surplus(key) * x_vars[key] for key in option_data.keys())

    status = problem.solve(solver)
    status_str = pulp.LpStatus[status]

    selected: Set[int] = set()
    total_utility = 0.0
    if status_str in ("Optimal", "Integer Feasible"):
        for key, var in x_vars.items():
            if var.varValue and var.varValue > 0.5:
                items, cost, base_util = option_data[key]
                selected.update(items)
                total_utility += base_util - cost

    return selected, total_utility


# ---------------------------------------------------------------------------
# Equilibrium check


def _count_demands(demands: Sequence[Set[int]], num_items: int) -> np.ndarray:
    """Count how many players demand each item."""
    counts = np.zeros(num_items, dtype=float)
    for bundle in demands:
        for item in bundle:
            counts[item] += 1
    return counts


def check_equilibrium(
    demands: Sequence[Set[int]],
    capacities: np.ndarray,
    prices: np.ndarray,
    slack: float,
) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
    """
    Evaluate the GRID equilibrium conditions with capacity slack.

    Conditions:
        1) demand <= capacity + slack
        2) If price > 0 then demand >= capacity - slack
    """
    num_items = len(capacities)
    demand_counts = _count_demands(demands, num_items)

    over_mask = demand_counts > capacities + slack
    under_mask = (prices > 0) & (demand_counts < capacities - slack)

    over = {int(idx) for idx in np.where(over_mask)[0]}
    under = {int(idx) for idx in np.where(under_mask)[0]}

    extras = {
        "demand_counts": demand_counts,
        "over_demanded": over,
        "under_demanded": under,
    }

    return not over and not under, demand_counts, extras


__all__ = [
    "generate_grid_instance",
    "compute_demand_efficient",
    "compute_demand_IP",
    "check_equilibrium",
]
