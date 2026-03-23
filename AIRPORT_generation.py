"""
Instance generation and demand logic for Airport Time-Slot Auctions.

This module mirrors the GRID architecture with domain-specific primitives:
players are airlines, items are time-slots, and each airline bids on a small
set of feasible arrival/departure slot pairs.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Utilities


def _seed_all(seed: Optional[int]) -> np.random.Generator:
    """Seed Python's RNG and return a dedicated NumPy generator."""
    if seed is not None:
        random.seed(seed)
    return np.random.default_rng(seed)


def _derive_num_players(num_items: int, max_cap: int) -> int:
    """Infer the player count from the requested market tightness relation."""
    numerator = num_items * (max_cap - 1)
    if numerator % 2 != 0:
        raise ValueError("num_items * (max_cap - 1) must be even to derive num_players.")
    return numerator // 2


def _sample_wait_time(num_items: int, rng: np.random.Generator) -> int:
    """Sample an even waiting time from [num_items/4, 3*num_items/4]."""
    low = int(np.ceil(num_items / 4))
    high = int(np.floor(3 * num_items / 4))
    if low % 2 == 1:
        low += 1
    if high % 2 == 1:
        high -= 1
    if low > high:
        raise ValueError("No feasible even wait time for the chosen num_items.")
    candidates = np.arange(low, high + 1, 2, dtype=int)
    return int(rng.choice(candidates))


def _build_shift_utility(
    peak_utility: float,
    shift: int,
    max_shift: int,
    max_budget: int,
    rng: np.random.Generator,
) -> float:
    """
    Build a bell-shaped utility profile around shift 0.

    The optimal shift (0) gets the highest utility. Utility decays smoothly with
    |shift| and remains in [0, 2 * max_budget].
    """
    # Bell-shaped baseline: nearby shifts stay close to peak, distant shifts
    # decay smoothly but remain at least 1 before adding noise.
    sigma = max(1.0, 0.9 * max_shift)
    bell = np.exp(-(float(shift) ** 2) / (2.0 * sigma**2))
    baseline = 1.0 + (peak_utility - 1.0) * bell

    # Add mild random noise, then clip to the allowed range.
    noise_std = max(0.5, 0.04 * peak_utility)
    noise = float(rng.normal(loc=0.0, scale=noise_std))
    utility = float(np.rint(baseline + noise))
    utility = max(0.0, min(utility, 2.0 * max_budget))
    return utility


# ---------------------------------------------------------------------------
# Instance generation


def generate_airport_instance(
    num_items: int,
    max_cap: int,
    max_budget: int,
    num_players: Optional[int] = None,
    rng_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate an airport slot-auction instance.

    Parameters
    ----------
    num_items:
        Number of time-slots.
    max_cap:
        Uniform capacity assigned to every slot.
    max_budget:
        Upper bound for bell-shaped budget generation.
    num_players:
        Optional override. If None, uses num_players * 2 = num_items * (max_cap - 1).
    rng_seed:
        Optional seed for reproducibility.

    Returns
    -------
    dict
        Contains capacities, budgets, and per-airline feasible pair bids.
    """
    if num_items < 2:
        raise ValueError("num_items must be at least 2.")
    if max_cap < 1:
        raise ValueError("max_cap must be positive.")
    if max_budget < 1:
        raise ValueError("max_budget must be at least 1.")

    rng = _seed_all(rng_seed)
    np.random.default_rng(rng_seed)

    derived_players = _derive_num_players(num_items, max_cap)
    num_players_value = derived_players if num_players is None else int(num_players)
    if num_players_value < 1:
        raise ValueError("num_players must be positive.")

    capacities = np.full(num_items, float(max_cap), dtype=float)

    # Bell-shaped budgets via the mean of two uniform draws, clipped to [1, max_budget].
    budget_draws = rng.integers(1, max_budget + 1, size=(2, num_players_value))
    budgets = np.rint(budget_draws.mean(axis=0)).astype(float)
    budgets = np.clip(budgets, 1, max_budget)

    players: List[Dict[str, Any]] = []
    for player_idx in range(num_players_value):
        wait_i = _sample_wait_time(num_items, rng)

        # Choose the optimal arrival so that departure = arrival + wait_i is in range.
        max_start = num_items - wait_i - 1
        if max_start < 0:
            max_start = 0
        arr_i = int(rng.integers(0, max_start + 1))
        dep_i = arr_i + wait_i

        max_shift = max(0, wait_i // 2 - 1)
        peak_utility = float(rng.integers(max(1, max_budget), 2 * max_budget + 1))

        feasible_pairs: List[Dict[str, Any]] = []
        for shift in range(-max_shift, max_shift + 1):
            g1 = arr_i + shift
            g2 = dep_i + shift
            if g1 < 0 or g2 < 0 or g1 >= num_items or g2 >= num_items:
                continue

            utility = _build_shift_utility(peak_utility, shift, max_shift, max_budget, rng)
            feasible_pairs.append(
                {
                    "pair": (int(g1), int(g2)),
                    "shift": int(shift),
                    "utility": float(utility),
                }
            )

        players.append(
            {
                "id": player_idx,
                "wait": int(wait_i),
                "arrival": int(arr_i),
                "departure": int(dep_i),
                "feasible_pairs": feasible_pairs,
            }
        )

    max_budget_value = float(np.max(budgets)) if budgets.size else 0.0
    item_labels = [f"slot_{idx}" for idx in range(num_items)]

    return {
        "num_items": int(num_items),
        "num_players": int(num_players_value),
        "derived_num_players": int(derived_players),
        "item_labels": item_labels,
        "capacities": capacities,
        "budgets": budgets,
        "max_budget": max_budget_value,
        "max_cap": int(max_cap),
        "players": players,
    }


# ---------------------------------------------------------------------------
# Demand logic


def compute_demand_bruteforce(
    player: Dict[str, Any],
    prices: np.ndarray,
    budget: float,
) -> Tuple[Set[int], float]:
    """
    Compute a bidder's demand by exhaustive search over feasible slot pairs.

    Returns
    -------
    bundle, surplus
        The chosen bundle (empty or one pair) and resulting utility minus price.
    """
    best_surplus = 0.0
    best_bundle: Set[int] = set()

    for entry in player["feasible_pairs"]:
        g1, g2 = entry["pair"]
        utility = float(entry["utility"])
        cost = float(prices[g1] + prices[g2])
        if cost > budget + 1e-9:
            continue

        surplus = utility - cost
        if surplus > best_surplus + 1e-12:
            best_surplus = surplus
            best_bundle = {int(g1), int(g2)}

    return best_bundle, best_surplus


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
    Evaluate the airport approximate-equilibrium conditions with slack.

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
    "generate_airport_instance",
    "compute_demand_bruteforce",
    "check_equilibrium",
]
