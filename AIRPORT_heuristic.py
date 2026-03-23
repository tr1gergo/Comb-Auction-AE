"""
Heuristic solver for Airport Time-Slot Auctions.

This module mirrors the GRID heuristic structure while replacing bidder demand
with an exact brute-force search over each airline's feasible slot pairs.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from AIRPORT_generation import (
    check_equilibrium,
    compute_demand_bruteforce,
)


def _seed_all(seed: Optional[int]) -> np.random.Generator:
    """Seed Python's RNG and return a dedicated NumPy generator."""
    if seed is not None:
        random.seed(seed)
    return np.random.default_rng(seed)


def run_airport_heuristic(
    instance: Dict[str, Any],
    max_rounds: int = 200,
    delta: float = 0.5,
    epsilon_scale: float = 0.1,
    price_floor: float = 0.0,
    rng_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute the airport heuristic with slack escalation and tatonnement updates.

    Returns a dictionary capturing the final prices, demands, slack used,
    whether boosting was applied, and diagnostic metadata.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if epsilon_scale < 0:
        raise ValueError("epsilon_scale must be non-negative.")
    if price_floor < 0:
        raise ValueError("price_floor must be non-negative.")

    rng = _seed_all(rng_seed)
    np.random.default_rng(rng_seed)

    players = instance["players"]
    num_players = int(instance["num_players"])
    num_items = int(instance["num_items"])
    capacities = np.array(instance["capacities"], dtype=float)
    budgets = np.array(instance["budgets"], dtype=float)
    max_budget_value = float(np.max(budgets)) if budgets.size else 0.0
    max_cap = int(instance["max_cap"])

    history: List[Dict[str, Any]] = []
    boost_round = 67

    for slack in range(0, 5):
        prices = np.zeros(num_items, dtype=float)
        current_delta = float(delta)
        is_boosted = False
        demands: List[Set[int]] = [set() for _ in range(num_players)]

        for round_index in range(1, max_rounds + 1):
            # ------------------------------ Step 1: Demand computation
            new_demands: List[Set[int]] = []
            prev_demands = demands
            for player_idx, player in enumerate(players):
                bundle, _ = compute_demand_bruteforce(player, prices, budgets[player_idx])
                new_demands.append(set(bundle))

            demands = new_demands
            demand_changes = sum(
                1 for idx in range(num_players) if demands[idx] != prev_demands[idx]
            )
            change_rate = demand_changes / max(1, num_players)

            equilibrium, demand_counts, extras = check_equilibrium(
                demands,
                capacities,
                prices,
                slack,
            )
            if equilibrium:
                return {
                    "prices": prices,
                    "demands": [set(bundle) for bundle in demands],
                    "slack": slack,
                    "is_boosted": is_boosted,
                    "equilibrium": True,
                    "rounds": round_index,
                    "delta": current_delta,
                    "meta": {"history": history},
                }

            # ------------------------------ Step 2: Nudged re-optimization
            epsilon = epsilon_scale / max(1, num_items)
            nudged_prices = prices.copy()
            for idx in extras["over_demanded"]:
                nudged_prices[idx] -= epsilon
            for idx in extras["under_demanded"]:
                nudged_prices[idx] += epsilon

            max_price = float(np.max(prices)) if prices.size else 0.0
            nudged_changed = False
            for player_idx, player in enumerate(players):
                budget_ratio = budgets[player_idx] / max_budget_value if max_budget_value > 0 else 1.0
                rank_factor = (player_idx + 1) / max(1, num_players)
                soft_extra = slack * max_price * budget_ratio * rank_factor
                relaxed_budget = budgets[player_idx] + soft_extra

                new_bundle, _ = compute_demand_bruteforce(player, nudged_prices, relaxed_budget)
                if new_bundle != demands[player_idx]:
                    demands[player_idx] = set(new_bundle)
                    nudged_changed = True

            if nudged_changed:
                equilibrium, demand_counts, extras = check_equilibrium(
                    demands,
                    capacities,
                    prices,
                    slack,
                )
                if equilibrium:
                    return {
                        "prices": prices,
                        "demands": [set(bundle) for bundle in demands],
                        "slack": slack,
                        "is_boosted": is_boosted,
                        "equilibrium": True,
                        "rounds": round_index,
                        "delta": current_delta,
                        "meta": {"history": history},
                    }

            # ------------------------------ Step 3: Tatonnement update
            over_mask = demand_counts > capacities + slack
            under_mask = (prices > price_floor) & (demand_counts < capacities - slack)

            prices = prices.copy()
            prices[over_mask] += current_delta
            prices[under_mask] = np.maximum(price_floor, prices[under_mask] - current_delta)

            # ------------------------------ Boosting to escape stagnation
            if round_index >= boost_round and not is_boosted and num_items > 0:
                current_max_price = float(np.max(prices)) if prices.size else 0.0
                noise = rng.uniform(
                    -current_max_price / 3 if current_max_price > 0 else 0.0,
                    current_max_price / 3 if current_max_price > 0 else 0.0,
                    size=num_items,
                )
                prices = np.maximum(price_floor, prices + noise)
                current_delta = current_delta / 2.0
                is_boosted = True

            max_price = float(np.max(prices)) if prices.size else 0.0
            price_std = float(np.std(prices)) if prices.size else 0.0
            history.append(
                {
                    "slack": slack,
                    "round": round_index,
                    "prices": prices.copy(),
                    "demand_counts": demand_counts.copy(),
                    "over_demanded": extras["over_demanded"],
                    "under_demanded": extras["under_demanded"],
                    "demand_changes": demand_changes,
                    "demand_change_rate": change_rate,
                    "max_price": max_price,
                    "price_std": price_std,
                    "delta": current_delta,
                    "is_boosted": is_boosted,
                }
            )

    return {
        "prices": prices,
        "demands": [set(bundle) for bundle in demands],
        "slack": slack,
        "is_boosted": is_boosted,
        "equilibrium": False,
        "rounds": max_rounds,
        "delta": current_delta,
        "meta": {"history": history},
    }


__all__ = ["run_airport_heuristic"]
