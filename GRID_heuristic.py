"""
Heuristic solver for GRID Auctions.

This module mirrors the ACE heuristic style: it iteratively adjusts demands,
applies nudges, and updates prices with optional boosting. Gurobi is preferred
when available, with CBC as a fallback.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from GRID_generation import (
    check_equilibrium,
    compute_demand_IP,
    compute_demand_efficient,
)


def _seed_all(seed: Optional[int]) -> np.random.Generator:
    """Seed Python's RNG and return a dedicated NumPy generator."""
    if seed is not None:
        random.seed(seed)
    return np.random.default_rng(seed)


def _count_demands(demands: Sequence[Set[int]], num_items: int) -> np.ndarray:
    """Count how many players demand each item."""
    counts = np.zeros(num_items, dtype=float)
    for bundle in demands:
        for item in bundle:
            counts[item] += 1
    return counts


def run_grid_heuristic(
    instance: Dict[str, Any],
    max_rounds: int = 200,
    delta: float = 0.5,
    epsilon_scale: float = 0.1,
    price_floor: float = 0.0,
    rng_seed: Optional[int] = None,
    solver_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the GRID heuristic with slack escalation and price nudging.

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
    max_path_len = int(instance["max_path_length"])

    history: List[Dict[str, Any]] = []

    for slack in range(0, 4 * max_path_len + 1):
        prices = np.zeros(num_items, dtype=float)
        current_delta = float(delta)
        is_boosted = False
        demands: List[Set[int]] = [set() for _ in range(num_players)]

        for round_index in range(1, max_rounds + 1):
            # ------------------------------ Demand computation (efficient + IP)
            new_demands: List[Set[int]] = []
            prev_demands = demands
            for player_idx, player in enumerate(players):
                efficient = compute_demand_efficient(player, prices, budgets[player_idx])
                if efficient is not None:
                    bundle, _ = efficient
                else:
                    bundle, _ = compute_demand_IP(
                        player,
                        prices,
                        budgets[player_idx],
                        nudges=None,
                        soft_budget=0.0,
                        solver_name=solver_name,
                    )
                new_demands.append(set(bundle))

            demands = new_demands
            demand_changes = sum(
                1 for idx in range(num_players) if demands[idx] != prev_demands[idx]
            )
            change_rate = demand_changes / max(1, num_players)

            # ------------------------------ Equilibrium check
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

            # ------------------------------ Nudging via relaxed IP
            epsilon = epsilon_scale / max(1, num_items)
            nudges = np.zeros(num_items, dtype=float)
            for idx in extras["over_demanded"]:
                nudges[idx] -= epsilon
            for idx in extras["under_demanded"]:
                nudges[idx] += epsilon

            sorted_prices = np.sort(prices)[::-1]
            top_k = min(slack, prices.size)
            avg_top_beta_price = float(np.mean(sorted_prices[:top_k])) if top_k > 0 else 0.0
            nudged_changed = False
            for player_idx, player in enumerate(players):
                budget_ratio = budgets[player_idx] / max_budget_value if max_budget_value > 0 else 1.0
                rank_factor = (player_idx + 1) / max(1, num_players)
                soft_extra = slack * avg_top_beta_price * budget_ratio * rank_factor

                new_bundle, _ = compute_demand_IP(
                    player,
                    prices,
                    budgets[player_idx],
                    nudges=nudges,
                    soft_budget=soft_extra,
                    solver_name=solver_name,
                )

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

            # ------------------------------ Price update
            over_mask = demand_counts > capacities + slack
            under_mask = (prices > price_floor) & (demand_counts < capacities - slack)

            prices = prices.copy()
            prices[over_mask] += current_delta
            prices[under_mask] = np.maximum(price_floor, prices[under_mask] - current_delta)

            # ------------------------------ Boosting to escape stagnation
            if (
                round_index > max(20, max_rounds / 3)
                and not is_boosted
                and num_items > 0
            ):
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

    # Termination without equilibrium.
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


from typing import Dict, Any, List, Set, Optional
import numpy as np

def run_grid_heuristic_2D_alpha(
    instance: Dict[str, Any],
    max_rounds: int = 200,
    delta: float = 0.5,
    epsilon_scale: float = 0.1,
    price_floor: float = 0.0,
    rng_seed: Optional[int] = None,
    solver_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the GRID heuristic with 2D (alpha, beta) escalation and price nudging.

    Returns a dictionary capturing the final prices, demands, alpha, beta,
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
    max_path_len = int(instance["max_path_length"])

    history: List[Dict[str, Any]] = []

    # 2D Loop over alpha and beta
    for sum_ab in range(4 * max_path_len + 1):
        # Order: (sum_ab, 0), (sum_ab-1, 1), ..., (0, sum_ab)
        for alpha in range(sum_ab, -1, -1):
            beta = sum_ab - alpha

            prices = np.zeros(num_items, dtype=float)
            current_delta = float(delta)
            is_boosted = False
            demands: List[Set[int]] = [set() for _ in range(num_players)]

            for round_index in range(1, max_rounds + 1):
                # ------------------------------ Demand computation (efficient + IP)
                new_demands: List[Set[int]] = []
                prev_demands = demands
                for player_idx, player in enumerate(players):
                    efficient = compute_demand_efficient(player, prices, budgets[player_idx])
                    if efficient is not None:
                        bundle, _ = efficient
                    else:
                        bundle, _ = compute_demand_IP(
                            player,
                            prices,
                            budgets[player_idx],
                            nudges=None,
                            soft_budget=0.0,
                            solver_name=solver_name,
                        )
                    new_demands.append(set(bundle))

                demands = new_demands
                demand_changes = sum(
                    1 for idx in range(num_players) if demands[idx] != prev_demands[idx]
                )
                change_rate = demand_changes / max(1, num_players)

                # ------------------------------ Equilibrium check (Uses alpha)
                equilibrium, demand_counts, extras = check_equilibrium(
                    demands,
                    capacities,
                    prices,
                    alpha,
                )
                if equilibrium:
                    return {
                        "prices": prices,
                        "demands": [set(bundle) for bundle in demands],
                        "alpha": alpha,
                        "beta": beta,
                        "is_boosted": is_boosted,
                        "equilibrium": True,
                        "rounds": round_index,
                        "delta": current_delta,
                        "meta": {"history": history},
                    }

                # ------------------------------ Nudging via relaxed IP
                epsilon = epsilon_scale / max(1, num_items)
                nudges = np.zeros(num_items, dtype=float)
                for idx in extras["over_demanded"]:
                    nudges[idx] -= epsilon
                for idx in extras["under_demanded"]:
                    nudges[idx] += epsilon

                # Average of the top 'beta' prices
                if beta > 0 and prices.size > 0:
                    sorted_prices = np.sort(prices)[::-1]
                    top_k = min(beta, prices.size)
                    avg_top_beta_price = float(np.mean(sorted_prices[:top_k]))
                else:
                    avg_top_beta_price = 0.0

                nudged_changed = False
                for player_idx, player in enumerate(players):
                    budget_ratio = budgets[player_idx] / max_budget_value if max_budget_value > 0 else 1.0
                    rank_factor = (player_idx + 1) / max(1, num_players)
                    
                    # Budget relaxation needs beta
                    soft_extra = beta * avg_top_beta_price * budget_ratio * rank_factor

                    new_bundle, _ = compute_demand_IP(
                        player,
                        prices,
                        budgets[player_idx],
                        nudges=nudges,
                        soft_budget=soft_extra,
                        solver_name=solver_name,
                    )

                    if new_bundle != demands[player_idx]:
                        demands[player_idx] = set(new_bundle)
                        nudged_changed = True

                if nudged_changed:
                    # Equilibrium check uses alpha
                    equilibrium, demand_counts, extras = check_equilibrium(
                        demands,
                        capacities,
                        prices,
                        alpha,
                    )
                    if equilibrium:
                        return {
                            "prices": prices,
                            "demands": [set(bundle) for bundle in demands],
                            "alpha": alpha,
                            "beta": beta,
                            "is_boosted": is_boosted,
                            "equilibrium": True,
                            "rounds": round_index,
                            "delta": current_delta,
                            "meta": {"history": history},
                        }

                # ------------------------------ Price update (Uses alpha)
                over_mask = demand_counts > capacities + alpha
                under_mask = (prices > price_floor) & (demand_counts < capacities - alpha)

                prices = prices.copy()
                prices[over_mask] += current_delta
                prices[under_mask] = np.maximum(price_floor, prices[under_mask] - current_delta)

                # ------------------------------ Boosting to escape stagnation
                if (
                    round_index > max(20, max_rounds / 3)
                    and not is_boosted
                    and num_items > 0
                ):
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
                        "alpha": alpha,
                        "beta": beta,
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

    # Termination without equilibrium.
    return {
        "prices": prices,
        "demands": [set(bundle) for bundle in demands],
        "alpha": alpha,
        "beta": beta,
        "is_boosted": is_boosted,
        "equilibrium": False,
        "rounds": max_rounds,
        "delta": current_delta,
        "meta": {"history": history},
    }


from typing import Dict, Any, List, Set, Optional
import numpy as np

def run_grid_heuristic_2D_beta(
    instance: Dict[str, Any],
    max_rounds: int = 200,
    delta: float = 0.5,
    epsilon_scale: float = 0.1,
    price_floor: float = 0.0,
    rng_seed: Optional[int] = None,
    solver_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the GRID heuristic with 2D (alpha, beta) escalation and price nudging.

    Returns a dictionary capturing the final prices, demands, alpha, beta,
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
    max_path_len = int(instance["max_path_length"])

    history: List[Dict[str, Any]] = []

    # 2D Loop over alpha and beta
    for sum_ab in range(4 * max_path_len + 1):
        # Order: (sum_ab, 0), (sum_ab-1, 1), ..., (0, sum_ab)
        for beta in range(sum_ab, -1, -1):
            alpha = sum_ab - beta

            prices = np.zeros(num_items, dtype=float)
            current_delta = float(delta)
            is_boosted = False
            demands: List[Set[int]] = [set() for _ in range(num_players)]

            for round_index in range(1, max_rounds + 1):
                # ------------------------------ Demand computation (efficient + IP)
                new_demands: List[Set[int]] = []
                prev_demands = demands
                for player_idx, player in enumerate(players):
                    efficient = compute_demand_efficient(player, prices, budgets[player_idx])
                    if efficient is not None:
                        bundle, _ = efficient
                    else:
                        bundle, _ = compute_demand_IP(
                            player,
                            prices,
                            budgets[player_idx],
                            nudges=None,
                            soft_budget=0.0,
                            solver_name=solver_name,
                        )
                    new_demands.append(set(bundle))

                demands = new_demands
                demand_changes = sum(
                    1 for idx in range(num_players) if demands[idx] != prev_demands[idx]
                )
                change_rate = demand_changes / max(1, num_players)

                # ------------------------------ Equilibrium check (Uses alpha)
                equilibrium, demand_counts, extras = check_equilibrium(
                    demands,
                    capacities,
                    prices,
                    alpha,
                )
                if equilibrium:
                    return {
                        "prices": prices,
                        "demands": [set(bundle) for bundle in demands],
                        "alpha": alpha,
                        "beta": beta,
                        "is_boosted": is_boosted,
                        "equilibrium": True,
                        "rounds": round_index,
                        "delta": current_delta,
                        "meta": {"history": history},
                    }

                # ------------------------------ Nudging via relaxed IP
                epsilon = epsilon_scale / max(1, num_items)
                nudges = np.zeros(num_items, dtype=float)
                for idx in extras["over_demanded"]:
                    nudges[idx] -= epsilon
                for idx in extras["under_demanded"]:
                    nudges[idx] += epsilon

                # Average of the top 'beta' prices
                if beta > 0 and prices.size > 0:
                    sorted_prices = np.sort(prices)[::-1]
                    top_k = min(beta, prices.size)
                    avg_top_beta_price = float(np.mean(sorted_prices[:top_k]))
                else:
                    avg_top_beta_price = 0.0

                nudged_changed = False
                for player_idx, player in enumerate(players):
                    budget_ratio = budgets[player_idx] / max_budget_value if max_budget_value > 0 else 1.0
                    rank_factor = (player_idx + 1) / max(1, num_players)
                    
                    # Budget relaxation needs beta
                    soft_extra = beta * avg_top_beta_price * budget_ratio * rank_factor

                    new_bundle, _ = compute_demand_IP(
                        player,
                        prices,
                        budgets[player_idx],
                        nudges=nudges,
                        soft_budget=soft_extra,
                        solver_name=solver_name,
                    )

                    if new_bundle != demands[player_idx]:
                        demands[player_idx] = set(new_bundle)
                        nudged_changed = True

                if nudged_changed:
                    # Equilibrium check uses alpha
                    equilibrium, demand_counts, extras = check_equilibrium(
                        demands,
                        capacities,
                        prices,
                        alpha,
                    )
                    if equilibrium:
                        return {
                            "prices": prices,
                            "demands": [set(bundle) for bundle in demands],
                            "alpha": alpha,
                            "beta": beta,
                            "is_boosted": is_boosted,
                            "equilibrium": True,
                            "rounds": round_index,
                            "delta": current_delta,
                            "meta": {"history": history},
                        }

                # ------------------------------ Price update (Uses alpha)
                over_mask = demand_counts > capacities + alpha
                under_mask = (prices > price_floor) & (demand_counts < capacities - alpha)

                prices = prices.copy()
                prices[over_mask] += current_delta
                prices[under_mask] = np.maximum(price_floor, prices[under_mask] - current_delta)

                # ------------------------------ Boosting to escape stagnation
                if (
                    round_index > max(20, max_rounds / 3)
                    and not is_boosted
                    and num_items > 0
                ):
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
                        "alpha": alpha,
                        "beta": beta,
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

    # Termination without equilibrium.
    return {
        "prices": prices,
        "demands": [set(bundle) for bundle in demands],
        "alpha": alpha,
        "beta": beta,
        "is_boosted": is_boosted,
        "equilibrium": False,
        "rounds": max_rounds,
        "delta": current_delta,
        "meta": {"history": history},
    }


import copy
import numpy as np
from GRID_generation import generate_grid_instance


def expected_utility_grid(target_player_id, true_V_i, fake_V_i, target_budget, base_instance_params, num_samples=5, heuristic_kwargs=None):
    if heuristic_kwargs is None:
        heuristic_kwargs = {"max_rounds": 200, "delta": 0.5, "epsilon_scale": 0.1, "price_floor": 0.0}

    total_true_surplus = 0.0
    total_prices = None # Will initialize on first run to match the item array size
    
    # Pre-map true item utilities for rapid surplus calculation
    true_item_utilities = {}
    for dir_name, path in true_V_i.items():
        for item_idx, util in path:
            true_item_utilities[item_idx] = util
            
    for _ in range(num_samples):
        # 1. Resample private info (Budgets/Utilities) but keep Public info fixed
        seed = int(np.random.randint(0, 1_000_000))
        instance = generate_grid_instance(**base_instance_params, rng_seed=seed)
        
        # 2. Inject fake bid and real budget
        instance["players"][target_player_id]["paths"] = copy.deepcopy(fake_V_i)
        instance["budgets"][target_player_id] = float(target_budget)
        
        # 3. Run heuristic
        result = run_grid_heuristic(instance, rng_seed=seed, **heuristic_kwargs)
        received_bundle = result["demands"][target_player_id] 
        final_prices = result["prices"] 
        
        # Track prices across samples
        if total_prices is None:
            total_prices = np.zeros_like(final_prices)
        total_prices += final_prices
        
        # 4. Calculate Surplus
        true_gross_u = sum(true_item_utilities.get(item, 0.0) for item in received_bundle)
        total_price_paid = sum(final_prices[item] for item in received_bundle)
        
        total_true_surplus += (true_gross_u - total_price_paid)
        
    expected_surplus = total_true_surplus / num_samples
    expected_prices = total_prices / num_samples if total_prices is not None else np.array([])
    
    # Now returns a tuple: (Expected Surplus, Expected Prices)
    return expected_surplus, expected_prices


def hill_climb_manipulation_grid(target_player_id, base_instance_params, target_budget, true_V_i, eta=2.0, explore_samples=7, verify_samples=50, max_iters=20, heuristic_kwargs=None):
    """
    Algorithm 2: Hill-climber to find profitable manipulations in the GRID auction.
    Modified to use a two-tiered sample check: fast explore (7) and deep verification (50).
    """
    current_fake_V = copy.deepcopy(true_V_i)
    
    # 1. INITIAL UTILITY CALCULATION (Deep Verification - 50 samples)
    current_expected_u, current_expected_prices = expected_utility_grid(
        target_player_id, true_V_i, current_fake_V, 
        target_budget, base_instance_params, num_samples=verify_samples, heuristic_kwargs=heuristic_kwargs
    )
    
    utility_history = [current_expected_u]
    price_history = [current_expected_prices] # Track price arrays at every step
    
    print(f"Starting Truthful Expected Utility ({verify_samples} samples): {current_expected_u:.2f}")
    
    for iteration in range(max_iters):
        best_tweak_V = None
        best_tweak_u = current_expected_u
        best_tweak_prices = current_expected_prices
        
        for dir_name, path in current_fake_V.items():
            if not path:
                continue 
                
            for sign in [1, -1]:
                test_V = copy.deepcopy(current_fake_V)
                
                tweaked_path = []
                for item_idx, util in test_V[dir_name]:
                    new_util = max(0.0, util + (sign * eta))
                    tweaked_path.append((item_idx, new_util))
                
                test_V[dir_name] = tweaked_path
                
                # 2. FAST CHECK (Explore level - 7 samples)
                # We don't strictly need the prices here if we are just checking for utility improvement
                test_u_fast, _ = expected_utility_grid(
                    target_player_id, true_V_i, test_V, 
                    target_budget, base_instance_params, num_samples=explore_samples, heuristic_kwargs=heuristic_kwargs
                )
                
                # 3. VERIFY IF PROPOSING AN IMPROVEMENT (Deep Verification - 50 samples)
                # Compare against the current best known utility (which was calculated at 50 samples)
                if test_u_fast > best_tweak_u:
                    
                    test_u_verified, test_prices_verified = expected_utility_grid(
                        target_player_id, true_V_i, test_V, 
                        target_budget, base_instance_params, num_samples=verify_samples, heuristic_kwargs=heuristic_kwargs
                    )
                    
                    # Only accept and overwrite if the 50-sample test CONFIRMS the improvement
                    if test_u_verified > best_tweak_u:
                        best_tweak_u = test_u_verified
                        best_tweak_V = copy.deepcopy(test_V)
                        best_tweak_prices = test_prices_verified.copy()
        
        if best_tweak_V is not None:
            current_fake_V = best_tweak_V
            current_expected_u = best_tweak_u
            current_expected_prices = best_tweak_prices
            
            utility_history.append(current_expected_u)
            price_history.append(current_expected_prices)
            print(f"Iteration {iteration+1}: Found verified improvement! New Expected Utility: {current_expected_u:.2f}")
        else:
            print(f"Iteration {iteration+1}: No local tweaks passed 50-sample verification. Stopping.")
            break
            
    # Return the fake bid, utility history, AND price history
    return current_fake_V, utility_history, price_history

__all__ = ["run_grid_heuristic", "expected_utility_grid", "hill_climb_manipulation_grid"]
