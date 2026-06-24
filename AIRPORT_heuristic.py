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

    from typing import Dict, Any, List, Set, Optional
import numpy as np

# Assuming _seed_all, compute_demand_bruteforce, and check_equilibrium are defined elsewhere
# def _seed_all(seed): ...
# def compute_demand_bruteforce(player, prices, budget): ...
# def check_equilibrium(demands, capacities, prices, alpha): ...

def run_airport_heuristic_2D_alpha(
    instance: Dict[str, Any],
    max_rounds: int = 200,
    delta: float = 0.5,
    epsilon_scale: float = 0.1,
    price_floor: float = 0.0,
    rng_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute the airport heuristic with 2D (alpha, beta) escalation and tatonnement updates.

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
    max_cap = int(instance["max_cap"])

    history: List[Dict[str, Any]] = []
    boost_round = 67

    # 2D Loop over alpha and beta
    # Sum goes 0, 1, 2, 3, 4 (matching previous range(0,5))
    for sum_ab in range(6):
        # Order: (sum_ab, 0), (sum_ab-1, 1), ..., (0, sum_ab)
        for alpha in range(sum_ab, -1, -1):
            beta = sum_ab - alpha
            
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

                # Check equilibrium using alpha instead of slack
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

                # ------------------------------ Step 2: Nudged re-optimization
                epsilon = epsilon_scale / max(1, num_items)
                nudged_prices = prices.copy()
                for idx in extras["over_demanded"]:
                    nudged_prices[idx] -= epsilon
                for idx in extras["under_demanded"]:
                    nudged_prices[idx] += epsilon

                # Calculate the average of the top 'beta' prices
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
                    
                    # Use alpha and the average top beta price instead of slack and max_price
                    soft_extra = beta * avg_top_beta_price * budget_ratio * rank_factor
                    relaxed_budget = budgets[player_idx] + soft_extra

                    new_bundle, _ = compute_demand_bruteforce(player, nudged_prices, relaxed_budget)
                    if new_bundle != demands[player_idx]:
                        demands[player_idx] = set(new_bundle)
                        nudged_changed = True

                if nudged_changed:
                    # Check equilibrium using alpha instead of slack
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

                # ------------------------------ Step 3: Tatonnement update
                # Update masks to use alpha instead of slack
                over_mask = demand_counts > capacities + alpha
                under_mask = (prices > price_floor) & (demand_counts < capacities - alpha)

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



def run_airport_heuristic_2D_beta(
    instance: Dict[str, Any],
    max_rounds: int = 200,
    delta: float = 0.5,
    epsilon_scale: float = 0.1,
    price_floor: float = 0.0,
    rng_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute the airport heuristic with 2D (alpha, beta) escalation and tatonnement updates.

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
    max_cap = int(instance["max_cap"])

    history: List[Dict[str, Any]] = []
    boost_round = 67

    # 2D Loop over alpha and beta
    # Sum goes 0, 1, 2, 3, 4 (matching previous range(0,5))
    for sum_ab in range(6):
        # Order: (sum_ab, 0), (sum_ab-1, 1), ..., (0, sum_ab)
        for beta in range(sum_ab, -1, -1):
            alpha = sum_ab - beta      
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

                # Check equilibrium using alpha instead of slack
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

                # ------------------------------ Step 2: Nudged re-optimization
                epsilon = epsilon_scale / max(1, num_items)
                nudged_prices = prices.copy()
                for idx in extras["over_demanded"]:
                    nudged_prices[idx] -= epsilon
                for idx in extras["under_demanded"]:
                    nudged_prices[idx] += epsilon

                # Calculate the average of the top 'beta' prices
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
                    
                    # Use alpha and the average top beta price instead of slack and max_price
                    soft_extra = beta * avg_top_beta_price * budget_ratio * rank_factor
                    relaxed_budget = budgets[player_idx] + soft_extra

                    new_bundle, _ = compute_demand_bruteforce(player, nudged_prices, relaxed_budget)
                    if new_bundle != demands[player_idx]:
                        demands[player_idx] = set(new_bundle)
                        nudged_changed = True

                if nudged_changed:
                    # Check equilibrium using alpha instead of slack
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

                # ------------------------------ Step 3: Tatonnement update
                # Update masks to use alpha instead of slack
                over_mask = demand_counts > capacities + alpha
                under_mask = (prices > price_floor) & (demand_counts < capacities - alpha)

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
from AIRPORT_generation import generate_airport_instance

def expected_utility_airport(target_player_id, true_feasible_pairs, fake_feasible_pairs, target_budget, base_instance_params, fixed_schedules, num_samples=5, heuristic_kwargs=None):
    """
    Evaluates the expected SURPLUS and expected PRICES of submitting 'fake_feasible_pairs'.
    """
    if heuristic_kwargs is None:
        heuristic_kwargs = {"max_rounds": 200, "delta": 0.5, "epsilon_scale": 0.1, "price_floor": 0.0}

    total_true_surplus = 0.0
    num_items = base_instance_params["num_items"]
    price_sums = np.zeros(num_items)
    
    for _ in range(num_samples):
        seed = int(np.random.randint(0, 1_000_000))
        instance = generate_airport_instance(**base_instance_params, rng_seed=seed, fixed_schedules=fixed_schedules)
        
        instance["players"][target_player_id]["feasible_pairs"] = copy.deepcopy(fake_feasible_pairs)
        instance["budgets"][target_player_id] = float(target_budget)
        
        result = run_airport_heuristic(instance, rng_seed=seed, **heuristic_kwargs)
        
        received_bundle = result["demands"][target_player_id]  
        final_prices = result["prices"] 
        
        true_gross_u = 0.0
        total_price_paid = 0.0
        
        if len(received_bundle) > 0:
            for pair_info in true_feasible_pairs:
                if set(pair_info["pair"]) == received_bundle:
                    true_gross_u = pair_info["utility"]
                    break
            
            total_price_paid = sum(final_prices[item] for item in received_bundle)
                    
        true_surplus = true_gross_u - total_price_paid
        total_true_surplus += true_surplus
        
        # Track prices for all items in this market sample
        for j in range(num_items):
            price_sums[j] += final_prices[j]
            
    avg_surplus = total_true_surplus / num_samples
    avg_prices = price_sums / num_samples
        
    return avg_surplus, avg_prices

import copy

def hill_climb_manipulation_airport(target_player_id, base_instance_params, fixed_schedules, target_budget, true_feasible_pairs, eta=2.0, explore_samples=7, verify_samples=50, max_iters=20, heuristic_kwargs=None):
    """
    Hill-climber that tracks both utility and prices at every step.
    Modified to use a two-tiered sample check: fast explore (7) and deep verification (50).
    """
    current_fake_pairs = copy.deepcopy(true_feasible_pairs)
    
    # 1. INITIAL UTILITY CALCULATION (Deep Verification - 50 samples)
    current_expected_u, current_expected_prices = expected_utility_airport(
        target_player_id, true_feasible_pairs, current_fake_pairs, 
        target_budget, base_instance_params, fixed_schedules, num_samples=verify_samples, heuristic_kwargs=heuristic_kwargs
    )
    
    utility_history = [current_expected_u]
    price_history = [current_expected_prices] # Track price arrays
    
    print(f"Starting Truthful Expected Utility ({verify_samples} samples): {current_expected_u:.2f}")
    
    for iteration in range(max_iters):
        best_tweak_pairs = None
        best_tweak_u = current_expected_u
        best_tweak_prices = current_expected_prices
        
        for idx, pair_info in enumerate(current_fake_pairs):
            for sign in [1, -1]:
                test_pairs = copy.deepcopy(current_fake_pairs)
                test_pairs[idx]["utility"] = max(0.0, test_pairs[idx]["utility"] + (sign * eta))
                
                # 2. FAST CHECK (Explore level - 7 samples)
                # We discard the returned prices here since we are just checking for utility improvement
                test_u_fast, _ = expected_utility_airport(
                    target_player_id, true_feasible_pairs, test_pairs, 
                    target_budget, base_instance_params, fixed_schedules, num_samples=explore_samples, heuristic_kwargs=heuristic_kwargs
                )
                
                # 3. VERIFY IF PROPOSING AN IMPROVEMENT (Deep Verification - 50 samples)
                if test_u_fast > best_tweak_u:
                    
                    test_u_verified, test_prices_verified = expected_utility_airport(
                        target_player_id, true_feasible_pairs, test_pairs, 
                        target_budget, base_instance_params, fixed_schedules, num_samples=verify_samples, heuristic_kwargs=heuristic_kwargs
                    )
                    
                    # Only accept if the 50-sample test CONFIRMS the improvement
                    if test_u_verified > best_tweak_u:
                        best_tweak_u = test_u_verified
                        best_tweak_pairs = copy.deepcopy(test_pairs)
                        best_tweak_prices = test_prices_verified.copy()
        
        if best_tweak_pairs is not None:
            current_fake_pairs = best_tweak_pairs
            current_expected_u = best_tweak_u
            current_expected_prices = best_tweak_prices
            
            utility_history.append(current_expected_u)
            price_history.append(current_expected_prices)
            print(f"Iteration {iteration+1}: Found verified improvement! New Utility: {current_expected_u:.2f}")
        else:
            print(f"Iteration {iteration+1}: No local tweaks passed 50-sample verification. Stopping.")
            break
            
    return current_fake_pairs, utility_history, price_history

# Update the __all__ export at the bottom of the file to include these
__all__ = ["run_airport_heuristic", "expected_utility_airport", "hill_climb_manipulation_airport"]