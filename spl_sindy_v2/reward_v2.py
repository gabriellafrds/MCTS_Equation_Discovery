# spl_sindy_v2/reward_v2.py
#
# Reward function for single-feature MCTS search.
#
# Uses R^2 from simple linear regression: y_dot ~ alpha * phi(x)
# This is the same reward discussed in the conversation —
# it measures how much variance of y_dot is explained by phi alone,
# which is more discriminating than Pearson correlation.
#
# Key difference from the original SPL reward:
#   - No Powell optimization (no C placeholders in grammar)
#   - R^2 instead of 1/(1+MSE) — better at discriminating useful features
#   - eta=0.99 (stronger parsimony) instead of 0.9999

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tree import build_tree_step_by_step
from spl_sindy_v2.evaluate_v2 import evaluate_node


def compute_reward(sequence, data, grammar, eta=0.99):
    """
    Compute the reward for a single candidate basis function phi(x).

    Reward = eta^n * R^2

    where:
        eta^n  : parsimony penalty (eta=0.99, n=sequence length)
                 eta=0.99 means a 10-rule expression is penalized by
                 0.99^10 = 0.904 vs 0.99^2 = 0.980 for a 2-rule one.
                 This is much more discriminating than eta=0.9999.
        R^2    : coefficient of determination from fitting y_dot ~ alpha*phi
                 Measures what fraction of variance in y_dot is explained
                 by a single linear rescaling of phi(x).

    Why R^2 and not Pearson correlation?
        Pearson measures linear association but does not account for
        the scale of the fit. R^2 from the optimal linear fit gives
        the same information but is bounded in [0, 1] and more
        naturally interpretable as "explained variance".

    sequence : list of (left, right) rules from MCTS
    data     : dict with "variables" and "y_dot"
    grammar  : grammar object with is_complete
    eta      : parsimony discount (default 0.99, stronger than SPL's 0.9999)

    Returns a scalar reward in [0, 1].
    """
    if not grammar.is_complete(sequence):
        return 0.0

    variables = data["variables"]
    y_dot     = data["y_dot"]

    try:
        tree = build_tree_step_by_step(sequence)
        phi  = evaluate_node(tree, variables)

        # Reject non-finite or constant functions
        if not np.all(np.isfinite(phi)):
            return 0.0
        if np.std(phi) < 1e-10:
            return 0.0   # constant function — useless as basis

        # Optimal linear coefficient: alpha = <phi, y_dot> / <phi, phi>
        # This minimizes ||y_dot - alpha*phi||^2
        alpha  = np.dot(phi, y_dot) / (np.dot(phi, phi) + 1e-10)
        y_pred = alpha * phi

        # R^2 = 1 - SS_res / SS_tot
        ss_res = np.sum((y_dot - y_pred) ** 2)
        ss_tot = np.sum((y_dot - np.mean(y_dot)) ** 2)

        if ss_tot < 1e-10:
            return 0.0

        r2 = max(0.0, 1.0 - ss_res / ss_tot)

        # Parsimony: penalize long sequences
        n         = len(sequence)
        parsimony = eta ** n

        return float(parsimony * r2)

    except Exception:
        return 0.0


def scale_reward(reward, global_max_reward):
    """
    Normalize reward by current global maximum.
    Implements Equation 3 of the SPL paper — keeps Q in [0,1] for UCT.
    """
    if global_max_reward <= 0:
        return 0.0
    return reward / global_max_reward
