import numpy as np
from tree import build_tree_step_by_step
from evaluate import evaluate_tree_sindy

# ---------------------------------------------------------------------------
# Reward function (Hybrid SPL + SINDy)
# ---------------------------------------------------------------------------

def compute_reward(sequence, data, grammar, lambda_val=0.01, alpha=0.005, beta=1.0):
    """
    Compute the reward for a complete sequence of production rules.

    This implements an additive penalty reward model:
        r = 1 / (1 + MSE + lambda * ||Xi||_0 + alpha * C_active + beta * P_physics)

    where:
        MSE         measures the STLSQ sparse regression fit.
        ||Xi||_0    is the number of surviving features in the SINDy model.
        C_active    is the structural complexity of the generated features (sequence length).
        P_physics   is a boolean penalty (1 if breaks physical rules, 0 if valid).
    """
    # Safety check: sequence must be complete
    if not grammar.is_complete(sequence):
        return 0.0

    # --- Step 1: reconstruct the expression tree ---
    tree = build_tree_step_by_step(sequence)

    variables = data["variables"]
    y_dot     = data["y_dot"]

    # --- Step 2: evaluate the feature matrix and fit STLSQ ---
    try:
        mse, coefs = evaluate_tree_sindy(tree, variables, y_dot)

        # Guard against nan or inf in the evaluation
        if not np.isfinite(mse) or mse > 1e10:
            return 0.0

    except Exception:
        # If anything goes wrong during evaluation, reward is 0
        return 0.0

    # --- Step 3: compute penalties ---
    
    # 1. ||Xi||_0 penalty (SINDy sparsity)
    l0_norm = np.count_nonzero(coefs) if len(coefs) > 0 else 0
    
    # 2. C_active penalty (Total grammar sequence length)
    c_active = len(sequence)
    
    # 3. P_physics penalty (Placeholder for hard boundary checks, currently 0)
    p_physics = 0.0 

    # --- Step 4: final reward ---
    penalty_sum = mse + (lambda_val * l0_norm) + (alpha * c_active) + (beta * p_physics)
    reward = 1.0 / (1.0 + penalty_sum)

    return reward


# ---------------------------------------------------------------------------
# Adaptive reward scaling (Equation 3 of the SPL paper)
# ---------------------------------------------------------------------------

def scale_reward(reward, global_max_reward):
    """
    Normalize the reward by the current global maximum.
    """
    if global_max_reward <= 0:
        return 0.0
    return reward / global_max_reward