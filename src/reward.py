import numpy as np
from src.tree import build_tree_step_by_step, extract_features_from_tree, count_nodes
from src.evaluate import evaluate_tree_sindy

# ---------------------------------------------------------------------------
# Reward function (BIC-based)
# ---------------------------------------------------------------------------

def compute_reward(sequence, data, grammar, gamma=0.01, alpha=0.005, beta=0.05):
    """
    Compute the reward for a complete sequence of production rules.

    This implements the BIC-based exponential decay reward:
        R_final = exp(-gamma * BIC) / (1 + alpha * C_grammar)
    
    where:
        BIC = n_points * ln(MSE) + ||Xi||_0 * ln(n_points)
        C_grammar is the total grammatical length of the newly generated feature.
    """
def compute_reward(sequence, data, grammar, locked_features_cols=None, gamma=0.01, alpha=0.005):
    if locked_features_cols is None:
        locked_features_cols = []
        
    if not grammar.is_complete(sequence):
        return 0.0

    tree = build_tree_step_by_step(sequence)
    variables = data["variables"]
    y_dot     = data["y_dot"]
    n_points  = len(y_dot)

    try:
        mse, coefs = evaluate_tree_sindy(tree, variables, y_dot, locked_features_cols)
        if not np.isfinite(mse) or mse > 1e10:
            return 0.0
    except Exception:
        return 0.0

    l0_norm = np.count_nonzero(coefs) if len(coefs) > 0 else 0
    
    # Clip MSE to avoid log(0)
    mse_clamped = max(mse, 1e-12)
    
    # Calculate BIC
    bic = n_points * np.log(mse_clamped) + l0_norm * np.log(n_points)
    
    # Limit exponential argument to prevent float64 overflow
    exp_arg = np.clip(-gamma * bic, -700, 700)
    exp_decay = np.exp(exp_arg)

    # Calculate total structural complexity of the newly proposed feature
    c_grammar = count_nodes(tree)
    complexity_penalty = 1.0 / (1.0 + alpha * c_grammar)
    
    reward = exp_decay * complexity_penalty

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