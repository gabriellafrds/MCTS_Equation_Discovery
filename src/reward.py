import numpy as np
import warnings
from src.tree import build_tree_step_by_step, count_nodes
from src.evaluate import evaluate_tree_sindy

def compute_reward(sequence, data, grammar, locked_features_cols=None, gamma=0.01, alpha=0.005):
    """
    Evaluate AST sequences using a BIC-penalized STLSQ reward.
    """
    if not grammar.is_complete(sequence):
        return 0.0

    locked_features_cols = locked_features_cols or []
    tree = build_tree_step_by_step(sequence)
    variables = data["variables"]
    y_dot = data["y_dot"]
    n_points = len(y_dot)

    # Safely catch singular matrix warnings during search
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mse, coefs = evaluate_tree_sindy(tree, variables, y_dot, locked_features_cols)
            if not np.isfinite(mse) or mse > 1e10:
                return 0.0
        except BaseException:
            return 0.0

    l0_norm = np.count_nonzero(coefs) if len(coefs) > 0 else 0
    mse_clamped = max(mse, 1e-12)
    bic = n_points * np.log(mse_clamped) + l0_norm * np.log(n_points)
    
    exp_decay = np.exp(np.clip(-gamma * bic, -700, 700))
    complexity_penalty = 1.0 / (1.0 + alpha * count_nodes(tree))
    
    return float(exp_decay * complexity_penalty)

def scale_reward(reward, global_max_reward):
    """
    Scale rewards to [0, 1] relative to the best observed outcome.
    """
    return reward / global_max_reward if global_max_reward > 0 else 0.0