import numpy as np
from tree import build_tree_step_by_step
from evaluate import evaluate_tree, count_constants, estimate_constants

# ---------------------------------------------------------------------------
# Reward function (Equation 2 of the SPL paper)
# ---------------------------------------------------------------------------

def compute_reward(sequence, data, grammar, eta=0.9999):
    """
    Compute the reward for a complete sequence of production rules.

    This implements Equation 2 from the SPL paper:

        r = eta^n / (1 + MSE(y_dot, f_tilde(Y)))

    where:
        eta^n       penalizes complexity (parsimony term)
        n           is the number of production rules in the tree
        MSE         measures how well the equation fits the data
        y_dot       is the measured state derivative
        f_tilde(Y)  is the equation's prediction on the input data

    sequence  : list of rules (left, right) chosen by the MCTS agent
    data      : dict with keys "variables" and "y_dot"
                  - variables : dict {"x": array, "y": array, ...}
                  - y_dot     : target derivative array
    grammar   : Grammar object (used to check completeness)
    eta       : parsimony discount factor (default 0.9999 as in paper)

    Returns a scalar reward in (0, 1].
    """
    # Safety check: sequence must be complete
    if not grammar.is_complete(sequence):
        return 0.0

    # --- Step 1: reconstruct the expression tree ---
    tree = build_tree_step_by_step(sequence)

    # --- Step 2: estimate constant values if needed ---
    variables = data["variables"]
    y_dot     = data["y_dot"]

    try:
        n_constants = count_constants(tree)
        if n_constants > 0:
            # Optimize constant values using Powell's method (as in the paper)
            constants = estimate_constants(tree, variables, y_dot)
        else:
            constants = []

        # --- Step 3: evaluate the equation on the data ---
        y_pred = evaluate_tree(tree, variables, constants)

        # --- Step 4: compute MSE ---
        mse = np.mean((y_pred - y_dot) ** 2)

        # Guard against nan or inf in the evaluation
        if not np.isfinite(mse):
            return 0.0

    except Exception:
        # If anything goes wrong during evaluation, reward is 0
        return 0.0

    # --- Step 5: compute parsimony penalty ---
    # n = number of production rules = length of the sequence
    n = len(sequence)
    parsimony = eta ** n

    # --- Step 6: final reward (Equation 2) ---
    reward = parsimony / (1.0 + mse)

    return reward


# ---------------------------------------------------------------------------
# Adaptive reward scaling (Equation 3 of the SPL paper)
# ---------------------------------------------------------------------------

def scale_reward(reward, global_max_reward):
    """
    Normalize the reward by the current global maximum.
    This implements Equation 3 from the SPL paper:

        Q(s,a) = r*(s,a) / max_{s,a} Q(s,a)

    Keeps all Q values in [0, 1] regardless of data scale,
    ensuring the UCT exploration-exploitation balance stays calibrated.

    reward            : raw reward from compute_reward
    global_max_reward : best reward seen so far across all episodes

    Returns a scaled reward in [0, 1].
    """
    if global_max_reward <= 0:
        return 0.0
    return reward / global_max_reward