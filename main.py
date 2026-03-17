import numpy as np
from grammar import get_valid_actions, is_complete, RULES, NON_TERMINALS, TERMINALS
from tree import build_tree_step_by_step
from evaluate import evaluate_tree
from mcts import MCTS

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data_nguyen1(n_points=20, noise=0.0):
    """
    Generate data for the Nguyen-1 benchmark: y = x^3 + x^2 + x
    This is one of the standard symbolic regression benchmarks
    used in the SPL paper (Table 1).

    n_points : number of data points
    noise    : standard deviation of Gaussian noise added to y_dot

    Returns a data dict with keys "variables" and "y_dot".
    """
    # Input variable x uniformly sampled in [-1, 1]
    x = np.linspace(-1, 1, n_points)

    # Target equation: x^3 + x^2 + x
    y_dot = x**3 + x**2 + x

    # Add optional Gaussian noise
    if noise > 0:
        y_dot = y_dot + np.random.normal(0, noise, size=y_dot.shape)

    data = {
        "variables": {"x": x},
        "y_dot":     y_dot
    }
    return data


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def sequence_to_string(sequence):
    """
    Convert a sequence of production rules to a readable string.
    Extracts only the right-hand side symbols to show the expression structure.

    Example:
        [("f", ["M"]), ("M", ["M", "+", "M"]), ("M", ["x"]), ("M", ["x"])]
        -> "M | M + M | x | x"
    """
    parts = [" ".join(right) for (left, right) in sequence]
    return " | ".join(parts)


def print_results(best_sequence, best_reward, data):
    """
    Print the results of the MCTS search:
    - the sequence of rules chosen
    - the reward obtained
    - the predicted values vs target values on a few data points
    """
    print("\n" + "="*60)
    print("SEARCH COMPLETE")
    print("="*60)

    if best_sequence is None:
        print("No valid equation found.")
        return

    print(f"Best reward      : {best_reward:.6f}")
    print(f"Sequence length  : {len(best_sequence)} rules")
    print(f"Sequence         : {sequence_to_string(best_sequence)}")

    # Reconstruct and evaluate the best tree
    tree   = build_tree_step_by_step(best_sequence)
    y_pred = evaluate_tree(tree, data["variables"])
    y_dot  = data["y_dot"]

    print("\nPrediction vs target (first 5 points):")
    print(f"  {'x':>8}  {'predicted':>12}  {'target':>12}  {'error':>12}")
    x_vals = data["variables"]["x"]
    for i in range(min(5, len(x_vals))):
        error = abs(y_pred[i] - y_dot[i])
        print(f"  {x_vals[i]:>8.4f}  {y_pred[i]:>12.6f}  {y_dot[i]:>12.6f}  {error:>12.6f}")

    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Hyperparameters (from Table B.2 of the SPL paper) ---
    config = {
        "n_episodes"    : 10000,  # number of MCTS episodes
        "n_simulations" : 10,     # rollouts per expansion
        "t_max"         : 50,     # maximum sequence length
        "c"             : 1.0,    # UCT exploration constant
        "eta"           : 0.9999, # parsimony discount factor
        "noise"         : 0.0,    # noise level on target data
        "n_points"      : 20,     # number of data points
    }

    print("Symbolic Physics Learner — MCTS")
    print("Target equation: x^3 + x^2 + x  (Nguyen-1)")
    print(f"Episodes        : {config['n_episodes']}")
    print(f"Simulations     : {config['n_simulations']}")
    print(f"t_max           : {config['t_max']}")
    print(f"eta             : {config['eta']}")

    # --- Generate data ---
    data = generate_data_nguyen1(
        n_points=config["n_points"],
        noise=config["noise"]
    )

    # --- Build grammar object ---
    # We pass the grammar functions directly as a simple namespace object
    # so that mcts.py can call grammar.get_valid_actions and grammar.is_complete
    import types
    grammar = types.SimpleNamespace(
        get_valid_actions = get_valid_actions,
        is_complete       = is_complete,
        rules             = RULES,
    )

    # --- Run MCTS ---
    print("\nStarting search...")
    agent = MCTS(
        grammar=grammar,
        data=data,
        c=config["c"],
        n_simulations=config["n_simulations"],
        t_max=config["t_max"],
        eta=config["eta"],
    )

    best_sequence, best_reward = agent.run(n_episodes=config["n_episodes"])

    # --- Print results ---
    print_results(best_sequence, best_reward, data)


if __name__ == "__main__":
    main()