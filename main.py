import numpy as np
from scipy.integrate import odeint
from grammar import get_valid_actions, is_complete, RULES
from tree import build_tree_step_by_step, extract_features_from_tree
from evaluate import evaluate_tree_sindy
from mcts import MCTS

# ---------------------------------------------------------------------------
# Data generation (Harmonic Oscillator Benchmark)
# ---------------------------------------------------------------------------

def generate_data_harmonic_oscillator(n_points=1000, noise=0.0):
    """
    Generate data for the Damped Harmonic Oscillator:
        dx/dt = -0.1*x + y
        dy/dt = -x - 0.1*y
    """
    def harmonic_oscillator(z, t):
        return [-0.1*z[0] + z[1], -z[0] - 0.1*z[1]]

    dt = 0.05
    t_train = np.arange(0, n_points * dt, dt)
    z0 = [1.0, 0.0]
    z_train = odeint(harmonic_oscillator, z0, t_train)

    if noise > 0:
        z_train += np.random.normal(0, noise, z_train.shape)

    # We target discovering the underlying physics of dx/dt
    y_dot = np.array([harmonic_oscillator(z, 0)[0] for z in z_train])

    data = {
        "variables": {"x": z_train[:, 0], "y": z_train[:, 1]},
        "y_dot": y_dot
    }
    return data


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def sequence_to_library_strings(sequence):
    """
    Converts a sequence of MCTS production rules into a readable list of 
    basis feature strings (e.g. ['x', 'sin(y)', 'x * y']).
    """
    tree = build_tree_step_by_step(sequence)
    features = extract_features_from_tree(tree)
    
    def render_node(node):
        if node.is_terminal(): return node.symbol
        if len(node.children) == 1: return render_node(node.children[0])
        if len(node.children) == 2: return f"{node.children[0].symbol}({render_node(node.children[1])})"
        if len(node.children) == 3: return f"({render_node(node.children[0])} {node.children[1].symbol} {render_node(node.children[2])})"
        return "?"
        
    return [render_node(f) for f in features]


def print_results(best_sequence, best_reward, data):
    print("\n" + "="*60)
    print("SEARCH COMPLETE")
    print("="*60)

    if best_sequence is None:
        print("No valid equation found.")
        return

    print(f"Best raw MCTS reward: {best_reward:.6f}")
    
    tree = build_tree_step_by_step(best_sequence)
    feature_strings = sequence_to_library_strings(best_sequence)
    
    # Run the final evaluation to retrieve the STLSQ coefficients
    mse, coefs = evaluate_tree_sindy(tree, data["variables"], data["y_dot"])
    
    print("\n--- Discovered SINDy Equation ---")
    print(f"MSE: {mse:.6e}")
    if len(coefs) > 0:
        eq_parts = []
        flat_coefs = np.ravel(coefs)
        for i, c in enumerate(flat_coefs):
            if abs(c) > 1e-5: # filter out zero or near-zero coefficients
                eq_parts.append(f"{c:.4f} * {feature_strings[i]}")
        
        if not eq_parts:
            print("dx/dt = 0")
        else:
            print("dx/dt = " + " + ".join(eq_parts))
    else:
        print("dx/dt = 0 (No features survived thresholding)")
        
    print(f"\nFull Feature Library Generated: {feature_strings}")
    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = {
        "n_episodes"    : 5000,   # Needs enough episodes to organically find the structure ['x', 'y']
        "n_simulations" : 10,     # Rollouts per expansion
        "t_max"         : 20,     # Maximum sequence length
        "c"             : 1.0,    # UCT exploration constant
        "lambda_val"    : 0.01,   # SINDy sparsity penalty (L0)
        "alpha"         : 0.005,  # MCTS sequence length penalty
        "beta"          : 1.0,    # Physical validity penalty
        "noise"         : 0.0,    # Noise level on target data
        "n_points"      : 1000,   # High point count for good sparse regression
    }

    print("Symbolic Physics Learner — Dynamic PySINDy Integration")
    print("Target: Damped Harmonic Oscillator (dx/dt = -0.1*x + 1.0*y)")
    print(f"Episodes        : {config['n_episodes']}")
    
    data = generate_data_harmonic_oscillator(n_points=config["n_points"], noise=config["noise"])

    import types
    grammar = types.SimpleNamespace(
        get_valid_actions = get_valid_actions,
        is_complete       = is_complete,
        rules             = RULES,
    )

    print("\nStarting search...")
    agent = MCTS(
        grammar=grammar,
        data=data,
        c=config["c"],
        n_simulations=config["n_simulations"],
        t_max=config["t_max"],
        lambda_val=config["lambda_val"],
        alpha=config["alpha"],
        beta=config["beta"]
    )

    best_sequence, best_reward = agent.run(n_episodes=config["n_episodes"])
    print_results(best_sequence, best_reward, data)


if __name__ == "__main__":
    main()