import sys
import numpy as np
import copy
import types
from mcts import MCTS
from tree import build_tree_step_by_step, extract_features_from_tree
from grammar import get_valid_actions, is_complete, RULES
from main import sequence_to_library_strings, print_final_equation, evaluate_locked_dictionary
from evaluate import evaluate_node, evaluate_tree_sindy
from data_generators import very_complex_three_var

# Explicitly defining the Report Benchmarks
def damped_harmonic_oscillator(n_points=500, noise=0.0):
    # dx/dt = -0.1x + y
    rng = np.random.default_rng(seed=42)
    x = rng.uniform(-2, 2, n_points)
    y = rng.uniform(-2, 2, n_points)
    x_dot = -0.1 * x + y
    if noise > 0:
        x_dot += rng.normal(0, noise, size=x_dot.shape)
    return {"variables": {"x": x, "y": y}, "y_dot": x_dot}

def full_lorenz_z(n_points=500, noise=0.0):
    # dz/dt = x * y - 2.66 * z
    t = np.linspace(0, 10, n_points)
    rng = np.random.default_rng(seed=42)
    x = rng.uniform(-10, 10, n_points)
    y = rng.uniform(-10, 10, n_points)
    z = rng.uniform(-10, 10, n_points)
    z_dot = x * y - 2.66 * z
    if noise > 0:
        z_dot += rng.normal(0, noise, size=z_dot.shape)
    return {"variables": {"x": x, "y": y, "z": z}, "y_dot": z_dot}

def run_benchmark(name, data_func, config):
    print(f"\n=======================================================")
    print(f"BENCHMARK: {name}")
    print(f"=======================================================\n")
    
    data = data_func(n_points=config["n_points"], noise=config["noise"])
    
    grammar = types.SimpleNamespace(
        get_valid_actions=get_valid_actions,
        is_complete=is_complete,
        rules=RULES
    )

    locked_features_cols = []
    locked_features_strings = []
    best_global_bic = float('inf')
    max_features = 10

    for iteration in range(max_features):
        print(f"\n--- Outputting Feature {iteration+1} ---")
        agent = MCTS(
            grammar=grammar,
            data=data,
            locked_features_cols=locked_features_cols,
            c=config["c"],
            n_simulations=config["n_simulations"],
            t_max=config["t_max"],
            gamma=config["gamma"],
            alpha=config["alpha"]
        )

        best_sequence, best_reward = agent.run(n_episodes=config["n_episodes"])
        if not best_sequence:
            print("MCTS Search Terminated.")
            break

        tree = build_tree_step_by_step(best_sequence)
        features = extract_features_from_tree(tree)
        if not features:
            continue

        new_str = sequence_to_library_strings(best_sequence)[0]
        new_col = evaluate_node(features[0], data["variables"])
        if np.isscalar(new_col):
            new_col = np.full(data["y_dot"].shape[0], new_col)
        else:
            new_col = new_col.reshape(-1)

        # PySINDy Evaluation
        mse, coefs = evaluate_tree_sindy(tree, data["variables"], data["y_dot"], locked_features_cols)
        l0_norm = np.count_nonzero(coefs)
        n_points = len(data["y_dot"])
        mse_clamped = max(mse, 1e-12)
        bic = n_points * np.log(mse_clamped) + l0_norm * np.log(n_points)

        print(f"Proposed: {new_str} | MSE: {mse:.4e} | BIC: {bic:.2f}")

        # Occam's Razor
        if new_str in locked_features_strings:
            print(f"Duplicated {new_str}. Terminating loop.")
            break

        if abs(np.ravel(coefs)[-1]) < 1e-5:
            print(f"-> REJECTED. SINDy assigned 0.0 to '{new_str}'. Terminating loop.")
            break

        if bic <= best_global_bic - 10.0:
            best_global_bic = bic
            locked_features_cols.append(new_col)
            locked_features_strings.append(new_str)
            print(f"-> ACCEPTED! Locked dictionary: {locked_features_strings}")
        else:
            print(f"-> REJECTED! BIC did not improve significantly. Terminating loop.")
            break

    # Final Output
    print(f"\n--- FINAL DISCOVERED EQUATION FOR {name} ---")
    mse, coefs = evaluate_locked_dictionary(locked_features_cols, data["y_dot"])
    formula_parts = []
    if len(coefs) > 0:
        flat_coefs = np.ravel(coefs)
        for i, c in enumerate(flat_coefs):
            if abs(c) > 1e-5 and i < len(locked_features_strings):
                formula_parts.append(f"{c:.4f} * {locked_features_strings[i]}")
    final_formula = " + ".join(formula_parts) if formula_parts else "dx/dt = 0"
    print(f"Formula: {final_formula}")
    print(f"Final MSE: {mse:.4e}")
    print(f"Final BIC: {best_global_bic:.2f}\n")


if __name__ == "__main__":
    config = {
        "n_episodes": 1000,
        "n_simulations": 20,
        "t_max": 15,
        "c": 0.5,
        "gamma": 0.005,
        "alpha": 0.5,
        "noise": 0.0,
        "n_points": 500
    }
    
    run_benchmark("1. Linear Baseline (Damped Harmonic Oscillator)", damped_harmonic_oscillator, config)
    run_benchmark("2. Chaotic System (Lorenz Attractor Z-axis)", full_lorenz_z, config)
    run_benchmark("3. Deep Nesting Bound (Very Complex Three Var)", very_complex_three_var, config)
