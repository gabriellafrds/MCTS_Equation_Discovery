import numpy as np
import types
from src.mcts import MCTS
from src.tree import build_tree_step_by_step, extract_features_from_tree
from src.grammar import get_valid_actions, is_complete, RULES
from src.main import sequence_to_library_strings, evaluate_locked_dictionary
from src.evaluate import evaluate_node, evaluate_tree_sindy
from utils.data_generators import damped_harmonic_oscillator_rk45, lorenz_attractor_rk45, deep_nested_rk45

def run_benchmark(name, data_func, config):
    print(f"\n{'='*55}\nBENCHMARK: {name}\n{'='*55}")
    data = data_func(n_points=config["n_points"], noise=config["noise"])
    grammar = types.SimpleNamespace(get_valid_actions=get_valid_actions, is_complete=is_complete, rules=RULES)

    locked_cols, locked_strs = [], []
    best_bic = float('inf')

    for iteration in range(10):
        print(f"\n--- Outputting Feature {iteration + 1} ---")
        agent = MCTS(grammar, data, locked_cols, **{k: config[k] for k in ["c", "n_simulations", "t_max", "gamma", "alpha"]})
        best_seq, _ = agent.run(config["n_episodes"])
        
        if not best_seq:
            print("MCTS Search Terminated.")
            break

        tree = build_tree_step_by_step(best_seq)
        features = extract_features_from_tree(tree)
        if not features: continue

        new_str = sequence_to_library_strings(best_seq)[0]
        new_col = evaluate_node(features[0], data["variables"])
        new_col = np.full(data["y_dot"].shape[0], new_col).reshape(-1) if np.isscalar(new_col) else new_col.reshape(-1)

        mse, coefs = evaluate_tree_sindy(tree, data["variables"], data["y_dot"], locked_cols)
        n_points = len(data["y_dot"])
        bic = n_points * np.log(max(mse, 1e-12)) + np.count_nonzero(coefs) * np.log(n_points)

        print(f"Proposed: {new_str} | MSE: {mse:.4e} | BIC: {bic:.2f}")

        if new_str in locked_strs or abs(np.ravel(coefs)[-1]) < 1e-5:
            print("-> REJECTED (Duplicate or Zero Coefficient).")
            break

        if bic <= best_bic - 10.0:
            best_bic = bic
            locked_cols.append(new_col)
            locked_strs.append(new_str)
            print(f"-> ACCEPTED! Locked dictionary: {locked_strs}")
        else:
            print("-> REJECTED! BIC did not improve significantly.")
            break

    print(f"\n--- FINAL DISCOVERED EQUATION FOR {name} ---")
    mse, coefs = evaluate_locked_dictionary(locked_cols, data["y_dot"])
    
    eq_parts = [f"{c:.4f} * {s}" for c, s in zip(np.ravel(coefs), locked_strs) if abs(c) > 1e-5]
    print(f"Formula: {' + '.join(eq_parts) if eq_parts else 'dx/dt = 0'}")
    print(f"Final MSE: {mse:.4e} | Final BIC: {best_bic:.2f}\n")

if __name__ == "__main__":
    config = {
        "n_episodes": 1000, "n_simulations": 20, "t_max": 15, "c": 0.5,
        "gamma": 0.005, "alpha": 0.5, "noise": 0.0, "n_points": 500
    }
    
    run_benchmark("1. Linear Baseline (Damped Harmonic Oscillator)", damped_harmonic_oscillator_rk45, config)
    run_benchmark("2. Chaotic System (Lorenz Attractor Z-axis)", lorenz_attractor_rk45, config)
    run_benchmark("3. Deep Nesting Bound (Very Complex Three Var)", deep_nested_rk45, config)
