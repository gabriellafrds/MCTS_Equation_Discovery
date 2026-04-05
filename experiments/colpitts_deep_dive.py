import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import types
import time
import json
from utils.metrics import compute_coef_rmse

from src.mcts import MCTS
from src.grammar import get_valid_actions, is_complete, RULES
from src.tree import build_tree_step_by_step, extract_features_from_tree
from src.main import sequence_to_library_strings, evaluate_locked_dictionary
from src.evaluate import evaluate_node, evaluate_tree_sindy
from utils.data_generators import colpitts_oscillator_rk45

def discover_dimension(target, config):
    print(f"\n{'='*50}\nDiscovering exact equation for {target}_dot\n{'='*50}")
    
    # Generate targeted data
    data = colpitts_oscillator_rk45(n_points=config["n_points"], target=target, noise=config["noise"])
    grammar = types.SimpleNamespace(get_valid_actions=get_valid_actions, is_complete=is_complete, rules=RULES)

    locked_cols, locked_strs = [], []
    best_bic = float('inf')
    search_history = []

    for iteration in range(5):  # Max 5 terms expected per dimension
        agent = MCTS(grammar, data, locked_cols, **{k: config[k] for k in ["c", "n_simulations", "t_max", "gamma", "alpha"]})
        best_seq, _ = agent.run(config["n_episodes"])
        
        if not best_seq: break

        tree = build_tree_step_by_step(best_seq)
        features = extract_features_from_tree(tree)
        if not features: continue

        new_str = sequence_to_library_strings(best_seq)[0]
        new_col = evaluate_node(features[0], data["variables"])
        new_col = np.full(data["y_dot"].shape[0], new_col).reshape(-1) if np.isscalar(new_col) else new_col.reshape(-1)

        mse, coefs = evaluate_tree_sindy(tree, data["variables"], data["y_dot"], locked_cols)
        if len(coefs) == 0: break

        n_points = len(data["y_dot"])
        bic = n_points * np.log(max(mse, 1e-12)) + np.count_nonzero(coefs) * np.log(n_points)

        print(f"[{target}_dot Iteration {iteration+1}] Proposed: {new_str} | BIC: {bic:.2f} | MSE: {mse:.4e}")

        # Occam's Razor Heuristics
        if new_str in locked_strs or abs(np.ravel(coefs)[-1]) < 1e-5:
            print(f"  -> Rejected. (Duplicate or Zero Coef)")
            search_history.append({"iteration": iteration + 1, "proposed_feature": new_str, "bic": float(bic), "mse": float(mse), "accepted": False, "reason": "Duplicate or Zero Coef"})
            break

        if bic <= best_bic - 2.0:
            best_bic = bic
            locked_cols.append(new_col)
            locked_strs.append(new_str)
            print(f"  -> ACCEPTED. Current Dictionary: {locked_strs}")
            search_history.append({"iteration": iteration + 1, "proposed_feature": new_str, "bic": float(bic), "mse": float(mse), "accepted": True, "reason": "BIC Improvement"})
        else:
            print(f"  -> Terminating (Occam's razor active).")
            search_history.append({"iteration": iteration + 1, "proposed_feature": new_str, "bic": float(bic), "mse": float(mse), "accepted": False, "reason": "Occam's Razor"})
            break

    mse, coefs = evaluate_locked_dictionary(locked_cols, data["y_dot"])
    found_dict = {s: float(c) for s, c in zip(locked_strs, np.ravel(coefs))} if len(coefs) > 0 else {}
    print(f"\nFinal {target}_dot Equation: {found_dict}\n")
    return found_dict, search_history

def simulate_and_plot(found_x, found_y, found_z):
    print("\nGenerating Phase Space Comparison Plot...")
    n_points = 2000
    t_eval = np.linspace(0, 50, n_points)
    
    # Ground Truth Dynamics
    alpha, beta, gamma, delta = 5.0, -1.0, 1.0, 1.0
    def truth_deriv(t, state):
        x, y, z = state
        return [
            alpha * y,
            -gamma * (x + z) - delta * y,
            beta * (y - (1.0 - np.exp(-x)))
        ]
        
    sol_truth = solve_ivp(truth_deriv, [0, 50], [2.0, 2.0, 2.0], t_eval=t_eval, method='RK45')
    
    # Build a fast mapping for the discovered dynamics
    def parse_term(term_str, x, y, z):
        # We know our grammar limits. This handles the raw string evaluation quickly.
        # Safe eval environment:
        env = {"x": x, "y": y, "z": z, "sin": np.sin, "cos": np.cos, "exp": np.exp, "1": 1.0}
        # Clean grammar notation: (x * y) -> (x*y) 
        clean_str = term_str.replace(" ", "")
        return eval(clean_str, {"__builtins__": None}, env)

    def pred_deriv(t, state):
        x, y, z = state
        dx = sum(coef * parse_term(term, x, y, z) for term, coef in found_x.items())
        dy = sum(coef * parse_term(term, x, y, z) for term, coef in found_y.items())
        dz = sum(coef * parse_term(term, x, y, z) for term, coef in found_z.items())
        return [dx, dy, dz]

    sol_pred = solve_ivp(pred_deriv, [0, 50], [1.0, 1.0, 1.0], t_eval=t_eval, method='RK45')

    # Plot
    fig = plt.figure(figsize=(16, 8))
    
    # Subplot 1: Truth
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(sol_truth.y[0], sol_truth.y[1], sol_truth.y[2], c=sol_truth.t, cmap='viridis', s=2)
    ax1.set_title("Ground Truth (RK45)\nColpitts Chaos Oscillator", fontsize=14, pad=20)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    # Subplot 2: Discovery
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(sol_pred.y[0], sol_pred.y[1], sol_pred.y[2], c=sol_pred.t, cmap='viridis', s=2)
    ax2.set_title("MCTS-SINDy Reconstructed Dynamics", fontsize=14, pad=20)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    cbar = fig.colorbar(sc2, ax=[ax1, ax2], shrink=0.6, pad=0.1)
    cbar.set_label("Simulation Time (t)")
    
    os.makedirs("results", exist_ok=True)
    out_path = "results/colpitts_deep_dive.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

def main():
    config = {
        "n_episodes": 2000, 
        "n_simulations": 20, 
        "t_max": 15, 
        "c": 0.5,
        "gamma": 0.005, 
        "alpha": 0.5, 
        "noise": 0.01,  # 1% standard deviation noise added
        "n_points": 500
    }
    
    # Ground truth targets for metric computation
    truths = {
        'x': {'y': 5.0},
        'y': {'x': -1.0, 'z': -1.0, 'y': -1.0},
        'z': {'y': -1.0, '1': 1.0, 'exp(-(x))': -1.0}
    }
    metrics = {}

    for target in ['x', 'y', 'z']:
        start_t = time.time()
        found_dict, search_history = discover_dimension(target, config)
        exec_time = time.time() - start_t
        
        # Verify if MCTS managed to find exactly the True algebraic features
        # Allowing for commutative permutations like (z * x) vs (x * z)
        truth_keys = [set(k.replace(' ', '').split('*')) for k in truths[target].keys()]
        found_keys = [set(k.replace(' ', '').split('*')) for k in found_dict.keys()]
        
        is_exact = len(truth_keys) == len(found_keys) and all(tk in found_keys for tk in truth_keys)
        rmse = compute_coef_rmse(found_dict, truths[target]) if is_exact else None
        
        metrics[target] = {
            "discovered_equation": found_dict,
            "target_truth": truths[target],
            "parsimony": len(found_dict),
            "exact_match": is_exact,
            "coef_rmse": rmse,
            "time_seconds": round(exec_time, 2),
            "search_history": search_history
        }

    # Dump the metrics dictionary directly to JSON for user observation
    os.makedirs("results", exist_ok=True)
    with open("results/colpitts_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        print("Detailed tabular metrics saved to results/colpitts_metrics.json")

    # Automatically trigger plot generation using the discovered formulas
    simulate_and_plot(metrics['x']['discovered_equation'], metrics['y']['discovered_equation'], metrics['z']['discovered_equation'])

if __name__ == "__main__":
    main()
