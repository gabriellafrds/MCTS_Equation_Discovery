import time
import json
import numpy as np
import types
import copy
import pysindy as ps

from src.mcts import MCTS
from src.tree import build_tree_step_by_step, extract_features_from_tree
from src.grammar import get_valid_actions, is_complete, RULES
from src.main import sequence_to_library_strings, evaluate_locked_dictionary
from src.evaluate import evaluate_node, evaluate_tree_sindy
from utils.data_generators import damped_harmonic_oscillator_rk45, lorenz_attractor_rk45, deep_nested_rk45
from utils.metrics import exact_match_check, compute_coef_rmse

def run_sindy_baseline(data, truth_dict, name, noise):
    """
    Standard PySINDy with a fixed degree 2 polynomial + sine/cosine library.
    """
    print(f"  [PySINDy] {name} (Noise: {noise})")
    X = np.column_stack(list(data["variables"].values()))
    y_dot = data["y_dot"]
    library = ps.PolynomialLibrary(degree=2) + ps.FourierLibrary(n_frequencies=1)

    start_time = time.time()
    try:
        model = ps.SINDy(feature_library=library, optimizer=ps.STLSQ(threshold=0.05, alpha=0.05))
        model.fit(X, x_dot=y_dot.reshape(-1, 1))
        y_pred = model.predict(X)
        mse = float(np.mean((y_pred.flatten() - y_dot) ** 2))

        # Filter out negligible terms
        found_dict = {f: float(c) for f, c in zip(model.get_feature_names(), model.coefficients().flatten()) if abs(c) > 1e-3}
        parsimony = len(found_dict)
        is_exact = False
        coef_rmse = 0.0

    except BaseException as e:
        print(f"    PySINDy FAILED: {e}")
        mse = coef_rmse = float("inf")
        found_dict = {}
        parsimony = 0
        is_exact = False

    total_time = time.time() - start_time
    print(f"    -> Finished in {total_time:.3f}s | Parsimony: {parsimony} | MSE: {mse:.4e}")
    
    return {
        "compute_time_sec": round(total_time, 3), "parsimony": parsimony,
        "exact_discovery": is_exact, "coefficient_rmse": coef_rmse,
        "final_equation": str(found_dict), "final_mse": mse
    }

def run_spl_proxy(data_func, config, truth_dict, name):
    """
    SPL Proxy: Pure MCTS exploration evaluated via direct L2 projection (No SINDy loop).
    """
    print(f"  [SPL Proxy] {name} (Noise: {config['noise']})")
    data = data_func(n_points=config["n_points"], noise=config["noise"])
    grammar = types.SimpleNamespace(get_valid_actions=get_valid_actions, is_complete=is_complete, rules=RULES)

    start_time = time.time()
    agent = MCTS(grammar, data, locked_features_cols=[], **{k: config[k] for k in ["c", "n_simulations", "t_max", "gamma", "alpha"]})
    best_sequence, _ = agent.run(n_episodes=config["n_episodes"])
    total_time = time.time() - start_time

    if not best_sequence:
        print(f"    -> FAILED")
        return {"compute_time_sec": round(total_time, 2), "parsimony": 0, "exact_discovery": False, "coefficient_rmse": float("inf"), "final_equation": "FAILED", "final_mse": float("inf")}

    tree = build_tree_step_by_step(best_sequence)
    best_str = sequence_to_library_strings(best_sequence)[0]
    features = extract_features_from_tree(tree)
    
    if features:
        col = evaluate_node(features[0], data["variables"])
        col = np.full(data["y_dot"].shape[0], col).reshape(-1) if np.isscalar(col) else col.reshape(-1)
        
        c_hat = float(np.dot(col, data["y_dot"]) / (np.dot(col, col) + 1e-12))
        mse = float(np.mean((c_hat * col - data["y_dot"]) ** 2))
        found_dict = {best_str: round(c_hat, 4)}
    else:
        mse = float("inf")
        found_dict = {}

    print(f"    -> Finished in {total_time:.2f}s | Best term: {best_str} | MSE: {mse:.4e}")
    return {
        "compute_time_sec": round(total_time, 2), "parsimony": 1,
        "exact_discovery": False, "coefficient_rmse": float("inf"),
        "final_equation": str(found_dict), "final_mse": mse
    }

def run_our_model(data_func, config, truth_dict, name):
    """
    MCTS-SINDy iteratively locking basis functions with Occam's Razor rejection.
    """
    print(f"  [MCTS-SINDy] {name} (Noise: {config['noise']})")
    data = data_func(n_points=config["n_points"], noise=config["noise"])
    grammar = types.SimpleNamespace(get_valid_actions=get_valid_actions, is_complete=is_complete, rules=RULES)

    locked_cols, locked_strs = [], []
    best_bic = float('inf')

    start_time = time.time()
    for _ in range(10):  # max_features
        agent = MCTS(grammar, data, locked_cols, **{k: config[k] for k in ["c", "n_simulations", "t_max", "gamma", "alpha"]})
        best_sequence, _ = agent.run(n_episodes=config["n_episodes"])
        
        if not best_sequence: break

        tree = build_tree_step_by_step(best_sequence)
        features = extract_features_from_tree(tree)
        if not features: continue

        new_str = sequence_to_library_strings(best_sequence)[0]
        new_col = evaluate_node(features[0], data["variables"])
        new_col = np.full(data["y_dot"].shape[0], new_col).reshape(-1) if np.isscalar(new_col) else new_col.reshape(-1)

        mse, coefs = evaluate_tree_sindy(tree, data["variables"], data["y_dot"], locked_cols)
        if len(coefs) == 0: break

        n_points = len(data["y_dot"])
        bic = n_points * np.log(max(mse, 1e-12)) + np.count_nonzero(coefs) * np.log(n_points)

        if new_str in locked_strs or abs(np.ravel(coefs)[-1]) < 1e-5 or bic > best_bic - 10.0:
            break

        best_bic = bic
        locked_cols.append(new_col)
        locked_strs.append(new_str)

    total_time = time.time() - start_time
    mse, coefs = evaluate_locked_dictionary(locked_cols, data["y_dot"])
    
    found_dict = {s: float(c) for s, c in zip(locked_strs, np.ravel(coefs))} if len(coefs) > 0 else {}
    is_exact = exact_match_check(found_dict, truth_dict)
    
    print(f"    -> Finished in {total_time:.2f}s | Parsimony: {len(locked_strs)} | Exact: {is_exact}")
    return {
        "compute_time_sec": round(total_time, 2), "parsimony": len(locked_strs),
        "exact_discovery": is_exact, "coefficient_rmse": round(compute_coef_rmse(found_dict, truth_dict), 6),
        "final_equation": str(found_dict), "final_mse": float(mse)
    }

def run_pipeline():
    experiments = [
        {"name": "Damped Harmonic Oscillator", "func": damped_harmonic_oscillator_rk45, "truth": {"y": 2.0, "x": -0.1}},
        {"name": "Lorenz Attractor Z-axis", "func": lorenz_attractor_rk45, "truth": {"(x * y)": 1.0, "z": -2.6667}},
        {"name": "Deep Nested Bound", "func": deep_nested_rk45, "truth": {"x": -1.0}}
    ]

    base_config = {"n_episodes": 1000, "n_simulations": 20, "t_max": 15, "c": 0.5, "gamma": 0.005, "alpha": 0.5, "n_points": 500}
    noise_levels = [0.0, 0.01, 0.05]
    results_matrix = {exp["name"]: {} for exp in experiments}

    for exp in experiments:
        for noise in noise_levels:
            config = {**base_config, "noise": noise}
            noise_key = f"Noise_{int(noise * 100)}%"

            print(f"\n{'='*60}\nBENCHMARK: {exp['name']} | {noise_key}\n{'='*60}")
            data = exp["func"](n_points=config["n_points"], noise=noise)

            results_matrix[exp["name"]][noise_key] = {
                "PySINDy": run_sindy_baseline(data, exp["truth"], exp["name"], noise),
                "SPL_Proxy": run_spl_proxy(exp["func"], config, exp["truth"], exp["name"]),
                "MCTS_SINDy": run_our_model(exp["func"], config, exp["truth"], exp["name"])
            }

            with open("results/benchmark_results.json", "w") as f:
                json.dump(results_matrix, f, indent=4)

    print("\n\nBenchmark Suite Complete — results saved to results/benchmark_results.json")

if __name__ == "__main__":
    run_pipeline()
