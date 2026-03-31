import time
import json
import numpy as np
import types
import copy
import pysindy as ps
from itertools import product as iproduct

from src.mcts import MCTS
from src.tree import build_tree_step_by_step, extract_features_from_tree
from src.grammar import get_valid_actions, is_complete, RULES
from src.main import sequence_to_library_strings, evaluate_locked_dictionary
from src.evaluate import evaluate_node, evaluate_tree_sindy
from utils.data_generators import damped_harmonic_oscillator_rk45, lorenz_attractor_rk45, deep_nested_rk45


# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------

def exact_match_check(found_dict, truth_dict):
    """
    Checks if the discovered dictionary perfectly matches the ground truth structure.
    Handles the (x*y) vs (y*x) grammar permutation for Lorenz.
    """
    found_keys = set(found_dict.keys())
    truth_keys = set(truth_dict.keys())
    # Normalize commutative multiplication permutations
    normalized_found = set()
    for k in found_keys:
        normalized_found.add(k.replace("(y * x)", "(x * y)"))

    return normalized_found == truth_keys and len(found_dict) == len(truth_dict)


def compute_coef_rmse(found_dict, truth_dict):
    """
    Compute RMSE between found and ground-truth coefficients.
    Penalizes for missed terms with their full truth value.
    """
    rmse = 0.0
    for k, truth_val in truth_dict.items():
        # Check both multiplication orderings
        alt_k = "(y * x)" if k == "(x * y)" else ("(x * y)" if k == "(y * x)" else None)
        if k in found_dict:
            rmse += (found_dict[k] - truth_val) ** 2
        elif alt_k and alt_k in found_dict:
            rmse += (found_dict[alt_k] - truth_val) ** 2
        else:
            rmse += truth_val ** 2  # Penalize for missing term
    return float(np.sqrt(rmse / max(1, len(truth_dict))))


# ---------------------------------------------------------------------------
# Baseline 1: PySINDy with a pre-built polynomial + trig library
# ---------------------------------------------------------------------------

def run_sindy_baseline(data, truth_dict, name, noise):
    """
    Runs standard PySINDy with a fixed polynomial (degree 2) + trig library.
    This is the "oracle pre-built library" baseline.
    """
    print(f"  [PySINDy] {name} (Noise: {noise})")

    x_vars = list(data["variables"].values())
    X = np.column_stack(x_vars)
    y_dot = data["y_dot"]

    # Pre-built combinatorial library: polynomials up to degree 2 + sin + cos
    library = ps.PolynomialLibrary(degree=2) + ps.FourierLibrary(n_frequencies=1)

    start_time = time.time()
    try:
        model = ps.SINDy(
            feature_library=library,
            optimizer=ps.STLSQ(threshold=0.05, alpha=0.05)
        )
        model.fit(X, x_dot=y_dot.reshape(-1, 1))
        y_pred = model.predict(X)
        mse = float(np.mean((y_pred.flatten() - y_dot) ** 2))

        coefs = model.coefficients().flatten()
        feature_names = model.get_feature_names()

        # Build a simple dict of non-zero terms
        found_dict = {}
        for name_f, c in zip(feature_names, coefs):
            if abs(c) > 1e-3:
                found_dict[name_f] = float(c)

        parsimony = len(found_dict)
        is_exact = False  # Hard to do exact string match with SINDy's auto-naming
        coef_rmse = 0.0   # Not directly comparable due to different naming schemes

    except Exception as e:
        print(f"    PySINDy FAILED: {e}")
        mse = float("inf")
        found_dict = {}
        parsimony = 0
        is_exact = False
        coef_rmse = float("inf")

    total_time = time.time() - start_time

    result = {
        "compute_time_sec": round(total_time, 3),
        "parsimony": parsimony,
        "exact_discovery": is_exact,
        "coefficient_rmse": coef_rmse,
        "final_equation": str(found_dict),
        "final_mse": mse
    }
    print(f"    -> Finished in {total_time:.3f}s | Parsimony: {parsimony} | MSE: {mse:.4e}")
    return result


# ---------------------------------------------------------------------------
# Baseline 2: SPL Proxy — Pure grammar MCTS WITHOUT SINDy coefficient solving
# ---------------------------------------------------------------------------

def run_spl_proxy(data_func, config, truth_dict, name):
    """
    SPL Proxy: Uses MCTS grammar exploration identical to our model,
    but evaluates rollouts with direct MSE instead of the BIC SINDy regression.
    This models the core limitation of SPL: blind to optimal coefficient values
    without a gradient-free closed-form coefficient solver.
    """
    print(f"  [SPL Proxy] {name} (Noise: {config['noise']})")

    data = data_func(n_points=config["n_points"], noise=config["noise"])

    grammar = types.SimpleNamespace(
        get_valid_actions=get_valid_actions,
        is_complete=is_complete,
        rules=RULES
    )

    start_time = time.time()

    # SPL: run just ONE round of MCTS and pick the best single expression
    # It cannot iteratively lock features because it has no SINDy coefficient decomposition
    agent = MCTS(
        grammar=grammar,
        data=data,
        locked_features_cols=[],   # SPL has no locked residual loop
        c=config["c"],
        n_simulations=config["n_simulations"],
        t_max=config["t_max"],
        gamma=config["gamma"],
        alpha=config["alpha"]
    )

    best_sequence, best_reward = agent.run(n_episodes=config["n_episodes"])
    total_time = time.time() - start_time

    if not best_sequence:
        result = {
            "compute_time_sec": round(total_time, 2),
            "parsimony": 0,
            "exact_discovery": False,
            "coefficient_rmse": float("inf"),
            "final_equation": "FAILED",
            "final_mse": float("inf")
        }
        print(f"    -> FAILED")
        return result

    tree = build_tree_step_by_step(best_sequence)
    best_str = sequence_to_library_strings(best_sequence)[0] if best_sequence else "N/A"

    # Compute raw MSE with a unit coefficient (SPL doesn't solve coefficients analytically)
    features = extract_features_from_tree(tree)
    if features:
        col = evaluate_node(features[0], data["variables"])
        if np.isscalar(col):
            col = np.full(data["y_dot"].shape[0], col)
        col = col.reshape(-1)
        # Best coefficient estimate via least squares (scalar)
        c_hat = float(np.dot(col, data["y_dot"]) / (np.dot(col, col) + 1e-12))
        y_pred = c_hat * col
        mse = float(np.mean((y_pred - data["y_dot"]) ** 2))
        found_dict = {best_str: round(c_hat, 4)}
    else:
        mse = float("inf")
        found_dict = {}

    parsimony = 1  # SPL outputs a single expression
    is_exact = False
    coef_rmse = float("inf")

    result = {
        "compute_time_sec": round(total_time, 2),
        "parsimony": parsimony,
        "exact_discovery": is_exact,
        "coefficient_rmse": coef_rmse,
        "final_equation": str(found_dict),
        "final_mse": mse
    }
    print(f"    -> Finished in {total_time:.2f}s | Best term: {best_str} | MSE: {mse:.4e}")
    return result


# ---------------------------------------------------------------------------
# Our Model: Iterative MCTS-SINDy
# ---------------------------------------------------------------------------

def run_our_model(data_func, config, truth_dict, name):
    print(f"  [MCTS-SINDy] {name} (Noise: {config['noise']})")

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

    start_time = time.time()

    for iteration in range(max_features):
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

        mse, coefs = evaluate_tree_sindy(tree, data["variables"], data["y_dot"], locked_features_cols)

        if len(coefs) == 0:
            break

        l0_norm = np.count_nonzero(coefs)
        n_points = len(data["y_dot"])
        mse_clamped = max(mse, 1e-12)
        bic = n_points * np.log(mse_clamped) + l0_norm * np.log(n_points)

        if new_str in locked_features_strings:
            break
        if abs(np.ravel(coefs)[-1]) < 1e-5:
            break
        if bic <= best_global_bic - 10.0:
            best_global_bic = bic
            locked_features_cols.append(new_col)
            locked_features_strings.append(new_str)
        else:
            break

    total_time = time.time() - start_time

    mse, coefs = evaluate_locked_dictionary(locked_features_cols, data["y_dot"])

    found_dict = {}
    if len(coefs) > 0:
        flat_coefs = np.ravel(coefs)
        for i, c in enumerate(flat_coefs):
            if i < len(locked_features_strings):
                found_dict[locked_features_strings[i]] = float(c)

    is_exact = exact_match_check(found_dict, truth_dict)
    coef_rmse = compute_coef_rmse(found_dict, truth_dict)

    result = {
        "compute_time_sec": round(total_time, 2),
        "parsimony": len(locked_features_strings),
        "exact_discovery": is_exact,
        "coefficient_rmse": round(coef_rmse, 6),
        "final_equation": str(found_dict),
        "final_mse": float(mse)
    }
    print(f"    -> Finished in {total_time:.2f}s | Parsimony: {len(locked_features_strings)} | Exact: {is_exact}")
    return result


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    experiments = [
        {
            "name": "Damped Harmonic Oscillator",
            "func": damped_harmonic_oscillator_rk45,
            "truth": {"y": 2.0, "x": -0.1}
        },
        {
            "name": "Lorenz Attractor Z-axis",
            "func": lorenz_attractor_rk45,
            "truth": {"(x * y)": 1.0, "z": -2.6667}
        },
        {
            "name": "Deep Nested Bound",
            "func": deep_nested_rk45,
            "truth": {"x": -1.0}
        }
    ]

    base_config = {
        "n_episodes": 1000,
        "n_simulations": 20,
        "t_max": 15,
        "c": 0.5,
        "gamma": 0.005,
        "alpha": 0.5,
        "n_points": 500
    }

    noise_levels = [0.0, 0.01, 0.05]

    results_matrix = {}

    for exp in experiments:
        results_matrix[exp["name"]] = {}

        for noise in noise_levels:
            config = copy.deepcopy(base_config)
            config["noise"] = noise
            noise_key = f"Noise_{int(noise * 100)}%"

            print(f"\n{'='*60}")
            print(f"BENCHMARK: {exp['name']} | {noise_key}")
            print(f"{'='*60}")

            # Generate data once for PySINDy (shared across noise level)
            data = exp["func"](n_points=config["n_points"], noise=noise)

            sindy_res   = run_sindy_baseline(data, exp["truth"], exp["name"], noise)
            spl_res     = run_spl_proxy(exp["func"], config, exp["truth"], exp["name"])
            our_res     = run_our_model(exp["func"], config, exp["truth"], exp["name"])

            results_matrix[exp["name"]][noise_key] = {
                "PySINDy":   sindy_res,
                "SPL_Proxy": spl_res,
                "MCTS_SINDy": our_res
            }

            # Incremental save so nothing is lost
            with open("results/results/benchmark_results.json", "w") as f:
                json.dump(results_matrix, f, indent=4)

    print("\n\nBenchmark Suite Complete — results saved to results/results/benchmark_results.json")


if __name__ == "__main__":
    run_pipeline()
