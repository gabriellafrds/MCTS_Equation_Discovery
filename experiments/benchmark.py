import csv
import itertools
import copy
from src.main import main as run_main  # Assure-toi que main.py a la fonction main paramétrable
from src.main import print_final_equation, sequence_to_library_strings, evaluate_locked_dictionary, complex_three_var, MCTS
from src.tree import build_tree_step_by_step, extract_features_from_tree
from src.grammar import get_valid_actions, is_complete, RULES
import types
import numpy as np


# ---------------------------------------------------------------------------
# Wrapper de main pour accepter une configuration
# ---------------------------------------------------------------------------
def run_sindy_experiment(config):
    """
    Run one experiment with the given config.
    Returns a dict with 'config', 'locked_features', 'mse', 'bic'
    """
    # Génération des données
    data = complex_three_var(n_points=config["n_points"], noise=config["noise"])

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
        agent = MCTS(
            grammar=grammar,
            data=data,
            locked_features_cols=locked_features_cols,
            c=config["c"],
            n_simulations=config["n_simulations"],
            t_max=config["t_max"],
            gamma=config["gamma"],
            alpha=config["alpha"],
            beta=config["beta"]
        )

        best_sequence, best_reward = agent.run(n_episodes=config["n_episodes"])
        if not best_sequence:
            break

        tree = build_tree_step_by_step(best_sequence)
        features = extract_features_from_tree(tree)
        if not features:
            continue

        new_str = sequence_to_library_strings(best_sequence)[0]
        from evaluate import evaluate_node
        new_col = evaluate_node(features[0], data["variables"])
        if new_col.ndim == 0:  # scalar case
            new_col = new_col * np.ones(data["y_dot"].shape[0])
        else:
            new_col = new_col.reshape(-1)

        # Re-evaluer SINDy
        from evaluate import evaluate_tree_sindy
        mse, coefs = evaluate_tree_sindy(tree, data["variables"], data["y_dot"], locked_features_cols)
        l0_norm = np.count_nonzero(coefs)
        n_points = len(data["y_dot"])
        mse_clamped = max(mse, 1e-12)
        bic = n_points * np.log(mse_clamped) + l0_norm * np.log(n_points)

        # Duplication
        if new_str in locked_features_strings:
            break

        # Occam's razor
        improvement_threshold = 10.0
        if bic <= best_global_bic - improvement_threshold or bic <= best_global_bic:
            best_global_bic = bic
            locked_features_cols.append(new_col)
            locked_features_strings.append(new_str)
        else:
            break

    # Évaluer la solution finale
    mse, coefs = evaluate_locked_dictionary(locked_features_cols, data["y_dot"])
    formula_parts = []
    if len(coefs) > 0:
        flat_coefs = np.ravel(coefs)
        for i, c in enumerate(flat_coefs):
            if abs(c) > 1e-5 and i < len(locked_features_strings):
                formula_parts.append(f"{c:.4f} * {locked_features_strings[i]}")
    final_formula = " + ".join(formula_parts) if formula_parts else "dx/dt = 0"

    # Print final equation
    print_final_equation(locked_features_strings, locked_features_cols, data["y_dot"])

    # Return results pour CSV
    return {
        "config" : config,
        "locked_features": copy.deepcopy(locked_features_strings),
        "bic": best_global_bic,
        "mse": mse,
        "formula": final_formula  # <-- formule finale SINDy lisible
    }


# ---------------------------------------------------------------------------
# Définition des hyperparamètres à tester
# ---------------------------------------------------------------------------
param_grid = {
    "n_episodes": [200, 500],
    "n_simulations": [20, 40],
    "t_max": [15, 20],
    "c": [0.5, 1.0],
    "gamma": [0.001, 0.005],
    "alpha": [0.5, 1.0],
    "beta": [0.0, 0.1],
    "noise": [0.0],
    "n_points": [500, 1000]
}

# ---------------------------------------------------------------------------
# Création de toutes les combinaisons
# ---------------------------------------------------------------------------
keys, values = zip(*param_grid.items())
configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

# ---------------------------------------------------------------------------
# Boucle sur toutes les combinaisons et sauvegarde CSV
# ---------------------------------------------------------------------------
output_file = "results/results/experiment_results_complex_three_var.csv"
with open(output_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["config", "locked_features", "mse", "bic", "formula"])
    writer.writeheader()

    for i, config in enumerate(configs):
        print(f"Running experiment {i + 1}/{len(configs)} with config: {config}")
        result = run_sindy_experiment(copy.deepcopy(config))
        # On stringify pour CSV
        result["config"] = str(result["config"])
        result["locked_features"] = str(result["locked_features"])
        writer.writerow(result)
        print(f"Result: MSE={result['mse']:.6e}, BIC={result['bic']:.2f}, Features={result['locked_features']}\n")

print(f"All experiments finished. Results saved to {output_file}")