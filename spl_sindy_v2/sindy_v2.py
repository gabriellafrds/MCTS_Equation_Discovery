# spl_sindy_v2/sindy_v2.py
#
# Builds the SINDy library matrix Theta from a list of candidate
# sequences (one per MCTS run) and runs STLSQ via pysindy.
#
# This is the second stage of the SPL-SINDy pipeline.
# The first stage (MCTSFeature) finds individual basis functions.
# This stage assembles them into a library and fits the sparse model.

import sys
import os
import numpy as np
import pysindy as ps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tree import build_tree_step_by_step
from spl_sindy_v2.evaluate_v2 import evaluate_node


def sequence_to_name(sequence):
    """
    Convert a sequence of production rules to a human-readable string.

    Strategy: collect all right-hand sides in order and join them.
    Example: [("f",["M"]), ("M",["M","*","M"]), ("M",["x"]), ("M",["x"])]
             -> "M | M * M | x | x"
    """
    parts = [" ".join(right) for (left, right) in sequence]
    return " | ".join(parts)


def build_theta(candidates, variables):
    """
    Build the SINDy library matrix Theta from candidate sequences.

    Each candidate is a (reward, sequence) tuple.
    Each sequence is evaluated as phi(x) and becomes one column of Theta.
    Columns that are non-finite, constant, or near-duplicate are discarded.

    candidates : list of (reward, sequence)
    variables  : dict {"x": array, "y": array, ...}

    Returns:
        Theta : numpy array (n_points x n_valid_columns)
        names : list of function name strings
        kept  : list of (reward, sequence) that survived filtering
    """
    columns  = []
    names    = []
    kept     = []

    for reward, seq in candidates:
        if seq is None:
            continue
        try:
            tree = build_tree_step_by_step(seq)
            phi  = evaluate_node(tree, variables)

            # Filter: non-finite
            if not np.all(np.isfinite(phi)):
                continue

            # Filter: constant (std too small)
            if np.std(phi) < 1e-10:
                continue

            # Filter: near-duplicate of existing column
            # Check correlation with all existing columns
            phi_norm = phi / (np.linalg.norm(phi) + 1e-10)
            duplicate = False
            for existing_col in columns:
                existing_norm = existing_col / (np.linalg.norm(existing_col) + 1e-10)
                corr = abs(np.dot(phi_norm, existing_norm))
                if corr > 0.999:   # nearly identical function
                    duplicate = True
                    break

            if duplicate:
                continue

            columns.append(phi)
            names.append(sequence_to_name(seq))
            kept.append((reward, seq))

        except Exception:
            continue

    if not columns:
        return None, [], []

    Theta = np.column_stack(columns)
    return Theta, names, kept


def run_stlsq(Theta, y_dot, lambda_threshold=0.1, max_iter=20):
    """
    Run STLSQ sparse regression via pysindy.

    Theta          : library matrix (n_samples x n_features)
    y_dot          : target derivative (n_samples,)
    lambda_threshold: sparsity threshold
    max_iter       : maximum STLSQ iterations

    Returns xi : sparse coefficient vector (n_features,)
    """
    optimizer = ps.STLSQ(threshold=lambda_threshold, max_iter=max_iter)
    optimizer.fit(Theta, y_dot.reshape(-1, 1))
    return optimizer.coef_.flatten()


def print_library(names, rewards):
    """Print the library functions with their MCTS rewards."""
    print(f"\nLibrary ({len(names)} functions):")
    for i, (name, reward) in enumerate(zip(names, rewards)):
        print(f"  [{i:2d}] r={reward:.4f}  {name}")


def print_equation(xi, names, variable="y_dot"):
    """Print the identified sparse equation."""
    print(f"\nIdentified equation:")
    terms = [(c, n) for c, n in zip(xi, names) if abs(c) > 1e-6]
    if terms:
        expr = " + ".join(f"({c:+.6f}) * [{n}]" for c, n in terms)
        print(f"  {variable} = {expr}")
    else:
        print(f"  {variable} = 0")
        print("  [No terms survived — try reducing lambda_threshold]")


def run_sindy_pipeline(candidates, data, lambda_threshold=0.1):
    """
    Full second-stage pipeline: build Theta, run STLSQ, report results.

    candidates       : list of (reward, sequence) from MCTSFeature runs
    data             : dict with "variables" and "y_dot"
    lambda_threshold : STLSQ sparsity threshold

    Returns:
        xi    : sparse coefficient vector
        names : function name strings
        Theta : library matrix
        mse   : mean squared error of identified model
    """
    variables = data["variables"]
    y_dot     = data["y_dot"]

    # Build library
    Theta, names, kept = build_theta(candidates, variables)

    if Theta is None:
        print("Library is empty — no valid candidates.")
        return None, [], None, None

    rewards = [r for (r, _) in kept]
    print_library(names, rewards)
    print(f"\nTheta shape: {Theta.shape[0]} points x {Theta.shape[1]} features")

    # Run STLSQ
    xi = run_stlsq(Theta, y_dot, lambda_threshold=lambda_threshold)

    # Print equation
    print_equation(xi, names)

    # Compute final MSE
    y_pred = Theta @ xi
    mse    = float(np.mean((y_pred - y_dot) ** 2))
    print(f"  MSE = {mse:.8f}")

    return xi, names, Theta, mse
