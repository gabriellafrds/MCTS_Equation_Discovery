# spl_sindy_v2/main_v2.py
#
# Main entry point for the SPL-SINDy v2 pipeline.
#
# Architecture: one independent MCTS run per basis function,
# then SINDy STLSQ to find sparse coefficients.
#
# Usage:
#   python spl_sindy_v2/main_v2.py
#
# The script runs three benchmarks in sequence:
#   1. Simple: y_dot = x^2
#   2. Two features: y_dot = x^2 + sin(x)
#   3. Nguyen-1: y_dot = x^3 + x^2 + x

import sys
import os
import numpy as np

# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spl_sindy_v2.grammar_v2 import get_grammar_for_variables
from spl_sindy_v2.mcts_feature import MCTSFeature
from spl_sindy_v2.sindy_v2 import run_sindy_pipeline
from spl_sindy_v2.data_generators import (
    nguyen1, lorenz_xdot, simple_polynomial, two_features
)


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------

def run_spl_sindy(data, variable_names, config):
    """
    Full SPL-SINDy v2 pipeline.

    Stage 1: Run config["n_features"] independent MCTS searches.
             Each search finds the best single basis function phi_i(x).
    Stage 2: Build library Theta = [phi_1 | phi_2 | ... | phi_k].
             Run STLSQ to find sparse coefficients xi.

    data           : dict with "variables" and "y_dot"
    variable_names : list of variable names in the system, e.g. ["x"]
    config         : hyperparameter dict

    Returns (xi, names, Theta, mse).
    """
    grammar = get_grammar_for_variables(variable_names)

    candidates = []   # list of (reward, sequence)

    print(f"\nStage 1: Searching for {config['n_features']} basis functions...")
    print(f"  MCTS config: {config['n_episodes']} episodes, "
          f"{config['n_simulations']} simulations, "
          f"t_max={config['t_max']}, eta={config['eta']}")

    for i in range(config["n_features"]):
        # Fresh MCTS agent for each feature
        agent = MCTSFeature(
            grammar       = grammar,
            data          = data,
            c             = config["c"],
            n_simulations = config["n_simulations"],
            t_max         = config["t_max"],
            eta           = config["eta"],
        )

        best_seq, best_r = agent.run(n_episodes=config["n_episodes"])

        if best_seq is not None:
            candidates.append((best_r, best_seq))
            parts = [" ".join(r) for (l, r) in best_seq]
            print(f"  Feature {i+1:2d}/{config['n_features']}: "
                  f"r={best_r:.4f}  {'  |  '.join(parts)}")
        else:
            print(f"  Feature {i+1:2d}/{config['n_features']}: "
                  f"no valid expression found")

    if not candidates:
        print("No candidates found — cannot run SINDy.")
        return None, [], None, None

    print(f"\nStage 2: SINDy sparse regression "
          f"(lambda={config['lambda_threshold']})...")
    xi, names, Theta, mse = run_sindy_pipeline(
        candidates,
        data,
        lambda_threshold=config["lambda_threshold"],
    )

    return xi, names, Theta, mse


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def benchmark_simple(config):
    """
    Benchmark 1: y_dot = x^2
    Expected result: one feature x*x with coefficient ~1.0
    """
    print("\n" + "="*65)
    print("BENCHMARK 1: y_dot = x^2")
    print("="*65)
    data = simple_polynomial(n_points=config["n_points"])
    return run_spl_sindy(data, ["x"], config)


def benchmark_two_features(config):
    """
    Benchmark 2: y_dot = x^2 + sin(x)
    Expected result: features [x*x, sin(x)] with coefficients [1, 1].
    """
    print("\n" + "="*65)
    print("BENCHMARK 2: y_dot = x^2 + sin(x)")
    print("="*65)
    data = two_features(n_points=config["n_points"])
    return run_spl_sindy(data, ["x"], config)


def benchmark_nguyen1(config):
    """
    Benchmark 3: y_dot = x^3 + x^2 + x  (Nguyen-1)
    Expected: features [x*x*x, x*x, x] with coefficients [1, 1, 1].
    """
    print("\n" + "="*65)
    print("BENCHMARK 3: y_dot = x^3 + x^2 + x  (Nguyen-1)")
    print("="*65)
    data = nguyen1(n_points=config["n_points"])
    return run_spl_sindy(data, ["x"], config)


def benchmark_lorenz(config):
    """
    Benchmark 4: x_dot = -y  (Lorenz-inspired, multi-variable)
    Expected: one feature y with coefficient -1.
    """
    print("\n" + "="*65)
    print("BENCHMARK 4: x_dot = -y  (multi-variable)")
    print("="*65)
    data = lorenz_xdot(n_points=config["n_points"])
    return run_spl_sindy(data, ["x", "y"], config)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Hyperparameters
    # Start conservative — increase n_episodes for better results
    config = {
        # MCTS parameters
        "n_features"      : 5,       # number of basis functions to discover
        "n_episodes"      : 2000,    # MCTS episodes per feature
        "n_simulations"   : 10,      # rollouts per expansion
        "t_max"           : 10,      # max sequence length per feature
        "c"               : 1.0,     # UCT exploration constant
        "eta"             : 0.99,    # parsimony discount (strong)
        # Data parameters
        "n_points"        : 50,
        # SINDy parameters
        "lambda_threshold": 0.05,    # STLSQ sparsity threshold
    }

    print("SPL-SINDy v2 — Per-Feature MCTS + SINDy Pipeline")
    print("="*65)
    print("Config:")
    for k, v in config.items():
        print(f"  {k:<20} : {v}")

    # Run benchmarks from simplest to hardest
    benchmark_simple(config)
    benchmark_two_features(config)
    benchmark_nguyen1(config)
    benchmark_lorenz(config)


if __name__ == "__main__":
    main()
