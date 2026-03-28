# data_generators.py
#
# Data generation functions for all benchmarks.
# Used by bench.py and main.py.
#
# All functions return a dict with:
#   "variables" : dict mapping variable names to numpy arrays
#   "y_dot"     : target derivative array

import numpy as np


def simple_polynomial(n_points=50, noise=0.0):
    """
    Simple test: y_dot = x^2
    Expected: one feature x*x with coefficient 1.
    """
    x     = np.linspace(-2, 2, n_points)
    y_dot = x**2
    if noise > 0:
        y_dot += np.random.normal(0, noise, size=y_dot.shape)
    return {"variables": {"x": x}, "y_dot": y_dot}


def two_features(n_points=50, noise=0.0):
    """
    Two-feature test: y_dot = x^2 + sin(x)
    Expected: two features [x*x, sin(x)] with coefficients [1, 1].
    """
    x     = np.linspace(-2, 2, n_points)
    y_dot = x**2 + np.sin(x)
    if noise > 0:
        y_dot += np.random.normal(0, noise, size=y_dot.shape)
    return {"variables": {"x": x}, "y_dot": y_dot}


def nguyen1(n_points=50, x_range=(-2, 2), noise=0.0):
    """
    Nguyen-1: y_dot = x^3 + x^2 + x
    Single variable, polynomial target.
    Range [-2, 2] gives better discrimination than [-1, 1].
    """
    x     = np.linspace(x_range[0], x_range[1], n_points)
    y_dot = x**3 + x**2 + x
    if noise > 0:
        y_dot += np.random.normal(0, noise, size=y_dot.shape)
    return {"variables": {"x": x}, "y_dot": y_dot}


def lorenz_xdot(n_points=200, noise=0.0):
    """
    Simple Lorenz-inspired: x_dot = -y
    x = cos(t), y = sin(t) -> x_dot = -sin(t) = -y
    Used to test multi-variable discovery.
    True equation: y_dot = -y  (coefficient -1 on feature y)
    """
    t     = np.linspace(0, 10, n_points)
    x     = np.cos(t)
    y     = np.sin(t)
    x_dot = -y
    return {"variables": {"x": x, "y": y}, "y_dot": x_dot}


def nonlinear_three_var(n_points=100, noise=0.0):
    """
    Nonlinear three-variable benchmark:

        x_dot = sin(x^2 * y) / (1 + exp(-z))

    Variables: x, y, z sampled independently on [-1, 1].

    This benchmark tests the ability to discover a non-separable
    compound expression involving three variables. The true equation
    is a single term — not a linear combination of simple features —
    which makes it a stress test for the MCTS exploration.

    SINDy will represent this as ONE column in Theta with coefficient
    ~1.0 if MCTS discovers the exact structure. Otherwise, SINDy
    will try to approximate it with a linear combination of simpler
    discovered features.
    """
    rng   = np.random.default_rng(seed=42)   # fixed seed for reproducibility
    x     = rng.uniform(-1, 1, n_points)
    y     = rng.uniform(-1, 1, n_points)
    z     = rng.uniform(-1, 1, n_points)

    y_dot = np.sin(x**2 * y) / (1.0 + np.exp(-z))

    if noise > 0:
        y_dot += np.random.normal(0, noise, size=y_dot.shape)

    return {
        "variables": {"x": x, "y": y, "z": z},
        "y_dot":     y_dot,
    }

def complex_three_var(n_points=100, noise=0.0):
    rng = np.random.default_rng(seed=42)
    x = rng.uniform(-1, 1, n_points)
    y = rng.uniform(-1, 1, n_points)
    z = rng.uniform(-1, 1, n_points)

    y_dot = np.sin(x**2 * y) + np.cos(y * z) + x

    if noise > 0:
        y_dot += np.random.normal(0, noise, size=y_dot.shape)

    return {"variables": {"x": x, "y": y, "z": z}, "y_dot": y_dot}