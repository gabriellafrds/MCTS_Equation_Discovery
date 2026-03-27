# spl_sindy_v2/data_generators.py
#
# Data generation functions for benchmarks used in the paper
# and in our experiments.
#
# All functions return a dict with:
#   "variables" : dict mapping variable names to numpy arrays
#   "y_dot"     : target derivative array

import numpy as np


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
