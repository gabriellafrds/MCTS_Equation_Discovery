# data_generators.py
#
# Data generation functions for all benchmarks.
# Used by bench.py and main.py.
#
# All functions return a dict with:
#   "variables" : dict mapping variable names to numpy arrays
#   "y_dot"     : target derivative array

import numpy as np
from scipy.integrate import solve_ivp


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

def very_complex_three_var(n_points=100, noise=0.0):
    rng = np.random.default_rng(seed=42)
    x = rng.uniform(-1, 1, n_points)
    y = rng.uniform(-1, 1, n_points)
    z = rng.uniform(-1, 1, n_points)

    # 1.0000 * x + 1.0000 * sin((((x * x) * z) * y)) + 1.0000 * cos(sin((z * y)))
    y_dot = x + np.sin((x**2) * z * y) + np.cos(np.sin(z * y))

    if noise > 0:
        y_dot += np.random.normal(0, noise, size=y_dot.shape)

    return {"variables": {"x": x, "y": y, "z": z}, "y_dot": y_dot}

# ---------------------------------------------------------------------------
# RK45 Benchmark Pipeline Generators
# ---------------------------------------------------------------------------

def damped_harmonic_oscillator_rk45(n_points=500, noise=0.0):
    """
    dx/dt = -0.1x + 2y
    dy/dt = -2x - 0.1y
    Target: dx/dt
    """
    def deriv(t, state):
        x, y = state
        return [-0.1*x + 2*y, -2*x - 0.1*y]
        
    t_eval = np.linspace(0, 10, n_points)
    sol = solve_ivp(deriv, [0, 10], [1.0, 0.0], t_eval=t_eval, method='RK45')
    
    x, y = sol.y
    x_dot = -0.1*x + 2*y
    
    if noise > 0:
        rng = np.random.default_rng(seed=42)
        x_dot += rng.normal(0, noise * np.std(x_dot), size=x_dot.shape)
        
    return {"variables": {"x": x, "y": y}, "y_dot": x_dot}


def lorenz_attractor_rk45(n_points=500, noise=0.0):
    """
    dx/dt = 10(y - x)
    dy/dt = x(28 - z) - y
    dz/dt = x*y - (8/3)z
    Target: dz/dt
    """
    def deriv(t, state):
        x, y, z = state
        return [10*(y - x), x*(28 - z) - y, x*y - (8/3)*z]
        
    t_eval = np.linspace(0, 10, n_points)
    sol = solve_ivp(deriv, [0, 10], [1.0, 1.0, 1.0], t_eval=t_eval, method='RK45')
    
    x, y, z = sol.y
    z_dot = x*y - (8/3)*z
    
    if noise > 0:
        rng = np.random.default_rng(seed=42)
        z_dot += rng.normal(0, noise * np.std(z_dot), size=z_dot.shape)
        
    return {"variables": {"x": x, "y": y, "z": z}, "y_dot": z_dot}


def deep_nested_rk45(n_points=500, noise=0.0):
    """
    To create a smooth physical state matrix, we define a chaotic base 
    and wrap it in the boundary equation:
    dy/dt = cos(t)  -> y = sin(t)
    dz/dt = -sin(t) -> z = cos(t)
    dx/dt = -x + sin(x^2 * y * z)
    Target: dx/dt
    """
    def deriv(t, state):
        x, y, z = state
        x_dot = -x + np.sin((x**2) * y * z)
        return [x_dot, np.cos(t), -np.sin(t)]
        
    t_eval = np.linspace(0, 10, n_points)
    sol = solve_ivp(deriv, [0, 10], [1.0, 0.0, 1.0], t_eval=t_eval, method='RK45')
    
    x, y, z = sol.y
    x_dot = -x + np.sin((x**2) * y * z)
    
    if noise > 0:
        rng = np.random.default_rng(seed=42)
        x_dot += rng.normal(0, noise * np.std(x_dot), size=x_dot.shape)
        
    return {"variables": {"x": x, "y": y, "z": z}, "y_dot": x_dot}