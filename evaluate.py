import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Tree evaluation
# ---------------------------------------------------------------------------

def evaluate_node(node, variables, constants, const_index):
    """
    Evaluate a single node of the expression tree recursively.

    node        : a Node object
    variables   : dict mapping variable names to numpy arrays
                  e.g. {"x": np.array([1,2,3])}
    constants   : list of numerical values for C placeholders
    const_index : list of one integer [i], tracking which constant
                  we are at (list used so it can be mutated across calls)

    Returns a numpy array of evaluated values.
    """
    s = node.symbol

    # --- Terminal nodes ---
    if s == "x":
        return variables["x"]

    if s == "y":
        return variables["y"]

    if s == "C":
        # Fetch the next constant value from the list
        val = constants[const_index[0]] if constants is not None else 1.0
        const_index[0] += 1
        return np.full_like(variables.get("x", np.array([0.0])),
                            val, dtype=float)

    # --- Binary operators ---
    if s == "+":
        left  = evaluate_node(node.children[0], variables, constants, const_index)
        right = evaluate_node(node.children[1], variables, constants, const_index)
        return left + right

    if s == "-":
        left  = evaluate_node(node.children[0], variables, constants, const_index)
        right = evaluate_node(node.children[1], variables, constants, const_index)
        return left - right

    if s == "*":
        left  = evaluate_node(node.children[0], variables, constants, const_index)
        right = evaluate_node(node.children[1], variables, constants, const_index)
        return left * right

    if s == "/":
        left  = evaluate_node(node.children[0], variables, constants, const_index)
        right = evaluate_node(node.children[1], variables, constants, const_index)
        # Avoid division by zero
        right = np.where(np.abs(right) < 1e-10, 1e-10, right)
        return left / right

    # --- Unary operators ---
    if s == "sin":
        child = evaluate_node(node.children[0], variables, constants, const_index)
        return np.sin(child)

    if s == "cos":
        child = evaluate_node(node.children[0], variables, constants, const_index)
        return np.cos(child)

    if s == "exp":
        child = evaluate_node(node.children[0], variables, constants, const_index)
        # Clip to avoid overflow
        return np.exp(np.clip(child, -100, 100))

    if s == "log":
        child = evaluate_node(node.children[0], variables, constants, const_index)
        # Log only defined for positive values
        return np.log(np.abs(child) + 1e-10)

    # --- Root and intermediate nodes ---
    # f and M just pass through to their single child
    if s in ("f", "M"):
        return evaluate_node(node.children[0], variables, constants, const_index)

    # Unknown symbol
    raise ValueError(f"Unknown symbol in tree: {s}")


def evaluate_tree(node, variables, constants=None):
    """
    Evaluate a complete expression tree on numerical data.

    node      : root Node of the expression tree
    variables : dict mapping variable names to numpy arrays
    constants : list of numerical values for C placeholders
                (None means all constants default to 1.0)

    Returns a numpy array of evaluated values.
    """
    # const_index is a list so it can be mutated inside evaluate_node
    const_index = [0]
    return evaluate_node(node, variables, constants, const_index)


# ---------------------------------------------------------------------------
# Constant handling
# ---------------------------------------------------------------------------

def count_constants(node):
    """
    Count the number of constant placeholders C in the tree.
    Needed to know how many values Powell's method must optimize.
    """
    if node.symbol == "C":
        return 1
    return sum(count_constants(child) for child in node.children)


def objective(constants, node, variables, y_dot):
    """
    Objective function for Powell's optimization.
    Returns the MSE between the tree's output and the target derivative.

    constants : array of constant values being optimized
    node      : expression tree with C placeholders
    variables : dict of input variable arrays
    y_dot     : target derivative array
    """
    try:
        y_pred = evaluate_tree(node, variables, list(constants))
        return np.mean((y_pred - y_dot) ** 2)
    except Exception:
        # If evaluation fails (e.g. nan/inf), return a large error
        return 1e10


def estimate_constants(node, variables, y_dot):
    """
    Find the optimal numerical values for all C placeholders
    in the tree using Powell's optimization method (as in the paper).

    node      : expression tree containing C placeholders
    variables : dict of input variable arrays
    y_dot     : target derivative array we are trying to match

    Returns a list of optimal constant values.
    """
    n_constants = count_constants(node)

    # No constants to optimize
    if n_constants == 0:
        return []

    # Start from all-ones initial guess
    x0 = np.ones(n_constants)

    result = minimize(
        objective,
        x0,
        args=(node, variables, y_dot),  # extra arguments passed to objective
        method="powell"
    )

    return list(result.x)