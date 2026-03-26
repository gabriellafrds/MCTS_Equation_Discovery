import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Tree evaluation
# ---------------------------------------------------------------------------

def evaluate_node(node, variables, constants, const_index):
    s = node.symbol

    # --- Terminal nodes ---
    if s == "x":
        return variables["x"]

    if s == "C":
        val = constants[const_index[0]] if constants is not None else 1.0
        const_index[0] += 1
        return np.full_like(variables.get("x", np.array([0.0])), val, dtype=float)

    # --- Pass-through nodes: f and M ---
    # These are structural nodes — find the real operator among children
    if s in ("f", "M"):
        # Find the operator child (non-M, non-terminal symbol)
        # and evaluate the full subtree rooted at M
        return evaluate_M(node, variables, constants, const_index)

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
        return np.exp(np.clip(child, -100, 100))

    if s == "log":
        child = evaluate_node(node.children[0], variables, constants, const_index)
        return np.log(np.abs(child) + 1e-10)

    raise ValueError(f"Unknown symbol in tree: {s}")


def evaluate_M(node, variables, constants, const_index):
    """
    Evaluate a node whose symbol is 'M' or 'f'.
    The structure of an M node depends on which rule was applied:

    - M -> x, y, C        : node has one child which is the terminal
    - M -> sin M, cos M   : node has two children: [operator_symbol, M_child]
                            but operator_symbol is stored as a child Node
    - M -> M + M          : node has three children: [M_left, operator, M_right]

    We inspect the children to determine the case.
    """
    children = node.children

    # No children: should not happen for a well-formed tree
    if not children:
        raise ValueError(f"M node has no children: {node.symbol}")

    # Case 1: single terminal child — M -> x, y, C
    if len(children) == 1:
        return evaluate_node(children[0], variables, constants, const_index)

    # Case 2: unary operator — M -> sin M, cos M, exp M, log M
    # children = [Node("sin"), Node("M")]  or [Node("cos"), Node("M")] etc.
    if len(children) == 2:
        op   = children[0].symbol   # the operator
        arg  = evaluate_node(children[1], variables, constants, const_index)
        if op == "sin": return np.sin(arg)
        if op == "cos": return np.cos(arg)
        if op == "exp": return np.exp(np.clip(arg, -100, 100))
        if op == "log": return np.log(np.abs(arg) + 1e-10)
        raise ValueError(f"Unknown unary operator: {op}")

    # Case 3: binary operator — M -> M + M, M * M, M / M
    # children = [Node("M"), Node("+"), Node("M")]
    if len(children) == 3:
        left  = evaluate_node(children[0], variables, constants, const_index)
        op    = children[1].symbol
        right = evaluate_node(children[2], variables, constants, const_index)
        if op == "+": return left + right
        if op == "-": return left - right
        if op == "*": return left * right
        if op == "/":
            right = np.where(np.abs(right) < 1e-10, 1e-10, right)
            return left / right
        raise ValueError(f"Unknown binary operator: {op}")

    raise ValueError(f"M node has unexpected number of children: {len(children)}")


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