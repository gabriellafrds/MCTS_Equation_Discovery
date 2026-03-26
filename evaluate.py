import numpy as np
import pysindy as ps
from tree import extract_features_from_tree

# ---------------------------------------------------------------------------
# Feature evaluation
# ---------------------------------------------------------------------------

def evaluate_node(node, variables):
    s = node.symbol

    if s == "x":
        return variables["x"]
    if s == "y":
        return variables.get("y", np.zeros_like(variables["x"]))

    if s in ("Feature", "M"):
        return evaluate_M(node, variables)

    raise ValueError(f"Unknown generic symbol in tree: {s}")

def evaluate_M(node, variables):
    """
    Evaluate an M node which can be a unary or binary operator.
    """
    children = node.children
    if not children:
        raise ValueError(f"Node has no children: {node.symbol}")

    if len(children) == 1:
        return evaluate_node(children[0], variables)

    if len(children) == 2:
        op  = children[0].symbol
        arg = evaluate_node(children[1], variables)
        if op == "sin": return np.sin(arg)
        if op == "cos": return np.cos(arg)
        if op == "exp": return np.exp(np.clip(arg, -100, 100))
        raise ValueError(f"Unknown unary operator: {op}")

    if len(children) == 3:
        left  = evaluate_node(children[0], variables)
        op    = children[1].symbol
        right = evaluate_node(children[2], variables)
        if op == "*": return left * right
        raise ValueError(f"Unknown binary operator: {op}")

    raise ValueError(f"Unexpected number of children: {len(children)}")


# ---------------------------------------------------------------------------
# SINDy Integration Core
# ---------------------------------------------------------------------------

def evaluate_tree_sindy(root_node, variables, y_dot):
    """
    Extracts all feature terms from the MCTS tree, evaluates them to build
    a feature matrix (Theta), and computes the sparse regression using STLSQ.
    
    Returns:
        mse: Mean Squared Error of the SINDy model prediction
        coefs: Solved STLSQ coefficients array
    """
    features = extract_features_from_tree(root_node)
    
    # If the tree generated no valid features
    if not features:
        return 1e10, []

    theta_cols = []
    
    # Extract the length of the data to expand scalar features properly
    # Get any active variable to check shape
    x_shape = variables.get("x", np.array([0.0])).shape[0]

    for f_node in features:
        val = evaluate_node(f_node, variables)
        
        # Handle scalar (e.g. if the grammar generated a constant term indirectly somehow)
        if np.isscalar(val):
            val = np.full(x_shape, val)
        else:
            val = val.reshape(-1)
            
        theta_cols.append(val)
        
    Theta = np.column_stack(theta_cols)
    
    # Initialize the core SINDy STLSQ optimizer
    # Threshold must be < 0.1 because our true physics constant is -0.1
    optimizer = ps.STLSQ(threshold=0.05, alpha=0.05)
    
    try:
        optimizer.fit(Theta, y_dot)
    except Exception:
        # Fails if matrix is completely singular or data is corrupted
        return 1e10, []
        
    y_pred = optimizer.predict(Theta)
    mse = np.mean((y_pred - y_dot) ** 2)
    
    return mse, optimizer.coef_