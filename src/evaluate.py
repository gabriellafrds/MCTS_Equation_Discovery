import numpy as np
import pysindy as ps
from src.tree import extract_features_from_tree

UNARY_OPS = {
    "sin": np.sin,
    "cos": np.cos,
    "exp": lambda x: np.exp(np.clip(x, -100, 100)),
    "-": np.negative
}

BINARY_OPS = {
    "*": np.multiply
}

def evaluate_node(node, variables):
    """
    Map AST nodes to physical data vectors.
    """
    s = node.symbol

    if s in ["x", "y", "z"]:
        # safe default to zeros if the data array lacks that dimension
        return variables.get(s, np.zeros_like(variables["x"]))
        
    if s == "1":
        return np.ones_like(variables["x"])

    if s == "M":
        return evaluate_M(node, variables)

    raise ValueError(f"Unknown generic symbol in tree: {s}")

def evaluate_M(node, variables):
    """
    Evaluate structural mathematical nodes via operator dispatch.
    """
    children = node.children
    if not children:
        raise ValueError(f"Node has no children: {node.symbol}")

    n_children = len(children)
    
    if n_children == 1:
        return evaluate_node(children[0], variables)

    if n_children == 2:
        op = children[0].symbol
        arg = evaluate_node(children[1], variables)
        if op in UNARY_OPS:
            return UNARY_OPS[op](arg)
        raise ValueError(f"Unknown unary operator: {op}")

    if n_children == 3:
        left = evaluate_node(children[0], variables)
        op = children[1].symbol
        right = evaluate_node(children[2], variables)
        if op in BINARY_OPS:
            return BINARY_OPS[op](left, right)
        raise ValueError(f"Unknown binary operator: {op}")

    raise ValueError(f"Unexpected number of children: {n_children}")

def evaluate_tree_sindy(root_node, variables, y_dot, locked_features_cols=None):
    """
    Compile AST into numeric vectors and resolve SINDy STLSQ sparse constants.
    """
    locked_features_cols = locked_features_cols or []
    features = extract_features_from_tree(root_node)
    
    # If the tree generated no valid features
    if not features:
        return 1e10, []

    # Map the current data vector shape
    x_shape = variables.get("x", np.array([0.0])).shape[0]

    theta_cols = []
    for f_node in features:
        val = evaluate_node(f_node, variables)
        
        # handle scalar (example if the grammar generated a constant term indirectly somehow)
        if np.isscalar(val):
            val = np.full(x_shape, val)
        else:
            val = val.reshape(-1)
            
        theta_cols.append(val)
        
    # Combine locked features from previous iterations with the new feature
    all_cols = locked_features_cols + theta_cols
    if not all_cols:
        return 1e10, []
        
    Theta = np.column_stack(all_cols)
    
    # init the core SINDy STLSQ optimizer
    # threshold < 0.1 because our true physics constant is -0.1
    optimizer = ps.STLSQ(threshold=0.01, alpha=0.01)
    
    try:
        optimizer.fit(Theta, y_dot)
    except Exception as e:
        print("Exception in evaluate:", e)
        # Fails if matrix is completely singular or data is corrupted
        return 1e10, []
        
    y_pred = optimizer.predict(Theta)
    mse = np.mean((y_pred.flatten() - y_dot) ** 2)
    
    return float(mse), optimizer.coef_