# spl_sindy_v2/evaluate_v2.py
#
# Evaluates a single expression tree (no C placeholders).
# Reuses the same evaluate_M logic as the original evaluate.py
# but without constant handling — SINDy handles coefficients.
#
# The evaluate_node function mirrors the original exactly:
# it handles f, M as pass-through nodes, and dispatches
# to operators or terminals.

import sys
import os
import numpy as np

# Allow imports from parent directory (tree.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_node(node, variables):
    """
    Recursively evaluate a single expression tree node.

    node      : a Node object (from tree.py)
    variables : dict mapping variable names to numpy arrays
                e.g. {"x": np.array([...]), "y": np.array([...])}

    Returns a numpy array of evaluated values.
    No constants — all leaf nodes are variables.
    """
    s = node.symbol

    # --- Terminal nodes ---
    if s == "x":
        return variables["x"]
    if s == "y":
        return variables.get("y", np.zeros_like(variables["x"]))
    if s == "z":
        return variables.get("z", np.zeros_like(variables["x"]))

    # --- Pass-through nodes: f and M ---
    if s in ("f", "M"):
        return evaluate_M(node, variables)

    # --- Binary operators (called directly if tree was built correctly) ---
    if s == "+":
        left  = evaluate_node(node.children[0], variables)
        right = evaluate_node(node.children[1], variables)
        return left + right

    if s == "-":
        left  = evaluate_node(node.children[0], variables)
        right = evaluate_node(node.children[1], variables)
        return left - right

    if s == "*":
        left  = evaluate_node(node.children[0], variables)
        right = evaluate_node(node.children[1], variables)
        return left * right

    if s == "/":
        left  = evaluate_node(node.children[0], variables)
        right = evaluate_node(node.children[1], variables)
        right = np.where(np.abs(right) < 1e-10, 1e-10, right)
        return left / right

    # --- Unary operators ---
    if s == "sin":
        child = evaluate_node(node.children[0], variables)
        return np.sin(child)

    if s == "cos":
        child = evaluate_node(node.children[0], variables)
        return np.cos(child)

    if s == "exp":
        child = evaluate_node(node.children[0], variables)
        return np.exp(np.clip(child, -100, 100))

    if s == "log":
        child = evaluate_node(node.children[0], variables)
        return np.log(np.abs(child) + 1e-10)

    raise ValueError(f"Unknown symbol in tree: {s}")


def evaluate_M(node, variables):
    """
    Evaluate a node whose symbol is 'M' or 'f'.
    Inspects children to determine which rule was applied:

    - 1 child  : terminal (x, y, z) or nested M
    - 2 children: unary operator (sin M, cos M, exp M, log M)
                  children = [Node(operator), Node(M)]
    - 3 children: binary operator (M + M, M * M, etc.)
                  children = [Node(M_left), Node(operator), Node(M_right)]
    """
    children = node.children

    if not children:
        raise ValueError(f"M/f node has no children: {node.symbol}")

    # Case 1: single child (terminal or nested M)
    if len(children) == 1:
        return evaluate_node(children[0], variables)

    # Case 2: unary operator
    if len(children) == 2:
        op  = children[0].symbol
        arg = evaluate_node(children[1], variables)
        if op == "sin": return np.sin(arg)
        if op == "cos": return np.cos(arg)
        if op == "exp": return np.exp(np.clip(arg, -100, 100))
        if op == "log": return np.log(np.abs(arg) + 1e-10)
        raise ValueError(f"Unknown unary operator: {op}")

    # Case 3: binary operator
    if len(children) == 3:
        left  = evaluate_node(children[0], variables)
        op    = children[1].symbol
        right = evaluate_node(children[2], variables)
        if op == "+": return left + right
        if op == "-": return left - right
        if op == "*": return left * right
        if op == "/":
            right = np.where(np.abs(right) < 1e-10, 1e-10, right)
            return left / right
        raise ValueError(f"Unknown binary operator: {op}")

    raise ValueError(f"M node has {len(children)} children — unexpected")


def evaluate_feature(sequence, variables):
    """
    Convenience function: build tree from sequence and evaluate it.

    sequence  : list of rules (left, right) from MCTS
    variables : dict of variable arrays

    Returns a numpy array or None if evaluation fails.
    """
    # Import here to avoid circular imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tree import build_tree_step_by_step

    try:
        tree = build_tree_step_by_step(sequence)
        phi  = evaluate_node(tree, variables)
        if not np.all(np.isfinite(phi)):
            return None
        return phi
    except Exception:
        return None
