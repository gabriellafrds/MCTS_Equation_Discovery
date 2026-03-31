import numpy as np
from src.tree import build_tree_step_by_step
from src.evaluate import evaluate_tree_sindy
from src.reward import compute_reward
import src.grammar as grammar

def test():
    print("Testing Grammar...")
    sequence = [
        ("f", ["Library"]),
        ("Library", ["Feature", "Library"]),
        ("Feature", ["M"]),
        ("M", ["x"]),
        ("Library", ["Feature"]),
        ("Feature", ["M"]),
        ("M", ["y"])
    ]
    
    # Check grammar
    assert grammar.is_complete(sequence)
    print("Grammar check passed.")
    
    # Generate dummy data
    t = np.linspace(0, 10, 100)
    x = np.cos(t)
    y = np.sin(t)
    
    x_dot = -y
    
    data = {
        "variables": {"x": x, "y": y},
        "y_dot": x_dot
    }
    
    print("Testing Evaluate and STLSQ...")
    tree = build_tree_step_by_step(sequence)
    mse, coefs = evaluate_tree_sindy(tree, data["variables"], data["y_dot"])
    
    print(f"MSE: {mse}")
    print(f"Coefs: {coefs}")
    
    print("Testing Reward Computation...")
    reward = compute_reward(sequence, data, grammar)
    print(f"Reward: {reward}")

if __name__ == "__main__":
    test()
