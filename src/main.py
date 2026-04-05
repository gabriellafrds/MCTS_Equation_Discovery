import types
import numpy as np
import pysindy as ps
from src.grammar import get_valid_actions, is_complete, RULES
from src.tree import build_tree_step_by_step, extract_features_from_tree
from src.evaluate import evaluate_tree_sindy, evaluate_node
from src.mcts import MCTS
from utils.data_generators import very_complex_three_var

def sequence_to_library_strings(sequence):
    """
    Convert an MCTS rule sequence to a mathematical feature string.
    """
    if not sequence: return []
    features = extract_features_from_tree(build_tree_step_by_step(sequence))
    
    def render(node):
        if node.is_terminal(): return node.symbol
        c = node.children
        if len(c) == 1: return render(c[0])
        if len(c) == 2: return f"{c[0].symbol}({render(c[1])})"
        if len(c) == 3: return f"({render(c[0])} {c[1].symbol} {render(c[2])})"
        return "?"
        
    return [render(f) for f in features]

def evaluate_locked_dictionary(locked_features_cols, y_dot):
    if not locked_features_cols:
        return 1e10, []
        
    Theta = np.column_stack(locked_features_cols)
    optimizer = ps.STLSQ(threshold=0.01, alpha=0.01)
    try:
        optimizer.fit(Theta, y_dot)
        y_pred = optimizer.predict(Theta)
        mse = np.mean((y_pred.flatten() - y_dot) ** 2)
        return float(mse), optimizer.coef_
    except BaseException:
        return 1e10, []

def print_final_equation(features_strings, features_cols, y_dot):
    print("\n" + "="*60)
    print("SEARCH COMPLETE - FINAL SINDy MODEL")
    print("="*60)
    
    mse, coefs = evaluate_locked_dictionary(features_cols, y_dot)
    print(f"Final Combined MSE: {mse:.6e}")
    
    eq_parts = [f"{c:.4f} * {s}" for c, s in zip(np.ravel(coefs), features_strings) if abs(c) > 1e-5]
    print(f"dx/dt = {' + '.join(eq_parts) if eq_parts else '0'}")
    print(f"Final Dictionary: {features_strings}\n" + "="*60)

def main(config=None):
    config = config or {
        "n_episodes": 1000, "n_simulations": 20, "t_max": 15, "c": 0.5,
        "gamma": 0.005, "alpha": 0.5, "noise": 0.0, "n_points": 500
    }

    print("Symbolic Physics Learner — Dynamic PySINDy Integration")
    data = very_complex_three_var(n_points=config["n_points"], noise=config["noise"])
    grammar = types.SimpleNamespace(get_valid_actions=get_valid_actions, is_complete=is_complete, rules=RULES)

    locked_features_cols, locked_features_strings = [], []
    best_global_bic = float('inf')
    
    for iteration in range(10):
        print(f"\n--- Iteration {iteration + 1} ---")
        agent = MCTS(grammar, data, locked_features_cols, **{k: config[k] for k in ["c", "n_simulations", "t_max", "gamma", "alpha"]})
        best_seq, _ = agent.run(config["n_episodes"])
        
        if not best_seq:
            print("MCTS failed to find valid feature.")
            break
            
        tree = build_tree_step_by_step(best_seq)
        features = extract_features_from_tree(tree)
        if not features: continue
            
        new_str = sequence_to_library_strings(best_seq)[0]
        new_col = evaluate_node(features[0], data["variables"])
        new_col = np.full(data["y_dot"].shape[0], new_col) if np.isscalar(new_col) else new_col.reshape(-1)
            
        mse, coefs = evaluate_tree_sindy(tree, data["variables"], data["y_dot"], locked_features_cols)
        
        n_points = len(data["y_dot"])
        bic = n_points * np.log(max(mse, 1e-12)) + np.count_nonzero(coefs) * np.log(n_points)
        
        print(f"Proposed: {new_str} | Combined BIC: {bic:.2f} | MSE: {mse:.6e}")
        
        if new_str in locked_features_strings or abs(np.ravel(coefs)[-1]) < 1e-5:
            print(f"-> REJECTED (Duplicate or Zero Coefficient)")
            break

        if bic <= best_global_bic - 10.0:
            best_global_bic = bic
            locked_features_cols.append(new_col)
            locked_features_strings.append(new_str)
            print(f"-> ACCEPTED! Dictionary: {locked_features_strings}")
        else:
            print(f"-> REJECTED. Insufficient information gain.")
            break

    print_final_equation(locked_features_strings, locked_features_cols, data["y_dot"])
    
    final_mse = float('inf')
    if locked_features_cols:
        mse, _ = evaluate_locked_dictionary(locked_features_cols, data["y_dot"])
        final_mse = mse
        
    return {"final_features": locked_features_strings, "bic": best_global_bic, "mse": final_mse}

if __name__ == "__main__":
    main()