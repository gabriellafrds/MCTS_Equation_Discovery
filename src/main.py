import numpy as np
from scipy.integrate import odeint
from src.grammar import get_valid_actions, is_complete, RULES
from src.tree import build_tree_step_by_step, extract_features_from_tree
from src.evaluate import evaluate_tree_sindy
from src.mcts import MCTS
from utils.data_generators import lorenz_xdot, nonlinear_three_var, complex_three_var, very_complex_three_var
import pysindy as ps

# ---------------------------------------------------------------------------
# Data generation (Harmonic Oscillator Benchmark)
# ---------------------------------------------------------------------------

def generate_data_harmonic_oscillator(n_points=1000, noise=0.0):
    """
    Generate data for the Damped Harmonic Oscillator:
        dx/dt = -0.1*x + y
        dy/dt = -x - 0.1*y
    """
    def harmonic_oscillator(z, t):
        return [-0.1*z[0] + z[1], -z[0] - 0.1*z[1]]

    dt = 0.05
    t_train = np.arange(0, n_points * dt, dt)
    z0 = [1.0, 0.0]
    z_train = odeint(harmonic_oscillator, z0, t_train)

    if noise > 0:
        z_train += np.random.normal(0, noise, z_train.shape)

    # We target discovering the underlying physics of dx/dt
    y_dot = np.array([harmonic_oscillator(z, 0)[0] for z in z_train])

    data = {
        "variables": {"x": z_train[:, 0], "y": z_train[:, 1]},
        "y_dot": y_dot
    }
    return data


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def sequence_to_library_strings(sequence):
    """
    Converts a sequence of MCTS production rules into a readable list of 
    basis feature strings (e.g. ['x']).
    """
    if not sequence: return []
    tree = build_tree_step_by_step(sequence)
    features = extract_features_from_tree(tree)
    
    def render_node(node):
        if node.is_terminal(): return node.symbol
        if len(node.children) == 1: return render_node(node.children[0])
        if len(node.children) == 2: return f"{node.children[0].symbol}({render_node(node.children[1])})"
        if len(node.children) == 3: return f"({render_node(node.children[0])} {node.children[1].symbol} {render_node(node.children[2])})"
        return "?"
        
    return [render_node(f) for f in features]

def evaluate_locked_dictionary(locked_features_cols, y_dot):
    """
    Evaluates the final locked dictionary of features to print the equation.
    """
    import pysindy as ps
    if not locked_features_cols:
        return 1e10, []
        
    Theta = np.column_stack(locked_features_cols)
    optimizer = ps.STLSQ(threshold=0.05, alpha=0.05)
    
    try:
        optimizer.fit(Theta, y_dot)
        y_pred = optimizer.predict(Theta)
        mse = np.mean((y_pred.flatten() - y_dot) ** 2)
        return mse, optimizer.coef_
    except:
        return 1e10, []

def print_final_equation(locked_features_strings, locked_features_cols, y_dot):
    print("\n" + "="*60)
    print("SEARCH COMPLETE - FINAL SINDy MODEL")
    print("="*60)
    
    mse, coefs = evaluate_locked_dictionary(locked_features_cols, y_dot)
    
    print(f"Final Combined MSE: {mse:.6e}")
    if len(coefs) > 0:
        eq_parts = []
        flat_coefs = np.ravel(coefs)
        for i, c in enumerate(flat_coefs):
            if abs(c) > 1e-5 and i < len(locked_features_strings):
                eq_parts.append(f"{c:.4f} * {locked_features_strings[i]}")
        
        if not eq_parts:
            print("dx/dt = 0")
        else:
            print("dx/dt = " + " + ".join(eq_parts))
    else:
        print("dx/dt = 0")
        
    print(f"\nFinal Dictionary: {locked_features_strings}")
    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config = None):
    if config is None:
        config = {
            "n_episodes"    : 1000,   
            "n_simulations" : 20,     
            "t_max"         : 15,     
            "c"             : 0.5,    
            "gamma"         : 0.005,  
            "alpha"         : 0.5,    
            "noise"         : 0.0,    
            "n_points"      : 500,   
        }

    print("Symbolic Physics Learner — Dynamic PySINDy Integration")
    print(f"Using config: {config}")
    print("Target: very_complex_three_var: y_dot = 1.0*x + 1.0*sin((((x * x) * z) * y)) + 1.0*cos(sin((z * y)))")
    print(f"Episodes        : {config['n_episodes']}")
    
    data = very_complex_three_var(n_points=config["n_points"], noise=config["noise"])

    import types
    grammar = types.SimpleNamespace(
        get_valid_actions = get_valid_actions,
        is_complete       = is_complete,
        rules             = RULES,
    )

    print("\nStarting Iterative Feature Search...")
    
    locked_features_cols = []
    locked_features_strings = []
    best_global_bic = float('inf')
    max_features = 10
    
    for iteration in range(max_features):
        print(f"\n--- Iteration {iteration+1}/{max_features} ---")
        agent = MCTS(
            grammar=grammar,
            data=data,
            locked_features_cols=locked_features_cols,
            c=config["c"],
            n_simulations=config["n_simulations"],
            t_max=config["t_max"],
            gamma=config["gamma"],
            alpha=config["alpha"]
        )

        best_sequence, best_reward = agent.run(n_episodes=config["n_episodes"])
        
        if not best_sequence:
            print("MCTS failed to find any valid feature.")
            break
            
        tree = build_tree_step_by_step(best_sequence)
        features = extract_features_from_tree(tree)
        if not features:
            continue
            
        new_str = sequence_to_library_strings(best_sequence)[0]
        
        # Calculate the numerical column of the new feature
        from evaluate import evaluate_node
        new_col = evaluate_node(features[0], data["variables"])
        if np.isscalar(new_col):
            new_col = np.full(data["y_dot"].shape[0], new_col)
        else:
            new_col = new_col.reshape(-1)
            
        # Re-evaluate with SINDy to get the MSE and BIC of the combined library
        mse, coefs = evaluate_tree_sindy(tree, data["variables"], data["y_dot"], locked_features_cols)
        
        l0_norm = np.count_nonzero(coefs)
        n_points = len(data["y_dot"])
        mse_clamped = max(mse, 1e-12)
        bic = n_points * np.log(mse_clamped) + l0_norm * np.log(n_points)
        
        print(f"Proposed Feature: {new_str} | Combined BIC: {bic:.2f} | MSE: {mse:.6e}")
        
        # Heuristic 1: Explicitly prevent exact duplicate feature strings
        if new_str in locked_features_strings:
            print(f"-> REJECTED. '{new_str}' is already in the dictionary.")
            print("-> Terminating search (Occam's razor).")
            break
            
        # Heuristic 2: Occam's razor. If BIC drops significantly, accept it.
        # We require a massive threshold improvement to avoid noise adding useless terms.
        improvement_threshold = 10.0
        
        if abs(np.ravel(coefs)[-1]) < 1e-5:
            print(f"-> REJECTED. SINDy assigned 0.0 to '{new_str}'.")
            break

        if bic <= best_global_bic - improvement_threshold:
            best_global_bic = bic
            locked_features_cols.append(new_col)
            locked_features_strings.append(new_str)
            print(f"-> ACCEPTED! Dictionary is now: {locked_features_strings}")
        else:
            print(f"-> REJECTED. Information gain too low. (Best BIC remains: {best_global_bic:.2f})")
            print("-> Terminating search (Occam's razor).")
            break

    print_final_equation(locked_features_strings, locked_features_cols, data["y_dot"])

    # Calculate final MSE from the locked features
    if len(locked_features_cols) > 0:
        Theta_final = np.column_stack(locked_features_cols)
        optimizer = ps.STLSQ(threshold=0.1)
        optimizer.fit(Theta_final, data["y_dot"])
        y_pred = optimizer.predict(Theta_final)
        final_mse = np.mean((data["y_dot"] - y_pred)**2)
    else:
        final_mse = float('inf')
        
    return {
        "final_features": locked_features_strings,
        "bic": best_global_bic,
        "mse": final_mse
    }


if __name__ == "__main__":
    main()