import numpy as np

def exact_match_check(found_dict, truth_dict):
    """
    Checks if the discovered dictionary perfectly matches the ground truth.
    Handles the (x*y) vs (y*x) grammar permutation for Lorenz.
    """
    found_keys = set(found_dict.keys())
    truth_keys = set(truth_dict.keys())
    
    normalized_found = {k.replace("(y * x)", "(x * y)") for k in found_keys}
    return normalized_found == truth_keys and len(found_dict) == len(truth_dict)

def compute_coef_rmse(found_dict, truth_dict):
    """
    Compute RMSE between found and ground-truth coefficients.
    Penalizes missing terms with their full truth value.
    """
    rmse = 0.0
    for k, truth_val in truth_dict.items():
        alt_k = "(y * x)" if k == "(x * y)" else ("(x * y)" if k == "(y * x)" else None)
        
        if k in found_dict:
            rmse += (found_dict[k] - truth_val) ** 2
        elif alt_k and alt_k in found_dict:
            rmse += (found_dict[alt_k] - truth_val) ** 2
        else:
            rmse += truth_val ** 2  # Penalize for missing term
            
    return float(np.sqrt(rmse / max(1, len(truth_dict))))
