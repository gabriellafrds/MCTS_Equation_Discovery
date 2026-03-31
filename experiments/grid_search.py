import itertools
import pandas as pd
from src.main import main
import os

def run_grid_search():
    param_grid = {
        "n_episodes": [1000, 2000],
        "n_simulations": [20, 40],
        "t_max": [15],
        "c": [0.5, 1.0],
        "gamma": [0.001, 0.005],
        "alpha": [0.5, 1.0],
        "beta": [0.0, 0.1],
        "noise": [0.0],
        "n_points": [500, 1000]
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[k] for k in keys)))
    
    print(f"Total parameter combinations to test: {len(combinations)}")
    
    results = []
    
    for i, values in enumerate(combinations):
        config = dict(zip(keys, values))
        print(f"\n=============================================")
        print(f"Running iteration {i+1}/{len(combinations)}")
        print(f"config: {config}")
        
        try:
            # We intercept stdout to keep the console clean-ish, but let main() output for monitoring
            res = main(config)
            
            # Record results
            row = config.copy()
            row["final_features"] = " + ".join(res["final_features"])
            row["bic"] = res["bic"]
            row["mse"] = res["mse"]
            results.append(row)
            
        except Exception as e:
            print(f"Error on iteration {i+1}: {e}")
            row = config.copy()
            row["error"] = str(e)
            results.append(row)

    # Save results
    df = pd.DataFrame(results)
    output_file = "results/results/grid_search_very_complex_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nGrid search complete. Results saved to {output_file}")

if __name__ == "__main__":
    run_grid_search()
