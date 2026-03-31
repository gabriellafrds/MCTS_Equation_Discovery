# Monte Carlo Tree Search for Equation Discovery (MCTS-SINDy)

This project explores a hybrid architecture for symbolic regression by combining the combinatorial search power of **Monte Carlo Tree Search (MCTS)** with the robust, mathematically parsimonious coefficient extraction of **Sparse Identification of Nonlinear Dynamical Systems (SINDy)**. 

This repository was developed as part of the "Monte Carlo Tree Search" course project for the M2 IASD Master's program at Université Paris Dauphine-PSL.

## Overview

Discovering governing physical equations from noisy trajectory data is a famously difficult combinatorial problem. Traditional sparse regression methods (like PySINDy) rely on pre-built structural libraries (e.g., polynomials up to degree $N$) and catastrophically overfit in the presence of noise when dealing with deeply nested non-linearities. Pure symbolic mathematics models, like the Symbolic Physics Learner (SPL), suffer from massive computational overhead because they attempt to optimize continuous physical constants simultaneously while randomly exploring discrete mathematical syntax.

**MCTS-SINDy** bridges this gap:
1. **MCTS** constructs mathematical grammar trees ($\sin(x)$, $x \cdot y$) sequentially as a discrete Markov Decision Process.
2. Instead of guessing coefficients, the MCTS sequence is immediately compiled into a sparse feature matrix.
3. **PySINDy's STLSQ (Sequential Threshold Ridge Regression)** analytically solves for the optimal continuous coefficients in closed form.
4. The tree is rewarded using a **Bayesian Information Criterion (BIC)**, enforcing strict Occam's Razor discipline to punish deeply nested structures that do not offer massive predictive improvements.

## Benchmarks & Results

The architecture is benchmarked against three systems of increasing complexity using high-precision Runge-Kutta 45 synthetic data, across multiple noise tiers (0%, 1%, 5% Gaussian noise applied to $\dot{X}$):

1. **Damped Harmonic Oscillator:** ($\dot{x} = -0.1x + 2y$). MCTS-SINDy perfectly bounds parsimony at 2 features across all noise tiers, whereas baseline algorithms degrade and overfit.
2. **Lorenz Attractor (Z-axis):** ($\dot{z} = xy - \frac{8}{3}z$). Resolves the exact mathematical cross-term coefficients analytically without triggering combinatorial explosions.
3. **Deep Nested Bound:** ($\dot{x} = -x + \sin(x^2 \cdot y \cdot z)$). This acts as a library "kill screen." While standard SINDy collapses into a 10-term polynomial mess regardless of noise, MCTS-SINDy elegantly uncovers the linear sink and constructs a profound approximation: $x \cdot \sin(x \cdot y \cdot z)$, maintaining strict interpretability.

## Repository Structure

```text
MCTS_Equation_Discovery/
├── src/                # Core algorithmic logic
│   ├── evaluate.py     # Maps grammar trees to numpy trajectory data
│   ├── grammar.py      # Abstract mathematical syntax definitions
│   ├── main.py         # Entry point for sequence-to-library conversion
│   ├── mcts.py         # The generic Monte Carlo Tree Search agent
│   ├── reward.py       # BIC scaling and STLSQ reward evaluations
│   └── tree.py         # Graph node parsers for symbol sequences

├── utils/
│   ├── metrics.py      # RMSE and maths checks
│   └── data_generators.py # RK45 high-precision physical environments
├── experiments/        # Analytical execution scripts
│   ├── benchmark.py
│   ├── grid_search.py
│   ├── run_benchmarks.py
│   ├── run_pipeline.py # 3-way evaluation gauntlet (MCTS vs SINDy vs SPL)
│   └── pysindy/        # Legacy standalone PySINDy baseline scripts
├── tests/              # Unit testing suite
├── results/            # Auto-generated JSON and CSV telemetry
├── documents/          # LaTeX source files, reports, and references
└── README.md
```

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/MCTS-project.git
   cd MCTS-project/MCTS_Equation_Discovery
   ```

2. **Install requirements:**
   *(Ensure you have Python 3.8+)*
   ```bash
   pip install -r requirements.txt
   ```
   *Core dependencies: `numpy`, `scipy`, `pysindy`.*

3. **Running Experiments:**
   > **CRITICAL:** Because the codebase is strictly modularized into Python packages, you **must** execute scripts from the root directory using the Python module execution flag (`-m`).

   To run the primary 3-way benchmarking pipeline:
   ```bash
   python -m experiments.run_pipeline
   ```
   
   To run isolated benchmarks or grid searches:
   ```bash
   python -m experiments.run_benchmarks
   python -m experiments.grid_search
   ```

4. **Viewing Results:**
   All execution pipelines will automatically format and export their telemetry directly into the `results/` directory (e.g., `results/benchmark_results.json`).