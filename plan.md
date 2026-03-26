# Project Plan: Dynamical-MCTS & SINDy Integration

## **Phase 1: Baseline Hardening & Environment Setup**
**Goal:** Optimize the existing MCTS codebase, eliminate redundant computations, and transition from static polynomial benchmarks to dynamical systems.

* **Task 1.1: Implement Expression Caching (Memoization)**
    * Create a hash map (e.g., using string representations of trees) to store previously evaluated mathematical structures.
    * Bypass the SciPy Powell optimizer and `evaluate_tree` functions if a tree has already been seen, drastically reducing rollout time.
* **Task 1.2: Establish Physics-Based Datasets**
    * Deprecate the `Nguyen-1` benchmark.
    * Generate time-series data ($dx/dt$, $dy/dt$) for foundational dynamical systems:
        * Simple Harmonic Oscillator (Linear)
        * Ideal Pendulum (Non-linear trigonometric)
* **Task 1.3: Grammar Pruning & Constraints**
    * Update `grammar.py` to be state-aware (preventing redundant loops like `exp(exp(x))` or `sin(cos(x))`).

## **Phase 2: The SINDy Integration (Optimization Core)**
**Goal:** Replace the slow, non-convex Powell optimization with a lightning-fast sparse regression layer to accurately identify constants and linear combinations.

* **Task 2.1: Implement Sparse Regression Solver**
    * Write a Sequentially Thresholded Least Squares (STLSQ) algorithm (the core of SINDy).
    * Test the solver in isolation on a hardcoded matrix of candidate functions.
* **Task 2.2: Modify MCTS Output**
    * Adjust the MCTS agent so that instead of returning one rigid equation (e.g., $C_1 \sin(x) + C_2 x^2$), it returns a feature library (a list of basis functions: `[sin(x), x^2, exp(y)]`).
* **Task 2.3: Hybrid Evaluation Pipeline**
    * Wire the MCTS feature library directly into the STLSQ solver. 
    * Use the error from the STLSQ solver as the new `MSE` for the MCTS reward function.

## **Phase 3: Intelligent Exploration (RL Core)**
**Goal:** Overcome the curse of dimensionality by replacing blind uniform random rollouts with directed, score-based exploration.

* **Task 3.1: Implement SHUSS (Sequential Halving Using Scores)**
    * Integrate the SHUSS algorithm to evaluate exploration terms and bias move selection at the root and deep nodes.
    * Use an AMAF (All-Moves-As-First) prior to rank and eliminate poor branches early.
* **Task 3.2: Mini-Batching for Rollout Proxies**
    * Modify the simulation phase to evaluate candidate trees on a small, randomized subset of the data (e.g., 10 points) rather than the full time-series array to maximize search speed.
* **Task 3.3: Early Stopping Mechanisms**
    * Implement fail-safes during rollouts: if an equation triggers mathematical invalidity (e.g., division by zero, massive overflow), terminate the rollout immediately and assign a reward of $0$.

## **Phase 4: Scaling & Complex Benchmarking**
**Goal:** Prove the capabilities of the hybrid model against state-of-the-art baselines using complex physical systems.

* **Task 4.1: Advanced Dynamical Systems Setup**
    * Simulate multi-dimensional, chaotic data:
        * Lotka-Volterra (Predator-Prey dynamics)
        * Lorenz Attractor (3D chaotic system)
* **Task 4.2: Head-to-Head Evaluation**
    * Run vanilla SINDy (with a standard polynomial library) on the datasets.
    * Run the hybrid Dynamical-MCTS on the same datasets.
    * Track and plot three key metrics: Compute Time, Mean Squared Error, and Equation Parsimony.

## **Phase 5: Synthesis & Documentation**
**Goal:** Package the empirical results into a rigorous, well-documented thesis that clearly articulates the mathematical and algorithmic novelty of the approach.

* **Task 5.1: Codebase Cleanup**
    * Ensure all modules are fully typed, documented, and conform to standard Python conventions.
* **Task 5.2: Data Visualization**
    * Generate phase-space plots comparing the ground-truth attractors vs. the MCTS-discovered attractors.
    * Generate convergence graphs showing how the MCTS reward scales over time with and without SHUSS.
* **Task 5.3: Final Write-up**
    * Draft the methodology, highlighting the theoretical bridge between tree search and sparse regression.
    * Detail the limitations and propose future research directions (e.g., replacing SHUSS with a deeply learned neural policy).