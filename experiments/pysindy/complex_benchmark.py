import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
import os

# Define a complex system with non-polynomial terms
# dx/dt = -0.5 * x + sin(y)
# dy/dt = -0.5 * y + cos(x) + exp(-x^2)

def complex_system(z, t):
    x, y = z
    dx = -0.5 * x + np.sin(y)
    dy = -0.5 * y + np.cos(x) + np.exp(-(x**2))
    return [dx, dy]

# Generate training data
dt = 0.05
t_train = np.arange(0, 15, dt)
z0 = [2.0, -2.0]  # Initial conditions
z_train = odeint(complex_system, z0, t_train)

# Add minimal noise
np.random.seed(42)
z_train_noisy = z_train + np.random.normal(0, 0.005, z_train.shape)

# Custom Feature Library containing all the terms we would like to test + extra terms for higher difficulty:
# x, y, x^2, y^2, sin, cos, exp, 1/(1+x^2), x^3, y^3, sin(x), cos(y), exp(-x^2), 1/(1+y^2)
library_functions = [
    lambda x: x,
    lambda x: x**2,
    lambda x: np.sin(x),
    lambda x: np.cos(x),
    lambda x: np.exp(-(x**2)),
    lambda x: 1 / (1 + x**2),
    lambda x: x**3
]
function_names = [
    lambda x: x,
    lambda x: x + "^2",
    lambda x: "sin(" + x + ")",
    lambda x: "cos(" + x + ")",
    lambda x: "exp(-" + x + "^2)",
    lambda x: "1/(1+" + x + "^2)",
    lambda x: x + "^3"
]

custom_library = ps.CustomLibrary(
    library_functions=library_functions,
    function_names=function_names
)

# Fit pySINDy model
from sklearn.linear_model import Lasso
start_time = time.time()

# We set threshold low enough to capture 0.5 coefficients 
optimizer = ps.STLSQ(threshold=0.1)
model = ps.SINDy(feature_library=custom_library, optimizer=optimizer)

model.fit(z_train_noisy, t=dt)

end_time = time.time()
fit_time = end_time - start_time

print("--- Complex Benchmark: Non-Polynomial System ---")
print(f"Fit time: {fit_time:.4f} seconds")
print("True Equations:")
print("x0' = -0.5 x0 + sin(x1)")
print("x1' = -0.5 x1 + cos(x0) + exp(-x0^2)")
print("\nDiscovered Equations:")
model.print()

# Simulate
try:
    z_sim = model.simulate(z0, t_train)
    mse = np.mean((z_train - z_sim)**2)
    print(f"\nMean Squared Error (clean vs simulated): {mse:.6e}")
    
    # Visualization
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(231)
    ax1.plot(z_train[:, 0], z_train[:, 1], label='True phase')
    ax1.plot(z_sim[:, 0], z_sim[:, 1], '--', label='Identified phase')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Phase space')
    ax1.legend()

    ax2 = fig.add_subplot(232)
    ax2.plot(t_train, z_train[:, 0], label='x True')
    ax2.plot(t_train, z_sim[:, 0], '--', label='x Identified')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('x')
    ax2.set_title('x over time')
    ax2.legend()

    ax3 = fig.add_subplot(233)
    ax3.plot(t_train, z_train[:, 1], label='y True')
    ax3.plot(t_train, z_sim[:, 1], '--', label='y Identified')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('y')
    ax3.set_title('y over time')
    ax3.legend()

    # visualisation of the weights of the model
    ax4 = fig.add_subplot(234)
    ax4.bar(model.get_feature_names(), model.coefficients()[0])
    ax4.set_xlabel('Features')
    ax4.set_ylabel('Coefficients')
    ax4.set_title('Model coefficients (x)')
    ax4.tick_params(axis='x', rotation=45)

    ax5 = fig.add_subplot(235)
    ax5.bar(model.get_feature_names(), model.coefficients()[1])
    ax5.set_xlabel('Features')
    ax5.set_ylabel('Coefficients')
    ax5.set_title('Model coefficients (y)')
    ax5.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    os.makedirs('./MCTS_Equation_Discovery/pysindy_experiments/results', exist_ok=True)
    plt.savefig('./MCTS_Equation_Discovery/pysindy_experiments/results/complex_benchmark.png')
    print("Saved visualization to ./MCTS_Equation_Discovery/pysindy_experiments/results/complex_benchmark.png")
except Exception as e:
    print("Simulation failed:", e)
