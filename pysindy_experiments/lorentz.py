import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
import os

# Define the Lorenz System
# dx/dt = sigma * (y - x)
# dy/dt = x * (rho - z) - y
# dz/dt = x * y - beta * z

sigma = 10.0
rho = 28.0
beta = 8.0/3.0

def lorenz(z, t):
    return [
        sigma * (z[1] - z[0]),
        z[0] * (rho - z[2]) - z[1],
        z[0] * z[1] - beta * z[2]
    ]

# Generate training data
dt = 0.002
t_train = np.arange(0, 10, dt)
z0 = [-8.0, 8.0, 27.0]  # Initial conditions
z_train = odeint(lorenz, z0, t_train)

# Add some noise
np.random.seed(42)
z_train_noisy = z_train + np.random.normal(0, 0.1, z_train.shape)

# Fit pySINDy model
start_time = time.time()

# We specify polynomial library up to degree 3, as the true system has degree 2
poly_library = ps.PolynomialLibrary(degree=3)
# To promote sparsity, we increase the threshold for STLSQ slightly 
optimizer = ps.STLSQ(threshold=0.1)
model = ps.SINDy(feature_library=poly_library, optimizer=optimizer)

model.fit(z_train_noisy, t=dt)

end_time = time.time()
fit_time = end_time - start_time

print("--- Hard Benchmark: Lorenz System ---")
print(f"Fit time: {fit_time:.4f} seconds")
print("Discovered Equations:")
model.print()

# Evaluate performance (Simulate discovered system)
# Warning: chaotic systems diverge quickly, so we expect MSE to grow large over time
# We'll simulate just to see if the attractor shape is captured.
try:
    z_sim = model.simulate(z0, t_train)
    mse = np.mean((z_train - z_sim)**2)
    print(f"Mean Squared Error (clean vs simulated): {mse:.6e}")

    # Visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(221, projection='3d')
    ax.plot(z_train[:, 0], z_train[:, 1], z_train[:, 2], label='True trajectory', lw=0.5)
    ax.plot(z_sim[:, 0], z_sim[:, 1], z_sim[:, 2], '--', label='Identified trajectory', lw=0.5)
    ax.set_title('Lorenz Attractor')
    ax.legend()

    ax2 = fig.add_subplot(222)
    ax2.plot(t_train[:1000], z_train[:1000, 0], label='x True')
    ax2.plot(t_train[:1000], z_sim[:1000, 0], '--', label='x Identified')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('x')
    ax2.set_title('x over time (first 1000 steps)')
    ax2.legend()

    ax3 = fig.add_subplot(223)
    ax3.plot(t_train[:1000], z_train[:1000, 1], label='y True')
    ax3.plot(t_train[:1000], z_sim[:1000, 1], '--', label='y Identified')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('y')
    ax3.set_title('y over time (first 1000 steps)')
    ax3.legend()

    ax4 = fig.add_subplot(224)
    ax4.plot(t_train[:1000], z_train[:1000, 2], label='z True')
    ax4.plot(t_train[:1000], z_sim[:1000, 2], '--', label='z Identified')
    ax4.set_xlabel('Time (t)')
    ax4.set_ylabel('z')
    ax4.set_title('z over time (first 1000 steps)')
    ax4.legend()

    plt.tight_layout()
    os.makedirs('./MCTS_Equation_Discovery/pysindy_experiments/results', exist_ok=True)
    plt.savefig('./MCTS_Equation_Discovery/pysindy_experiments/results/lorentz.png')
    print("Saved visualization to ./MCTS_Equation_Discovery/pysindy_experiments/results/lorentz.png")
except Exception as e:
    print("Simulation failed (likely exploded due to numerical instability):", e)
