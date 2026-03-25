import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
import os

# Define the Simple Harmonic Oscillator with a weak damping factor of 0.1
# dx/dt = -0.1*x + y
# dy/dt = -x - 0.1*y
def harmonic_oscillator(z, t):
    return [-0.1*z[0] + z[1], -z[0] - 0.1*z[1]]

# Generate training data
dt = 0.01
t_train = np.arange(0, 10, dt)
z0 = [1.0, 0.0]  # Initial conditions
z_train = odeint(harmonic_oscillator, z0, t_train)

# Add some noise
np.random.seed(42)
z_train_noisy = z_train + np.random.normal(0, 0.01, z_train.shape)

# Fit pySINDy model
start_time = time.time()

# We specify polynomial library up to degree 2
poly_library = ps.PolynomialLibrary(degree=2)
# We set threshold to 0.05 so it doesn't zero out the 0.1 damping coefficients
optimizer = ps.STLSQ(threshold=0.05)
model = ps.SINDy(feature_library=poly_library, optimizer=optimizer)

model.fit(z_train_noisy, t=dt)

end_time = time.time()
fit_time = end_time - start_time

print("--- Easy Benchmark: Damped Harmonic Oscillator ---")
print(f"library: {poly_library.get_feature_names()}")
print(f"optimizer: {optimizer}")
print(f"number of data points: {len(z_train)}")
print(f"Fit time: {fit_time:.4f} seconds")
print("Discovered Equations:")
model.print()

# Evaluate performance (Simulate discovered system)
z_sim = model.simulate(z0, t_train)

mse = np.mean((z_train - z_sim)**2)
print(f"Mean Squared Error (clean vs simulated): {mse:.6e}")

# Visualization
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(z_train[:, 0], z_train[:, 1], label='True trajectory')
plt.plot(z_sim[:, 0], z_sim[:, 1], '--', label='Identified trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase space')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_train, z_train[:, 0], label='x True')
plt.plot(t_train, z_sim[:, 0], '--', label='x Identified')
plt.xlabel('Time (t)')
plt.ylabel('x')
plt.title('x over time')
plt.legend()

plt.tight_layout()

os.makedirs('./MCTS_Equation_Discovery/pysindy_experiments/results', exist_ok=True)
plt.savefig('./MCTS_Equation_Discovery/pysindy_experiments/results/harmonic_oscillator.png')
print("Saved visualization to ./MCTS_Equation_Discovery/pysindy_experiments/results/harmonic_oscillator.png")
