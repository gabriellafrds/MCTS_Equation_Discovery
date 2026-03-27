import numpy as np
import pysindy as ps
from main import generate_data_harmonic_oscillator

data = generate_data_harmonic_oscillator()
y_dot = data["y_dot"]
x = data["variables"]["x"]
y = data["variables"]["y"]

true_y_dot = -0.1 * x + y
mse_true = np.mean((true_y_dot - y_dot)**2)
print(f"MSE of true equation: {mse_true:.6e}")

Theta_xy = np.column_stack([x, y])
opt_xy = ps.STLSQ(threshold=0.05, alpha=0.05)
opt_xy.fit(Theta_xy, y_dot)
pred = opt_xy.predict(Theta_xy)

mse_pred = np.mean((pred - y_dot)**2)
print(f"MSE of predict: {mse_pred:.6e}")
print(f"Shape of y_dot: {y_dot.shape}")
print(f"Shape of pred: {pred.shape}")
print(f"y_dot[:5]: {y_dot[:5]}")
print(f"pred[:5]: {pred[:5].flatten()}")
