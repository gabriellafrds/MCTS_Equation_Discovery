import numpy as np
import pysindy as ps
from main import generate_data_harmonic_oscillator

data = generate_data_harmonic_oscillator()
y_dot = data["y_dot"]
x = data["variables"]["x"]
y = data["variables"]["y"]

# Matrix with just x
Theta_x = np.column_stack([x])
opt_x = ps.STLSQ(threshold=0.05, alpha=0.05)
opt_x.fit(Theta_x, y_dot)
mse_x = np.mean((opt_x.predict(Theta_x) - y_dot)**2)
l0_x = np.count_nonzero(opt_x.coef_)
bic_x = 1000 * np.log(max(mse_x, 1e-12)) + l0_x * np.log(1000)

print(f"Bic Just X: {bic_x:.2f}, MSE: {mse_x:.6e}")

# Matrix with x and y
Theta_xy = np.column_stack([x, y])
opt_xy = ps.STLSQ(threshold=0.05, alpha=0.05)
opt_xy.fit(Theta_xy, y_dot)
mse_xy = np.mean((opt_xy.predict(Theta_xy) - y_dot)**2)
l0_xy = np.count_nonzero(opt_xy.coef_)
bic_xy = 1000 * np.log(max(mse_xy, 1e-12)) + l0_xy * np.log(1000)

print(f"Bic X and Y: {bic_xy:.2f}, MSE: {mse_xy:.6e}")
print(f"Coefs: {opt_xy.coef_}")
