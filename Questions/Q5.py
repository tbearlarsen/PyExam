import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data.Data import covariance_matrix, init_values, cov_matrix, x0, mean_vector, delta_t
from scipy.stats import norm, lognorm
import seaborn as sns
from Questions.Q4 import FX_1, V1_US_local, Z1_US_4Y, V1_EUR, Z1_EUR_4Y


# Transform to EUR
FX_1_inverse = 1 / FX_1
V1_US = V1_US_local * FX_1_inverse
Z1_US_4Y_EUR = Z1_US_4Y * FX_1_inverse

# Combine into the joint vector P1^EUR
P1_EUR = np.column_stack((FX_1_inverse, V1_US, V1_EUR, Z1_US_4Y_EUR, Z1_EUR_4Y))

# Compare the simulated and analytical distributions for V1_US
# Simulated distribution
plt.figure(figsize=(10, 6))
plt.hist(V1_US, bins=50, alpha=0.5, density=True, label="Simulated V1_US (EUR)")

# Analytical approximation
mu_v_us_local = np.log(np.mean(V1_US_local))
sigma_v_us_local = np.std(np.log(V1_US_local))
mu_fx = np.log(np.mean(FX_1))
sigma_fx = np.std(np.log(FX_1))

mu_analytical = mu_v_us_local - mu_fx  # Adjust mean for inverse FX
sigma_analytical = np.sqrt(sigma_v_us_local**2 + sigma_fx**2)  # Variance addition

x = np.linspace(V1_US.min(), V1_US.max(), 500)
pdf = lognorm.pdf(x, s=sigma_analytical, scale=np.exp(mu_analytical))
plt.plot(x, pdf, label="Analytical Approximation", color="red")

# Labels and legend
plt.title("Simulated vs Analytical Distribution for V1_US (EUR)")
plt.xlabel("V1_US (EUR)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()













