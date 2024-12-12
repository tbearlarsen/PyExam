import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data.Data import covariance_matrix, init_values, cov_matrix, x0, mean_vector, delta_t
from scipy.stats import norm, lognorm
from Questions.Q4 import mean_log_FX, var_log_FX, mean_log_V_US, var_log_V_US, mean_Z_US_4Y, var_Z_US_4Y



# Mean and variance for P1^{EUR}
# Reciprocal FX
mean_1_FX = np.exp(-mean_log_FX + var_log_FX)
var_1_FX = (np.exp(var_log_FX) - 1) * np.exp(-2 * mean_log_FX + var_log_FX)

# V1_US in EUR
mean_V1_US_EUR = np.exp(mean_log_V_US) * mean_1_FX
var_V1_US_EUR = (np.exp(var_log_V_US) - 1) * np.exp(2 * mean_log_V_US + var_log_V_US) + var_1_FX * mean_V1_US_EUR**2

# Z1_US_4Y in EUR
mean_Z1_US_4Y_EUR = np.exp(mean_Z_US_4Y) * mean_1_FX
var_Z1_US_4Y_EUR = (np.exp(var_Z_US_4Y) - 1) * np.exp(2 * mean_Z_US_4Y + var_Z_US_4Y) + var_1_FX * mean_Z1_US_4Y_EUR**2

# Analytical approximations for lognormal PDF
sigma_V1_US_EUR = np.sqrt(var_V1_US_EUR)
mu_V1_US_EUR = np.log(mean_V1_US_EUR) - 0.5 * sigma_V1_US_EUR**2


# Plot comparison between simulation and analytical PDF
# Simulated data (placeholder for demonstration)
V1_US_EUR_simulated = np.random.lognormal(mean=mu_V1_US_EUR, sigma=sigma_V1_US_EUR, size=10000)

plt.figure(figsize=(10, 6))

# Simulated histogram
plt.hist(V1_US_EUR_simulated, bins=50, density=True, alpha=0.5, label='Simulated V1_US (EUR)')

# Analytical PDF
x = np.linspace(min(V1_US_EUR_simulated), max(V1_US_EUR_simulated), 500)
pdf = lognorm.pdf(x, s=sigma_V1_US_EUR, scale=np.exp(mu_V1_US_EUR))
plt.plot(x, pdf, label='Analytical PDF', color='red')

# Labels and legend
plt.title('Comparison of Simulated and Analytical Distributions for V1_US (EUR)')
plt.xlabel('V1_US (EUR)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
