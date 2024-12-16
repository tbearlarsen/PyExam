import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from Data.Data import x0, init_values, cov_matrix
import pandas as pd
from scipy.stats import lognorm


# Load covariance matrix and initial values
covariance_matrix_path = 'covariance_matrix.xlsx'
init_values_path = 'init_values.xlsx'

cov_matrix = pd.read_excel(covariance_matrix_path, index_col=0)
init_values = pd.read_excel(init_values_path, index_col=0)

# Extract initial values and covariance matrix
x0 = init_values.squeeze().values
cov_matrix = cov_matrix.values

# Define parameters
mu = np.array([0, 0.07 / 52, 0.06 / 52] + [0] * (len(x0) - 3))  # Mean of changes per week
time_horizon = 52  # 1 year with weekly steps
num_simulations = 1000  # Number of Monte Carlo simulations

# Simulate the evolution of X_t
np.random.seed(42)  # For reproducibility
X_t = np.zeros((time_horizon + 1, len(x0), num_simulations))
X_t[0] = x0[:, np.newaxis]

for t in range(1, time_horizon + 1):
    delta_X_t = np.random.multivariate_normal(mu, cov_matrix, num_simulations).T
    X_t[t] = X_t[t-1] + delta_X_t

# Extract log(FX_t) from the simulations (first variable in X_t)
log_FX_t = X_t[:, 0, :]




# Step 1: Extract log(V_US,local)^0 from the initial values (it's the second entry in x0)
log_V_US_local_0 = x0[1]

# Step 2: Extract the log(V_US,local)^1 for all simulations (it's the second row in X_t at t=1)
log_V_US_local_1 = X_t[time_horizon, 1, :]  # Use t=52 (since weekly steps for 1 year)

# Step 3: Calculate V_US,local^1 from log(V_US,local)^1
V_US_local_1 = np.exp(log_V_US_local_1)

# Step 4: Plot the distribution of V_US,local^1 (histogram)
plt.figure(figsize=(10, 6))
plt.hist(V_US_local_1, bins=50, color='skyblue', edgecolor='black', density=True)
plt.xlabel('V_US,local^1')
plt.ylabel('Density')
plt.title('Distribution of V_US,local^1 at t=1 (USD)')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()

# Step 5: Calculate summary statistics
V_US_local_1_mean = np.mean(V_US_local_1)
V_US_local_1_std = np.std(V_US_local_1)

V_US_local_1_mean, V_US_local_1_std



# Step 1: Analytical parameters for log(V_US,local)^1
mu_delta_log_V = 0.07 * (1 / 52)  # Mean change in log(V_US,local) per week
sigma_delta_log_V = np.sqrt(cov_matrix[1, 1])  # Standard deviation for second variable

# Total mean and variance for log(V_US,local)^1
mu_log_V_US_local_1 = log_V_US_local_0 + 52 * mu_delta_log_V # ANALYTICAL MEAN, needs conversion to log-normal mean
sigma_log_V_US_local_1 = np.sqrt(52) * sigma_delta_log_V # ANALYTICAL STD, needs conversion to log-normal std.

# Step 2: Log-normal distribution parameters for V_US,local^1
mean_V_US_local_1 = np.exp(mu_log_V_US_local_1 + (sigma_log_V_US_local_1**2) / 2)
var_V_US_local_1 = (np.exp(sigma_log_V_US_local_1*2) - 1) * np.exp(2 * mu_log_V_US_local_1 + sigma_log_V_US_local_1*2)
std_V_US_local_1 = np.sqrt(var_V_US_local_1)

# Step 3: Plot simulated distribution of V_US,local^1 and overlay analytical PDF
x_vals = np.linspace(np.min(V_US_local_1), np.max(V_US_local_1), 1000)
pdf_analytical = lognorm.pdf(x_vals, s=sigma_log_V_US_local_1, scale=np.exp(mu_log_V_US_local_1))

plt.figure(figsize=(10, 6))
# Simulated histogram
plt.hist(V_US_local_1, bins=50, color='skyblue', edgecolor='black', density=True, label='Simulated')
# Analytical PDF
plt.plot(x_vals, pdf_analytical, 'r-', lw=2, label='Analytical PDF (Log-normal)')
plt.xlabel('V_US,local^1')
plt.ylabel('Density')
plt.title('Simulated vs Analytical Distribution of V_US,local^1 at t=1 (USD)')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()

# Print analytical mean and standard deviation
mean_V_US_local_1, std_V_US_local_1