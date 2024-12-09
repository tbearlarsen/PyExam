import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Data.Data import data_cov,data_init

# Load data from the provided files
covariance_matrix = data_cov
init_values = data_init

# Simulation parameters
time_horizon = 52  # Weekly steps (1 year)
delta_t = 1 / 52   # Time step (1 week)
mean_vector = np.array([0.07 * delta_t, 0.06 * delta_t] + [0] * (len(covariance_matrix) - 2))  # μ
cov_matrix = covariance_matrix.to_numpy()  # Σ
num_simulations = 10000  # Number of paths

# Initial values
x0 = init_values["initial values"].to_numpy()

# Simulating X_t evolution
np.random.seed(42)  # For reproducibility
simulations = np.zeros((time_horizon + 1, len(x0), num_simulations))
simulations[0] = x0[:, None]  # Set initial values

for t in range(1, time_horizon + 1):
    # Generate random shocks
    shocks = np.random.multivariate_normal(mean_vector, cov_matrix, num_simulations).T
    # Update simulations for time t
    simulations[t] = simulations[t - 1] + shocks

# Extract log FX evolution
log_fx_simulations = simulations[:, 0, :]  # log(FX_t) is the first variable

# Visualize the evolution of log(FX_t)
plt.figure(figsize=(10, 6))
for i in range(min(10, num_simulations)):  # Plot first 10 paths
    plt.plot(range(time_horizon + 1), log_fx_simulations[:, i], alpha=0.7)
plt.title("Evolution of log(FX_t) over Time")
plt.xlabel("Time (weeks)")
plt.ylabel("log(FX_t)")
plt.grid(True)
plt.show()

X1=simulations[1]



import numpy as np
import matplotlib.pyplot as plt

# Parameters and simulation setup
v_us_index = 1  # Assuming V1^US,local is the second variable in the covariance matrix
time_horizon = 1  # One time step (week)
num_simulations = 1000  # Number of simulations
mean_vector = np.array([0.07 * delta_t, 0.06 * delta_t] + [0] * (len(covariance_matrix) - 2))  # μ
cov_matrix = covariance_matrix.to_numpy()  # Covariance matrix
x0 = init_values["initial values"].to_numpy()  # Initial values

# Simulate the distribution of V1^US,local
np.random.seed(42)  # For reproducibility
simulated_shocks = np.random.multivariate_normal(mean_vector, cov_matrix, num_simulations)
simulated_v1_us_local = x0[v_us_index] + simulated_shocks[:, v_us_index]

# Analytical distribution of V1^US,local
mean_v1_us_local = x0[v_us_index] + mean_vector[v_us_index]
std_v1_us_local = np.sqrt(cov_matrix[v_us_index, v_us_index])

# Plot the simulated and analytical distributions
plt.figure(figsize=(10, 6))

# Simulated distribution
plt.hist(simulated_v1_us_local, bins=50, alpha=0.5, label="Simulated Distribution", density=True)

# Analytical normal distribution
x = np.linspace(mean_v1_us_local - 4 * std_v1_us_local, mean_v1_us_local + 4 * std_v1_us_local, 500)
pdf = (1 / (std_v1_us_local * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_v1_us_local) / std_v1_us_local) ** 2)
plt.plot(x, pdf, label="Analytical Distribution", color='red')

# Labels and legend
plt.title("Comparison of Simulated and Analytical Distributions for V1^US,local")
plt.xlabel("V1^US,local")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()


