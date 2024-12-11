import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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



# Define the time points (from 0 to 52 weeks)
time_points = range(time_horizon + 1)

# Plot the first simulation path
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(time_points, log_fx_simulations[:, 0], label='Simulation 1', linewidth=2, color='blue')
# Adding labels and title
plt.title('Evolution of Log FX over one year')
plt.xlabel('Weeks')
plt.ylabel('Log FX Rate')
# Show the plot
plt.show()



# Visualize the evolution of log(FX_t)
plt.figure(figsize=(10, 6))
for i in range(min(10, num_simulations)):  # Plot first 10 paths
    plt.plot(range(time_horizon + 1), log_fx_simulations[:, i], alpha=0.7)
plt.title("Evolution of Log FX over one year")
plt.xlabel("Weeks")
plt.ylabel("Log FX Rate")
plt.grid(True)
plt.show()