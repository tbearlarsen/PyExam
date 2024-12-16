import numpy as np
import matplotlib.pyplot as plt
from Data.Data import cov_matrix, x0, mean_vector

#Defining the parameters:
mu=mean_vector
time_horizon = 52
num_simulations = 1000

#Simulating X_t evolution:
np.random.seed(42)
X_t = np.zeros((time_horizon + 1, len(x0), num_simulations))
X_t[0] = x0[:, None]

for t in range(1, time_horizon + 1):
    delta_X_t = np.random.multivariate_normal(mean_vector, cov_matrix, num_simulations).T
    X_t[t] = X_t[t - 1] + delta_X_t

#Extract log FX evolution:
log_fx_simulations = X_t[:, 0, :]


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