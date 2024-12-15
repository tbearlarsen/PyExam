import numpy as np
import matplotlib.pyplot as plt
from Data.Data import covariance_matrix, init_values, cov_matrix, x0, mean_vector, delta_t
from scipy.stats import norm, lognorm

# Parameters and simulation setup
v_us_index = 1
mean_vector = np.array([0.07 * delta_t, 0.06 * delta_t] + [0] * (len(covariance_matrix) - 2))
num_simulations = 10000

#Simulate the distribution of V1^US,local
np.random.seed(42)
simulated_shocks = np.random.multivariate_normal(mean_vector, cov_matrix, num_simulations)
simulated_v1_us_local = x0[v_us_index] + simulated_shocks[:, v_us_index]
#simulated_v1_us_local = np.exp(x0[v_us_index] + simulated_shocks[:, v_us_index])

# Analytical distribution of V1^US,local
mean_v1_us_local = x0[v_us_index] + mean_vector[v_us_index]
#mean_v1_us_local = np.exp(x0[v_us_index] + mean_vector[v_us_index])
std_v1_us_local = np.sqrt(cov_matrix[v_us_index, v_us_index])

# Plot the simulated and analytical distributions
plt.figure(figsize=(10, 6))

# Simulated distribution
plt.hist(simulated_v1_us_local, bins=50, alpha=0.5, label="Simulated Distribution", density=True)

# Analytical normal distribution
x = np.linspace(mean_v1_us_local - 4 * std_v1_us_local, mean_v1_us_local + 4 * std_v1_us_local, 500) #generates the linearly spaced numbers over which the PDF will be calculated.
pdf = norm.pdf(x, loc=mean_v1_us_local, scale=std_v1_us_local) #calculates the PDF
#pdf = lognorm.pdf(x, s=std_v1_us_local, scale=mean_v1_us_local)
plt.plot(x, pdf, label="Analytical Distribution", color='red')

# Labels and legend
plt.title("Comparison of Simulated and Analytical Distributions for V1^US,local")
plt.xlabel("V1^US,local")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

