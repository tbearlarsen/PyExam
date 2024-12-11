import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data.Data import covariance_matrix, init_values, cov_matrix, x0, mean_vector, delta_t
from scipy.stats import norm, lognorm

# Parameters
initial_yield_index = 6  # Assuming the 5-year yield is at index 4 in the state vector
tau_initial = 5  # Initial maturity
tau_horizon = 4  # Maturity at horizon
num_simulations = 10000

# Initial values and parameters
initial_yield = x0[initial_yield_index]
mu_yield = mean_vector[initial_yield_index]
sigma_yield = np.sqrt(cov_matrix[initial_yield_index, initial_yield_index])

# Simulate weekly changes in yield
np.random.seed(42)
simulated_yields = np.random.normal(mu_yield, sigma_yield, size=num_simulations)

# Calculate bond prices at horizon
prices_horizon = np.exp(-simulated_yields * tau_horizon)

# Analytical distribution parameters
mu_price = -tau_horizon * mu_yield
sigma_price = np.sqrt(tau_horizon**2 * sigma_yield**2)

# Plot simulated and analytical distributions
plt.figure(figsize=(10, 6))

# Simulated distribution
plt.hist(prices_horizon, bins=50, alpha=0.5, label="Simulated Distribution", density=True)

# Analytical lognormal distribution
x = np.linspace(min(prices_horizon), max(prices_horizon), 500)
pdf = lognorm.pdf(x, s=sigma_price, scale=np.exp(mu_price))
plt.plot(x, pdf, label="Analytical Distribution", color='red')

# Labels and legend
plt.title("Comparison of Simulated and Analytical Distributions for Zero Coupon Bond")
plt.xlabel("Price of Zero Coupon Bond at Horizon")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
