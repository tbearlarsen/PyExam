import numpy as np
import matplotlib.pyplot as plt
from Questions.Q7 import E_pnl1, cov_pnl1, fx0
from scipy.stats import multivariate_normal

# True distributional parameters (from Q6 or given)
true_mu = E_pnl1  # Fixed expected PnL vector
true_sigma = cov_pnl1  # True covariance matrix

# Simulation settings
num_simulations = 1000
sample_size = 500  # Two years of daily data

# Arrays to store results
optimal_hedge_ratios = []
expected_pnls = []
std_devs = []
cvars = []


# Function to calculate CVaR
def calculate_cvar(port_returns, beta=0.95):
    perc = np.percentile(port_returns, (1 - beta) * 100)
    avg_tail = np.mean(port_returns[port_returns <= perc])
    return -avg_tail


# Function to compute metrics for a given hedge ratio
def compute_metrics(h, mu, sigma):
    expected_pnl = np.dot(h, mu)
    variance_pnl = np.dot(h.T, np.dot(sigma, h))
    std_dev = np.sqrt(variance_pnl)
    return expected_pnl, std_dev


# Simulation loop
for _ in range(num_simulations):
    # Step 1: Simulate market invariants
    simulated_data = np.random.multivariate_normal(true_mu, true_sigma, sample_size)

    # Step 2: Estimate covariance matrix
    sample_sigma = np.cov(simulated_data, rowvar=False)

    # Step 3: Calculate optimal hedge ratio for Portfolio C
    h2_C = np.array([0.2, 0.2, 0.3, 0.3])
    usd_exposure_C = (h2_C[0] + h2_C[2]) * fx0  # Use fx0 from Q7
    sigma_11 = sample_sigma[0, 0]
    sigma_12 = sample_sigma[0, 1:]
    optimal_h1 = -np.dot(sigma_12, h2_C) / sigma_11
    optimal_hr = optimal_h1 / usd_exposure_C

    # Combine hedge ratio with h2_C
    h = np.concatenate(([optimal_h1], h2_C))

    # Step 4: Evaluate using true distribution
    # Simulate portfolio returns using true parameters
    portfolio_returns = np.random.multivariate_normal(true_mu, true_sigma, 10000) @ h

    # Calculate portfolio metrics
    expected_pnl = np.mean(portfolio_returns)
    std_dev = np.std(portfolio_returns)
    cvar = calculate_cvar(portfolio_returns, beta=0.95)

    # Store results
    optimal_hedge_ratios.append(optimal_hr)
    expected_pnls.append(expected_pnl)
    std_devs.append(std_dev)
    cvars.append(cvar)

# Convert results to arrays
optimal_hedge_ratios = np.array(optimal_hedge_ratios)
expected_pnls = np.array(expected_pnls)
std_devs = np.array(std_devs)
cvars = np.array(cvars)

# Visualization
plt.figure(figsize=(16, 8))

# Histogram of optimal hedge ratios
plt.subplot(2, 2, 1)
plt.hist(optimal_hedge_ratios, bins=50, alpha=0.7, color='blue')
plt.title('Distribution of Optimal Hedge Ratios')
plt.xlabel('Hedge Ratio')
plt.ylabel('Frequency')

# Histogram of expected PnL
plt.subplot(2, 2, 2)
plt.hist(expected_pnls, bins=50, alpha=0.7, color='green')
plt.title('Distribution of Expected PnL')
plt.xlabel('Expected PnL')
plt.ylabel('Frequency')

# Histogram of standard deviation
plt.subplot(2, 2, 3)
plt.hist(std_devs, bins=50, alpha=0.7, color='orange')
plt.title('Distribution of Standard Deviation')
plt.xlabel('Standard Deviation')
plt.ylabel('Frequency')

# Histogram of 5% CVaR
plt.subplot(2, 2, 4)
plt.hist(cvars, bins=50, alpha=0.7, color='red')
plt.title('Distribution of 5% CVaR')
plt.xlabel('5% CVaR')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()












