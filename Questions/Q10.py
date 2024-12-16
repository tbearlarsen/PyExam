import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cvxpy as cp
from Questions.Q6 import E_pnl1, cov_pnl1, fx0
from Questions.Q7 import optimal_hr

#STRATEGY 1:
# Input Data
means = E_pnl1
cov_matrix = cov_pnl1
hedge_ratios = optimal_hr[0:2]

num_assets = len(means)

# Pre-specified hedge ratios
fx_hedge_bond = hedge_ratios[1]  # Hedge ratio for asset 3
fx_hedge_equity = hedge_ratios[0]  # Hedge ratio for asset 1

# Generate return targets
mu_targets = np.linspace(min(means), max(means), 500)

# Define CVXPY variables and constraints
weights = cp.Variable(num_assets)

constraints = [
    cp.sum(weights[1:]) == 1,  # Sum of weights excluding forward contract
    weights[1:] >= 0,  # No shorting for assets
    weights[0] == fx_hedge_equity * weights[1] + fx_hedge_bond * weights[3],  # Hedge ratio constraint
]

# Storage for results
portfolio_weights = []
portfolio_variances = []

# Solve for each target return
for pnl_target in mu_targets:
    # Add target return constraint
    constraints.append(weights @ means == pnl_target)

    # Define the optimization problem (minimize portfolio variance)
    objective = cp.Minimize(cp.quad_form(weights, cov_matrix))
    problem = cp.Problem(objective, constraints)

    try:
        # Solve the optimization problem
        problem.solve(solver=cp.SCS)
        if weights.value is not None:
            portfolio_weights.append(weights.value)
            portfolio_variances.append(weights.value @ cov_matrix @ weights.value)
    except Exception as e:
        print(f"Optimization failed for pnl_target={pnl_target}: {e}")

    # Remove target return constraint for next iteration
    constraints.pop()

# Convert results to arrays for further analysis
portfolio_weights = np.array(portfolio_weights)
portfolio_std = np.sqrt(portfolio_variances)
portfolio_returns = portfolio_weights @ means

# Identify key portfolios
min_var_index = np.argmin(portfolio_variances)
max_sharpe_index = np.argmax(portfolio_returns / portfolio_std)

# Visualization
plt.figure(figsize=(10, 6))

# Plot efficient frontier
plt.plot(portfolio_std, portfolio_returns, label='Efficient Frontier', color='blue', linewidth=2)

# Individual assets and forward contract
for i, (mean, std) in enumerate(zip(means, np.sqrt(np.diag(cov_matrix)))):
    label = 'Forward' if i == 0 else f'Asset {i}'
    plt.scatter(std, mean, label=label, zorder=5)

# Highlight minimum variance and maximum Sharpe portfolios
plt.scatter(portfolio_std[min_var_index], portfolio_returns[min_var_index],
            color='black', marker='o', s=150, label='Min Variance Portfolio', edgecolors='white', zorder=6)
plt.scatter(portfolio_std[max_sharpe_index], portfolio_returns[max_sharpe_index],
            color='gold', marker='X', s=150, label='Max Sharpe Portfolio', edgecolors='black', zorder=6)

# Add titles, labels, legend, and grid
plt.title('Refined Mean-Variance Efficient Frontier with Forward Contract')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Return')
plt.legend()
plt.grid()
plt.show()

# Summarize portfolio results
portfolio_summaries = [
    {
        'Portfolio Type': 'Min Variance',
        'Risk': portfolio_std[min_var_index],
        'Return': portfolio_returns[min_var_index],
        'Weights': portfolio_weights[min_var_index]
    },
    {
        'Portfolio Type': 'Max Sharpe',
        'Risk': portfolio_std[max_sharpe_index],
        'Return': portfolio_returns[max_sharpe_index],
        'Weights': portfolio_weights[max_sharpe_index]
    }
]

# Display results
portfolio_summary_df = pd.DataFrame(portfolio_summaries)
print(portfolio_summary_df)


#STRATEGY 2:
# Given data
means_no_forward = np.array([0.0769594, 0.0747646, 0.02660308, 0.01919606])
cov_matrix_no_forward = np.array([
    [0.025256, 0.019484, 0.002351, 0.000531],
    [0.019484, 0.027958, -0.000440, 0.000355],
    [0.002351, -0.000440, 0.003637, 0.000324],
    [0.000531, 0.000355, 0.000324, 0.000494],
])
n = len(means_no_forward)

# Variables
weights = cp.Variable(n)
budget_constraint = cp.sum(weights) == 1
non_shorting_constraint = weights >= 0

# Efficient frontier calculation with 1000 PnL targets
PnL_targets = np.linspace(min(means_no_forward), max(means_no_forward), 1000)
optimal_weights = []

for target in PnL_targets:
    objective = cp.Minimize(cp.quad_form(weights, cov_matrix_no_forward))
    constraints = [
        budget_constraint,
        non_shorting_constraint,
        means_no_forward @ weights >= target
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    optimal_weights.append(weights.value)

h2 = np.array(optimal_weights)
means = np.array([-0.01522965, 0.0769594, 0.0747646, 0.02660308, 0.01919606])
data = {
    0: [0.005220, -0.003732, -0.000248, -0.004000, 0.000071],
    1: [-0.003732, 0.025256, 0.019484, 0.002351, 0.000531],
    2: [-0.000248, 0.019484, 0.027958, -0.000440, 0.000355],
    3: [-0.004000, 0.002351, -0.000440, 0.003637, 0.000324],
    4: [0.000071, 0.000531, 0.000355, 0.000324, 0.000494]
}
cov_matrix = pd.DataFrame(data)

block_11 = cov_matrix.iloc[:1, :1].values
block_12 = cov_matrix.iloc[:1, 1:].values
block_22 = cov_matrix.iloc[1:, 1:].values

h1_values = [-np.dot(block_12, h2_vector) / block_11 for h2_vector in h2]
h1_values = np.array([h1.item() for h1 in h1_values])

combined_array = np.hstack((h1_values.reshape(-1, 1), h2))

# Optimization using combined weights and visualization of the efficient frontier
PnL_values = []
variances = []

for weights in combined_array:
    pnl = np.dot(means, weights)
    variance = np.dot(weights, np.dot(cov_matrix.values, weights))
    PnL_values.append(pnl)
    variances.append(variance)

# Plotting the efficient frontier with asset labels
plt.figure(figsize=(12, 7))
plt.plot(variances, PnL_values, label="Efficient Frontier")
plt.xlabel("Portfolio Variance")
plt.ylabel("Portfolio PnL")
plt.title("Efficient Frontier")

# Annotating assets using their means and variances from the covariance matrix
asset_labels = ["Forward", "Asset 1", "Asset 2", "Asset 3", "Asset 4"]
variances_assets = np.diag(cov_matrix.values)  # Variances of individual assets

for i, label in enumerate(asset_labels):
    plt.scatter(variances_assets[i], means[i], label=label, s=50)  # Scatter the individual asset points

plt.legend()
plt.grid()
plt.show()


#STRATEGY 3:
# Data input
means = E_pnl1
cov_matrix = cov_pnl1
forward_rate = fx0
budget = 1  # in EUR

# Convert USD assets to EUR using the forward exchange rate
conversion_factors = np.array([1, 1, 1 / forward_rate, 1, 1 / forward_rate])
means_eur = means * conversion_factors

# Covariance matrix adjusted for currency conversion
cov_matrix_eur = cov_matrix * conversion_factors[:, None] * conversion_factors[None, :]


# Define optimization functions
def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights


def portfolio_return(weights, means):
    return weights.T @ means


# Efficient frontier calculation
fine_targets = np.linspace(min(means_eur), max(means_eur), 500)  # Finer granularity
results_refined = []

for target in fine_targets:
    constraints = [
        {'type': 'eq', 'fun': lambda w: portfolio_return(w, means_eur) - target},  # Target return
        {'type': 'eq', 'fun': lambda w: np.sum(w[1:]) - budget},  # Budget constraint (EUR assets)
        {'type': 'ineq', 'fun': lambda w: w[1:]},  # No shorting for assets
    ]
    bounds = [(None, None)] + [(0, None)] * 4  # No bounds on forward, no shorting on assets
    initial_weights = np.zeros(5)  # Start with zero weights

    result = minimize(
        portfolio_variance,
        initial_weights,
        args=(cov_matrix_eur,),
        constraints=constraints,
        bounds=bounds,
        method='SLSQP',  # Use a solver optimized for smoothness
        options={'ftol': 1e-9}  # High precision
    )
    if result.success:
        variance = result.fun
        weights = result.x
        results_refined.append((target, np.sqrt(variance), weights))

# Extract refined results for plotting
efficient_frontier_refined = pd.DataFrame(results_refined, columns=['Return', 'Risk', 'Weights'])

# Visualization
plt.figure(figsize=(10, 6))

# Refined efficient frontier
plt.plot(efficient_frontier_refined['Risk'], efficient_frontier_refined['Return'], label='Efficient Frontier (Refined)',
         color='blue')

# Individual assets and forward contract
for i, (mean, std) in enumerate(zip(means_eur, np.sqrt(np.diag(cov_matrix_eur)))):
    label = 'Forward' if i == 0 else f'Asset {i}'
    plt.scatter(std, mean, label=label)

plt.title('Refined Mean-Variance Efficient Frontier with Forward Contract')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Return')
plt.legend()
plt.grid()
plt.show()