import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cvxpy as cp

# Data input
means = np.array([-0.01522965, 0.0769594, 0.0747646, 0.02660308, 0.01919606])
cov_matrix = np.array([
    [0.005220, -0.003732, -0.000248, -0.004000, 0.000071],
    [-0.003732, 0.025256, 0.019484, 0.002351, 0.000531],
    [-0.000248, 0.019484, 0.027958, -0.000440, 0.000355],
    [-0.004000, 0.002351, -0.000440, 0.003637, 0.000324],
    [0.000071, 0.000531, 0.000355, 0.000324, 0.000494]
])
budget = 1  # in EUR

# Efficient frontier calculation
fine_targets = np.linspace(min(means), max(means), 500)  # Finer granularity
results_refined = []

for target in fine_targets:
    # Define cvxpy variables
    weights = cp.Variable(len(means))
    
    # Define objective function (portfolio variance)
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    
    # Define constraints
    constraints = [
        weights[1:] >= 0,  # No shorting on assets
        cp.sum(weights[1:]) == budget,  # Budget constraint
        weights @ means == target  # Target return
    ]
    
    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    problem.solve()
    
    if problem.status == cp.OPTIMAL:
        variance = problem.value
        results_refined.append((target, np.sqrt(variance), weights.value))

# Extract refined results for plotting
efficient_frontier_refined = pd.DataFrame(results_refined, columns=['Return', 'Risk', 'Weights'])

# Analyze the Sharpe Ratio
risk_free_rate = 0  # Assume risk-free rate is 0 for simplicity
efficient_frontier_refined['Sharpe Ratio'] = efficient_frontier_refined['Return'] / efficient_frontier_refined['Risk']

# Find maximum Sharpe Ratio portfolio
max_sharpe_idx = efficient_frontier_refined['Sharpe Ratio'].idxmax()
max_sharpe_portfolio = efficient_frontier_refined.iloc[max_sharpe_idx]

# Find minimum variance portfolio
min_variance_idx = efficient_frontier_refined['Risk'].idxmin()
min_variance_portfolio = efficient_frontier_refined.iloc[min_variance_idx]

# Variances and Labels for Graph
variances_assets = np.diag(cov_matrix)
asset_labels = ["Forward", "USD Security", "EUR Security", "USD ZC", "EUR ZC"]
colors = {
    "Forward": "#1f77b4",
    "USD Security": "#ff7f0e",
    "EUR Security": "#2ca02c",
    "USD ZC": "#d62728",
    "EUR ZC": "#9467bd"
}

# Plot the Efficient Frontier
plt.figure(figsize=(12, 7))
plt.plot(efficient_frontier_refined['Risk'], efficient_frontier_refined['Return'], label="Efficient Frontier", color="#111111", linewidth=2)

# Annotate individual assets
for i, label in enumerate(asset_labels):
    plt.scatter(np.sqrt(variances_assets[i]), means[i], label=label, s=80, color=colors[label], edgecolor="black")

# Highlight the portfolio with the maximum Sharpe Ratio
plt.scatter(max_sharpe_portfolio['Risk'], max_sharpe_portfolio['Return'], color="blue", marker="x", label="Max Sharpe Ratio Portfolio", s=100, edgecolor="black")
plt.scatter(min_variance_portfolio['Risk'], min_variance_portfolio['Return'], color="red", marker="x", label="Minimum Variance Portfolio", s=100, edgecolor="black")

# Labels and grid
plt.xlabel("Portfolio Standard Deviation")
plt.ylabel("Portfolio Return")
plt.legend()
plt.grid()
plt.show()

# Weight Distribution Across Returns
weights_df = pd.DataFrame(efficient_frontier_refined['Weights'].tolist(), columns=asset_labels)
cutoff_return = 0.08
filtered_indices = efficient_frontier_refined[efficient_frontier_refined['Return'] <= cutoff_return].index
filtered_returns = efficient_frontier_refined.loc[filtered_indices, 'Return']
filtered_weights = weights_df.iloc[filtered_indices].drop(columns=["Forward"], errors="ignore")

# Stack plot of weight distribution
fig, ax = plt.subplots(figsize=(12, 6))
ax.stackplot(
    filtered_returns,
    filtered_weights.T,
    labels=filtered_weights.columns,
    colors=[colors[col] for col in filtered_weights.columns if col != "Forward"],  # Exclude Forward
    alpha=0.85
)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.set_xlabel("Portfolio Return")
ax.set_ylabel("Portfolio Weights")
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Assets")
plt.tight_layout()
plt.grid()
plt.show()
