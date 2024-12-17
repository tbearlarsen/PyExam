import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cvxpy as cp

# Input Data
means = np.array([-0.01522965, 0.0769594, 0.0747646, 0.02660308, 0.01919606])
cov_matrix = np.array([
    [0.005220, -0.003732, -0.000248, -0.004000,  0.000071],
    [-0.003732,  0.025256,  0.019484,  0.002351,  0.000531],
    [-0.000248,  0.019484,  0.027958, -0.000440,  0.000355],
    [-0.004000,  0.002351, -0.000440,  0.003637,  0.000324],
    [0.000071,  0.000531,  0.000355,  0.000324,  0.000494]
])
hedge_ratios = [0.6765882566104374, 0.7250622046062872]
num_assets = len(means)

# Pre-specified hedge ratios
fx_hedge_bond = hedge_ratios[1]  # Hedge ratio for asset 3
fx_hedge_equity = hedge_ratios[0]  # Hedge ratio for asset 1

# Generate return targets
mu_targets = np.linspace(min(means), max(means), 1000)

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
valid_targets = []
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
            valid_targets.append(pnl_target)  # Store only valid targets
    except Exception as e:
        print(f"Optimization failed for pnl_target={pnl_target}: {e}")

    # Remove target return constraint for next iteration
    constraints.pop()

# Convert results to arrays for further analysis
portfolio_weights = np.array(portfolio_weights)
portfolio_variances = np.array(portfolio_variances)
portfolio_std = np.sqrt(portfolio_variances)
portfolio_returns = portfolio_weights @ means
valid_targets = np.array(valid_targets)

# Identify key portfolios
min_var_index = np.argmin(portfolio_variances)
max_slope_index = np.argmax(portfolio_returns / portfolio_std)

asset_labels = ["Forward", "USD Security", "EUR Security", "USD ZC", "EUR ZC"]
colors = {
    "Forward": "#1f77b4",
    "USD Security": "#ff7f0e",
    "EUR Security": "#2ca02c",
    "USD ZC": "#d62728",
    "EUR ZC": "#9467bd"
}
variances_assets = np.diag(cov_matrix)  # Variances of individual assets

# Efficient Frontier Plot
plt.figure(figsize=(12, 7))
plt.plot(portfolio_std, portfolio_returns, label="Efficient Frontier", color="#111111", linewidth=2)  # Black line
plt.xlabel("Portfolio Standard Deviation")
plt.ylabel("Portfolio Return")

# Annotate individual assets
for i, label in enumerate(asset_labels):
    plt.scatter(np.sqrt(variances_assets[i]), means[i], label=label, s=80, color=colors[label], edgecolor="black")  # Larger dots, black edge

# Highlight minimum variance and maximum slope portfolios
plt.scatter(portfolio_std[min_var_index], portfolio_returns[min_var_index],
            color='red', marker='x', s=100, label='Min Variance Portfolio', edgecolors='black', zorder=6)
plt.scatter(portfolio_std[max_slope_index], portfolio_returns[max_slope_index],
            color='blue', marker='x', s=100, label='Max Slope Portfolio', edgecolors='black', zorder=6)

# Add legend and grid
plt.legend()
plt.grid()
plt.show()

# Create a DataFrame for weights
weights_df = pd.DataFrame(portfolio_weights, columns=asset_labels)

# Exclude "Forward" for the weight distribution visualization
weights_df_no_forward = weights_df.drop(columns=["Forward"], errors='ignore')

# Weight Distribution Across PnL Targets
fig, ax = plt.subplots(figsize=(12, 6))
ax.stackplot(
    portfolio_returns,  # Use portfolio_returns (Efficient Frontier y-axis) for the x-axis
    weights_df_no_forward.T,
    labels=weights_df_no_forward.columns,
    colors=[colors[col] for col in weights_df_no_forward.columns],
    alpha=0.85
)
# Remove the PercentFormatter for x-axis
# ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=portfolio_returns.max() * 100, decimals=1))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.set_xlabel("Portfolio Return")
ax.set_ylabel("Portfolio Weights")
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Assets")
plt.tight_layout()
plt.grid()
plt.show()
