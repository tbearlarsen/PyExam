import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cvxpy as cp

# Input Data
means_no_forward = np.array([0.0769594, 0.0747646, 0.02660308, 0.01919606])
cov_matrix_no_forward = np.array([
    [0.025256, 0.019484, 0.002351, 0.000531],
    [0.019484, 0.027958, -0.000440, 0.000355],
    [0.002351, -0.000440, 0.003637, 0.000324],
    [0.000531, 0.000355, 0.000324, 0.000494],
])
num_assets_no_forward = len(means_no_forward)

# Define CVXPY variables and constraints
weights_no_forward = cp.Variable(num_assets_no_forward)
constraints = [
    cp.sum(weights_no_forward) == 1,  # Sum of weights equals 1
    weights_no_forward >= 0  # No shorting
]

# Generate return targets
mu_targets_no_forward = np.linspace(min(means_no_forward), max(means_no_forward), 1000)

# Storage for results
optimal_weights_no_forward = []
valid_targets_no_forward = []

# Solve for each target return
for pnl_target in mu_targets_no_forward:
    # Add target return constraint
    constraints.append(weights_no_forward @ means_no_forward >= pnl_target)

    # Define the optimization problem (minimize portfolio variance)
    objective = cp.Minimize(cp.quad_form(weights_no_forward, cov_matrix_no_forward))
    problem = cp.Problem(objective, constraints)

    try:
        # Solve the optimization problem
        problem.solve()
        if weights_no_forward.value is not None:
            optimal_weights_no_forward.append(weights_no_forward.value)
            valid_targets_no_forward.append(pnl_target)
    except Exception as e:
        print(f"Optimization failed for PnL target {pnl_target}: {e}")

    # Remove target return constraint for next iteration
    constraints.pop()

# Convert results to arrays for further analysis
optimal_weights_no_forward = np.array(optimal_weights_no_forward)
valid_targets_no_forward = np.array(valid_targets_no_forward)

# Integrate forward contract into portfolio
means = np.array([-0.01522965, 0.0769594, 0.0747646, 0.02660308, 0.01919606])
cov_matrix = np.array([
    [0.005220, -0.003732, -0.000248, -0.004000,  0.000071],
    [-0.003732,  0.025256,  0.019484,  0.002351,  0.000531],
    [-0.000248,  0.019484,  0.027958, -0.000440,  0.000355],
    [-0.004000,  0.002351, -0.000440,  0.003637,  0.000324],
    [0.000071,  0.000531,  0.000355,  0.000324,  0.000494]
])
block_11 = cov_matrix[:1, :1]
block_12 = cov_matrix[:1, 1:]
block_22 = cov_matrix[1:, 1:]

# Compute forward contract weights
h1_values = [-np.dot(block_12, h2_vector) / block_11 for h2_vector in optimal_weights_no_forward]
h1_values = np.array([h1.item() for h1 in h1_values])
combined_weights = np.hstack((h1_values.reshape(-1, 1), optimal_weights_no_forward))

# Analyze portfolio returns and variances
portfolio_returns = combined_weights @ means
portfolio_variances = np.array([w @ cov_matrix @ w for w in combined_weights])
portfolio_std = np.sqrt(portfolio_variances)


# Calculate Sharpe Ratios
sharpe_ratios = (portfolio_returns) / portfolio_std

# Find the maximum Sharpe Ratio and corresponding portfolio
max_sharpe_index = np.argmax(sharpe_ratios)
max_sharpe_return = portfolio_returns[max_sharpe_index]
max_sharpe_std = portfolio_std[max_sharpe_index]

min_variance_index = np.argmin(portfolio_variances)
min_variance_return = portfolio_returns[min_variance_index]
min_variance_std = portfolio_std[min_variance_index]


# Calculate individual asset variances
variances_assets = np.diag(cov_matrix)

# Define asset labels and colors
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
plt.plot(portfolio_std, portfolio_returns, label="Efficient Frontier", color="#111111", linewidth=2)

# Annotate individual assets
for i, label in enumerate(asset_labels):
    plt.scatter(np.sqrt(variances_assets[i]), means[i], label=label, s=80, color=colors[label], edgecolor="black")

# Highlight the portfolio with the maximum Sharpe Ratio
plt.scatter(max_sharpe_std, max_sharpe_return, color="blue", marker="x",label="Max Sharpe Ratio Portfolio", s=100, edgecolor="black")
plt.scatter(min_variance_std, min_variance_return, color="red", marker="x", label="Minimum Variance Portfolio", s=100, edgecolor="black")

# Labels and grid
plt.xlabel("Portfolio Standard Deviation")
plt.ylabel("Portfolio Return")
plt.legend()
plt.grid()
plt.show()


# Create a DataFrame for weights
weights_df = pd.DataFrame(combined_weights, columns=asset_labels)

# Filter weights for visualization
cutoff_return = 0.065
filtered_indices = [i for i, r in enumerate(portfolio_returns) if r <= cutoff_return]
filtered_returns = portfolio_returns[filtered_indices]
filtered_weights = weights_df.iloc[filtered_indices].drop(columns=["Forward"], errors="ignore")

# Weight Distribution Across Returns
fig, ax = plt.subplots(figsize=(12, 6))
ax.stackplot(
    filtered_returns,
    filtered_weights.T,
    labels=filtered_weights.columns,
    colors=[colors[col] for col in filtered_weights.columns],
    alpha=0.85
)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.set_xlabel("Portfolio Return")
ax.set_ylabel("Portfolio Weights")
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Assets")
plt.tight_layout()
plt.grid()
plt.show()
