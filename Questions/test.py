import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt

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