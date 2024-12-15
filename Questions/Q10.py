import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Questions.Q6 import E_pnl1, cov_pnl1, fx0
import cvxpy as cp


##STRATEGY 1: PRE-SPECIFIED HEDGE RATIOS
# Define true parameters (from Q6/Q9)
mu = E_pnl1  # Expected returns
cov_matrix = cov_pnl1.to_numpy()  # Covariance matrix
fx_rate = fx0  # Forward exchange rate

# Step 1: Calculate minimum-variance hedge ratios
sigma_usd_usd = cov_matrix[0, 0]
sigma_i_usd = cov_matrix[0, 1:]
hedge_ratios = -sigma_i_usd / sigma_usd_usd

# Step 2: Define the optimization function using CVXPY
def optimize_portfolio_cvxpy(target_pnl, mu, cov_matrix):
    num_assets = len(mu)

    # Define variables
    w = cp.Variable(num_assets)

    # Define the objective function (portfolio variance)
    portfolio_variance = cp.quad_form(w, cov_matrix)

    # Define constraints
    constraints = [
        cp.sum(w) == 1,  # Budget constraint
        w @ mu == target_pnl,  # Target expected PnL
        w >= 0  # Non-shorting constraint
    ]

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    problem.solve()

    return w.value, problem.value  # Return optimal weights and variance

# Step 3: Generate the efficient frontier
pnl_targets = np.linspace(mu.min(), mu.max(), 50)  # Define PnL targets
variances = []
expected_pnls = []
portfolios = []  # Store portfolio weights for each PnL target

for target in pnl_targets:
    weights, variance = optimize_portfolio_cvxpy(target, mu, cov_matrix)
    variances.append(variance)
    expected_pnls.append(target)
    portfolios.append(weights)

variances = np.array(variances)
expected_pnls = np.array(expected_pnls)

# Step 4: Plot the efficient frontier with portfolios
plt.figure(figsize=(10, 6))

# Plot efficient frontier
plt.plot(variances, expected_pnls, label="Efficient Frontier (Strategy 1)", color="blue")

# Plot individual portfolios
for i, (variance, pnl, weights) in enumerate(zip(variances, expected_pnls, portfolios)):
    plt.scatter(variance, pnl, label=f"Portfolio {i+1}" if i < 5 else "", alpha=0.7, color="orange")
    if i < 5:  # Annotate the first few portfolios for illustration
        plt.annotate(f"P{i+1}", (variance, pnl), fontsize=9)

plt.xlabel("Portfolio Variance (Risk)")
plt.ylabel("Expected PnL (Return)")
plt.title("Efficient Frontier with Portfolios (Strategy 1)")
plt.legend(["Efficient Frontier", "Portfolios"])
plt.grid(True)
plt.show()







"""# True parameters (use E_pnl1 and cov_pnl1 from Q6/Q9)
mu = E_pnl1  # Expected returns
cov_matrix = cov_pnl1  # Covariance matrix
fx_rate = fx0  # Forward exchange rate (from Q7)


# Helper function to calculate portfolio variance
def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))


# Helper function to optimize a portfolio for a given PnL target
def optimize_portfolio(target_pnl, hedge_ratio=None):
    num_assets = len(mu)
    initial_weights = np.ones(num_assets) / num_assets  # Start with equal weights

    # Objective: Minimize portfolio variance
    def objective(weights):
        return portfolio_variance(weights, cov_matrix)

    # Constraint: Expected PnL must match the target
    constraints = [{'type': 'eq', 'fun': lambda w: np.dot(w, mu) - target_pnl}]

    # Constraint: Non-shorting (weights >= 0)
    bounds = [(0, None) for _ in range(num_assets)]

    # Apply hedge ratio if provided
    if hedge_ratio is not None:
        def hedge_constraint(weights):
            usd_exposure = np.sum(weights[:2]) * fx_rate  # Adjust USD exposure
            return usd_exposure * hedge_ratio  # Hedge the USD exposure

        constraints.append({'type': 'eq', 'fun': hedge_constraint})

    # Constraint: Budget constraint (weights sum to 1)
    constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Perform optimization
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
    return result.x, result.fun  # Return optimal weights and variance


# Generate PnL targets for the efficient frontier
pnl_targets = np.linspace(mu.min(), mu.max(), 50)

# Strategy 1: Pre-specified hedge ratios (minimum-variance hedge ratios for each asset)
hedge_ratios = -np.diag(cov_matrix) / fx_rate
strategy1_results = []
for target in pnl_targets:
    weights, variance = optimize_portfolio(target, hedge_ratio=hedge_ratios)
    strategy1_results.append((variance, target))

# Strategy 2: No hedging initially, followed by minimum-variance hedge ratio
strategy2_results = []
for target in pnl_targets:
    # Step 1: Optimize portfolio without hedging
    weights, variance = optimize_portfolio(target, hedge_ratio=None)
    # Step 2: Apply minimum-variance hedge ratio separately
    hedge_ratio = -np.diag(cov_matrix) / fx_rate
    strategy2_results.append((variance, target))

# Strategy 3: Full-scale optimization (allow weights and hedge ratio to vary)
strategy3_results = []
for target in pnl_targets:
    weights, variance = optimize_portfolio(target, hedge_ratio=None)  # Hedge ratio is part of optimization
    strategy3_results.append((variance, target))

# Convert results to arrays for plotting
strategy1_results = np.array(strategy1_results)
strategy2_results = np.array(strategy2_results)
strategy3_results = np.array(strategy3_results)

# Plot the Efficient Frontiers
plt.figure(figsize=(10, 6))
plt.plot(strategy1_results[:, 0], strategy1_results[:, 1], label='Strategy 1: Pre-specified Hedge Ratios', marker='o')
plt.plot(strategy2_results[:, 0], strategy2_results[:, 1], label='Strategy 2: No Hedge, Min-Var Hedge Ratio',
         marker='x')
plt.plot(strategy3_results[:, 0], strategy3_results[:, 1], label='Strategy 3: Full-Scale Optimization', marker='s')

plt.xlabel('Portfolio Variance (Risk)')
plt.ylabel('Expected PnL (Return)')
plt.title('Efficient Frontiers for Different Portfolio Optimization Strategies')
plt.legend()
plt.grid(True)
plt.show()"""









