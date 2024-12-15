import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Questions.Q6 import E_pnl1, cov_pnl1
from Questions.Q7 import usd_exposure_C, h2_C


def calculate_cvar(w: np.ndarray, returns: np.ndarray, beta: float = 0.95):
    # portfolio returns
    port_returns = returns @ w

    # percentile
    perc = np.percentile(port_returns, (1 - beta) * 100)

    # average in tail
    avg_tail = np.mean(port_returns[port_returns <= perc])

    return -avg_tail

#Define inputs:
num_simulations=10000

#Simulate portfolio returns:
sim_pnl_ret = np.random.multivariate_normal(E_pnl1, cov_pnl1, num_simulations)

#Defining the objective function for optimization (minimise 5% CVaR):
def objective(hedge_ratio):
    h1=hedge_ratio*usd_exposure_C
    h=np.concatenate((h1,h2_C))
    return calculate_cvar(h, sim_pnl_ret, beta=0.95)

#Optimizing the objective function:
result=minimize(
    fun=objective,
    x0=[0],
    bounds=[(-1, 1.5)],
    method="SLSQP"
)

# Extract the optimal hedge ratio
optimal_hedge_ratio = result.x[0]
optimal_h1 = optimal_hedge_ratio * usd_exposure_C
print(f'Optimal Hedge Ratio for Minimum 5% CVaR: {optimal_h1:.3f}')

# Combine the optimal hedge ratio h1 with h2 to create the full portfolio weights
h_optimal = np.concatenate(([optimal_h1], h2_C))

# Calculate portfolio returns using the optimal weights
portfolio_returns = sim_pnl_ret @ h_optimal

# Calculate 5% CVaR and expected PnL
sorted_returns = np.sort(portfolio_returns)
cvar_5_percent = -np.mean(sorted_returns[:int(0.05 * len(sorted_returns))])
expected_pnl = np.mean(portfolio_returns)

# Print results
print(f'5% CVaR with Optimal Hedge Ratio: {cvar_5_percent:.3f}')
print(f'Expected PnL with Optimal Hedge Ratio: {expected_pnl:.3f}')


# Plot combinations of 5% CVaR and expected PnL for hedge ratios ranging from -1 to 1.5
hedge_ratios = np.linspace(-1, 1.5, 100)
cvar_values = []
expected_pnls = []

for hedge_ratio in hedge_ratios:
    h1 = hedge_ratio * usd_exposure_C  # Convert hedge ratio to h1
    h = np.concatenate(([h1], h2_C))
    portfolio_returns = sim_pnl_ret @ h
    sorted_returns = np.sort(portfolio_returns)
    cvar = -np.mean(sorted_returns[:int(0.05 * len(sorted_returns))])
    expected_return = np.mean(portfolio_returns)
    cvar_values.append(cvar)
    expected_pnls.append(expected_return)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(cvar_values, expected_pnls, label="Different Hedge Ratios (-1.0,1.5)")
plt.scatter(cvar_5_percent, expected_pnl, color='brown', label=f"Optimal Point (CVaR={cvar_5_percent:.3f}, PnL={expected_pnl:.3f})")

plt.title("5% CVaR vs Expected PnL")
plt.xlabel("5% CVaR")
plt.ylabel("Expected PnL")
plt.legend()
plt.grid()
plt.show()

