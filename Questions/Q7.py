from Questions.Q6 import E_pnl1_port, var_pnl1_port, h, E_pnl1, cov_pnl1
import numpy as np
import matplotlib.pyplot as plt

# Given covariance matrix components
sigma_pnl_11=cov_pnl1.iloc[0,0]
sigma_pnl_12=cov_pnl1.iloc[0,1:]  # Example, adjust as needed

print(sigma_pnl_11)
print(sigma_pnl_12)


# Portfolios
portfolios = {
    "A": np.array([1, 0, 0, 0]),
    "B": np.array([0, 0, 0, 1]),
    "C": np.array([0.2, 0.2, 0.3, 0.3])
}


optimal_h1={}
for portfolio_name, h2 in portfolios.items():
    h1=-np.dot(sigma_pnl_12,h2)/sigma_pnl_11
    optimal_h1[portfolio_name]=h1


# Function to calculate optimal h1 and plot
def plot_pnl(portfolio_label, h2):
    hedge_ratios = np.linspace(-1, 1.5, 100)
    std_devs = []
    expected_pnls = []

    # Optimal h1
    h1_optimal = -np.dot(sigma_pnl_12, h2) / sigma_pnl_11
    h = np.append(h1_optimal, h2)
    optimal_pnl = np.dot(h, E_pnl1)  # Expected PnL
    optimal_std = np.sqrt(np.dot(h.T, np.dot(cov_pnl1, h)))  # Standard deviation

    for ratio in hedge_ratios:
        h1 = ratio * np.sum(h2)  # Define h1 based on the hedge ratio
        h = np.append(h1, h2)
        std_devs.append(np.sqrt(np.dot(h.T, np.dot(cov_pnl1, h))))
        expected_pnls.append(np.dot(h, E_pnl1))

    plt.plot(std_devs, expected_pnls, label=f"Portfolio {portfolio_label}")
    plt.scatter(optimal_std, optimal_pnl, color='red', label=f"Optimal {portfolio_label} ({h1_optimal:.2f})")

"""# Expected PnL vector and Covariance Matrix
E_pnl1 = np.array([-0.0152296548, 0.0231674692, 0.0747645957, -270.481033, 0.0191960561])
cov_pnl1 = np.array([...])  # Full covariance matrix"""

for label, h2 in portfolios.items():
    plot_pnl(label, h2)

plt.xlabel('Standard Deviation')
plt.ylabel('Expected PnL')
plt.title('PnL vs. Standard Deviation for Different Hedge Ratios')
plt.legend()
plt.grid(True)
plt.show()









