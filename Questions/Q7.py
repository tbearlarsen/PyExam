from Questions.Q6 import E_pnl1, cov_pnl1, fx0
import numpy as np
import matplotlib.pyplot as plt

# Given covariance matrix components
sigma_pnl_11=cov_pnl1.iloc[0,0]
sigma_pnl_12=cov_pnl1.iloc[0,1:]

print(sigma_pnl_11)
print(sigma_pnl_12)


#CALCULATING THE HEDGE RATIO FOR EACH PORTFOLIO:
#Portfolio A:
h2_A=np.array([1,0,0,0])
optimal_h1_A=-np.dot(sigma_pnl_12,h2_A)/sigma_pnl_11
usd_exposure_A=(h2_A[0]+h2_A[2])*fx0
hr_A=optimal_h1_A/usd_exposure_A
"""weights_A=np.concatenate(([optimal_h1_A],h2_A))
port_std_A=np.sqrt(weights_A @ cov_pnl1 @ weights_A)
E_pnl_A=weights_A @ E_pnl1
"""

#Portfolio B:
h2_B=np.array([0,0,1,0])
optimal_h1_B=-np.dot(sigma_pnl_12,h2_B)/sigma_pnl_11
usd_exposure_B=(h2_A[0]+h2_A[2])*fx0
hr_B=optimal_h1_B/usd_exposure_B

#Portfolio C:
h2_C=np.array([0.2, 0.2, 0.3, 0.3])
optimal_h1_C=-np.dot(sigma_pnl_12,h2_C)/sigma_pnl_11
usd_exposure_C=(h2_A[0]+h2_A[2])*fx0
hr_C=optimal_h1_C/usd_exposure_C

print(hr_A,hr_B,hr_C)

portfolios = {
    'A': {'h2': np.array([1, 0, 0, 0]), 'hr': hr_A},
    'B': {'h2': np.array([0, 0, 1, 0]), 'hr': hr_B},
    'C': {'h2': np.array([0.2, 0.2, 0.3, 0.3]), 'hr': hr_C}
}


#PLOTTING the combinations of standard deviation and expected PnL for hedge ratios ranging from -1 to 1.5
hedge_ratios = np.linspace(-1, 1.5, 100)

# Define a function to compute PnL and variance for a given hedge ratio
def compute_metrics(h1, h2):
    h = np.concatenate(([h1], h2))
    expected_pnl = np.dot(h, E_pnl1)
    variance_pnl = np.dot(h.T, np.dot(cov_pnl1, h))
    std_dev_pnl = np.sqrt(variance_pnl)

    return expected_pnl, std_dev_pnl

# Initialize plots
fig, ax = plt.subplots()

# Plotting using dictionary
for portfolio, data in portfolios.items():
    pnl_data = np.array([compute_metrics(data['hr'] * x, data['h2']) for x in hedge_ratios])
    ax.plot(pnl_data[:, 1], pnl_data[:, 0], label=f'Portfolio {portfolio}')

    # Optimal point
    optimal_pnl, optimal_std = compute_metrics(data['hr'], data['h2'])
    ax.scatter(optimal_std, optimal_pnl, color='red')
    ax.text(optimal_std, optimal_pnl, f' Optimal {portfolio}', fontsize=9)

ax.set_xlabel('Standard Deviation of PnL')
ax.set_ylabel('Expected PnL')
ax.set_title('PnL vs Standard Deviation for Various Hedge Ratios')
ax.legend()
plt.grid(True)
plt.show()



