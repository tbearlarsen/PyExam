import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data.Data import covariance_matrix, init_values, cov_matrix, x0, mean_vector, delta_t
from scipy.stats import norm, lognorm
import seaborn as sns

# Simulate the joint distribution of P1
np.random.seed(42)
simulated_shocks = np.random.multivariate_normal(mean_vector, cov_matrix, size=10000)
X_1 = x0 + simulated_shocks

# Calculate components of P1
FX_1 = np.exp(X_1[:, 0])  # Exchange rate
V1_US_local = np.exp(X_1[:, 1])  # US equities
V1_EUR = np.exp(X_1[:, 2])  # EUR equities

# Zero-coupon bond prices at t = 1
y1_US_4Y = X_1[:, 9]  # Assuming the 4-year USD yield is at index 9
y1_EUR_4Y = X_1[:, 7]  # Assuming the 4-year EUR yield is at index 7
Z1_US_4Y = np.exp(-y1_US_4Y * 4)
Z1_EUR_4Y = np.exp(-y1_EUR_4Y * 4)

# Combine into the joint vector P1
P1 = np.column_stack((FX_1, V1_US_local, V1_EUR, Z1_US_4Y, Z1_EUR_4Y))

# Visualise pairwise relationships
df = pd.DataFrame(P1, columns=["FX_1", "V1_US_local", "V1_EUR", "Z1_US_4Y", "Z1_EUR_4Y"])
sns.pairplot(df)
plt.show()








