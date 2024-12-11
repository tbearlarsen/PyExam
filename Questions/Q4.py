import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data.Data import covariance_matrix, init_values, cov_matrix, x0, mean_vector, delta_t
from scipy.stats import norm, lognorm
import seaborn as sns

# Simulate the joint distribution of P1
np.random.seed(42)
num_simulations = 10000
simulated_shocks = np.random.multivariate_normal(mean_vector, cov_matrix, num_simulations)
X_1 = x0 + simulated_shocks

#Indexing
FX_t_index=0
V_US_t_index=1
V_EUR_t_index=2

# Calculate components of P1
FX_1 = np.exp(X_1[:, FX_t_index])  # Exchange rate
V1_US_local = np.exp(X_1[:, V_US_t_index])  # US equities
V1_EUR = np.exp(X_1[:, V_EUR_t_index])  # EUR equities

#Interpolate 4-year yields
#Existing yields
y1_US_3=X_1[:,11]
y1_US_5=X_1[:,12]
y1_EUR_3=X_1[:,5]
y1_EUR_5=X_1[:,6]
y1_US_4 = y1_US_3 + (4 - 3) / (5 - 3) * (y1_US_5 - y1_US_3)
y1_EUR_4 = y1_EUR_3 + (4 - 3) / (5 - 3) * (y1_EUR_5 - y1_EUR_3)


# Zero-coupon bond prices at t = 1
Z1_US_4Y = np.exp(-y1_US_4 * 4)
Z1_EUR_4Y = np.exp(-y1_EUR_4 * 4)

# Combine into the joint vector P1
P1 = np.column_stack((FX_1, V1_US_local, V1_EUR, Z1_US_4Y, Z1_EUR_4Y))

# Visualise pairwise relationships
df = pd.DataFrame(P1, columns=["FX_1", "V1_US_local", "V1_EUR", "Z1_US_4Y", "Z1_EUR_4Y"])
sns.pairplot(df)
plt.show()








