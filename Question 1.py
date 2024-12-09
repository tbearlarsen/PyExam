import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Data.Data import data_cov,data_init
from scipy.stats import norm

#Extract the initial values as a vector
x0=data_init["initial values"].values

#Defining the time parameters
h=1 #1 year
dt=1/52 #Weekly time steps
steps=int(h/dt) #Number of steps

#Mean vector
mu = np.zeros(len(x0))
mu[1]=0.07*dt
mu[2]=0.06*dt

#Covariance matrix
sigma=data_cov.values

#Simulate dXt (weekly increments)
np.random.seed(42)  # For reproducibility
delta_X_t = np.random.multivariate_normal(mu, sigma, size=steps)

#Simulate the evolution of Xt
X_t = np.zeros((steps + 1, len(x0)))
X_t[0] = x0

for t in range(steps):
    X_t[t + 1] = X_t[t] + delta_X_t[t]

#Extract log FX_t (index 0)
log_FX_t = X_t[:, 0]

# Plot the evolution of log FX_t
plt.figure(figsize=(10, 6))
plt.plot(np.arange(steps + 1) * dt, log_FX_t, label='log(FX_t)')
plt.title('Evolution of log(FX_t) Over Time')
plt.xlabel('Time (Years)')
plt.ylabel('log(FX_t)')
plt.legend()
plt.grid(True)
plt.show()


#Distribution of X_t[1]
X_t_1_distribution = X_t[:, 1]

#Plot distribution of X_t[1]
plt.figure(figsize=(10, 6))
sns.histplot(X_t_1_distribution, bins=30, kde=True)
plt.title('Distribution of X_t[1]')
plt.xlabel('X_t[1]')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


