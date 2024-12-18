import numpy as np
import matplotlib.pyplot as plt
from Data.Data import covariance_matrix, init_values, cov_matrix, x0, delta_t
from scipy.stats import norm, lognorm
from Questions.Q1 import X_t, mu

#Define parameters:
time_horizon=52
log_v0_us_local=x0[1]
log_v1_us_local=X_t[time_horizon,1,:]
v1_us_local=np.exp(log_v1_us_local)

#Plot distribution of V_1^US,local (this is the simulation):
plt.figure(figsize=(10, 6))
plt.hist(v1_us_local, bins=50, color='skyblue', edgecolor='black', density=True)
plt.xlabel('V_US,local^1')
plt.ylabel('Density')
plt.title('Distribution of V_US,local^1 at t=1 (USD)')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()

#Summary statistics:
mean_V1_us_local=np.mean(v1_us_local)
std_V1_us_local=np.std(v1_us_local)

print(mean_V1_us_local, std_V1_us_local)


#Define analytical parameters:
an_mean_delta_log_v=mu[1]
an_std_delta_log_v=np.sqrt(cov_matrix[1,1])

#Calculate log values:
an_mean_log_v1_us_local=log_v0_us_local+time_horizon*an_mean_delta_log_v
an_std_log_v1_us_local=np.sqrt(time_horizon)*an_std_delta_log_v
an_var_log_v1_us_local=an_std_log_v1_us_local**2

#Log-normal disrtibution:
an_mean_v1_us_local=np.exp(an_mean_log_v1_us_local+(an_std_log_v1_us_local**2)/2)
an_var_v1_us_local=(np.exp(an_std_log_v1_us_local*2) - 1) * np.exp(2 * an_mean_log_v1_us_local + an_std_log_v1_us_local*2)
an_std_v1_us_local=np.sqrt(an_var_v1_us_local)


#Plot simulated and analytical distributions:
x=np.linspace(np.min(v1_us_local), np.max(v1_us_local), 1000)
an_pdf=lognorm.pdf(x, s=an_std_log_v1_us_local, scale=np.exp(an_mean_log_v1_us_local))
plt.figure(figsize=(10, 6))

#Simulated histogram:
plt.hist(v1_us_local, bins=50, color='skyblue', edgecolor='black', density=True, label='Simulated')

#Analytical PDF:
plt.plot(x, an_pdf, "r-", lw=2, label='Analytical PDF (Log-normal)')
plt.xlabel('V_US,local^1')
plt.ylabel('Density')
plt.title('Simulated vs Analytical Distribution of V_US,local^1 at t=1 (USD)')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()





