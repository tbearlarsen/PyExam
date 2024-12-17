import numpy as np
import matplotlib.pyplot as plt
from Data.Data import cov_matrix, x0
from scipy.stats import lognorm

#Define yield parameters for interpolation:
y0_us_3=x0[11]
y0_us_5=x0[12]
var_delta_y0_us_3=cov_matrix[11,11]
var_delta_y0_us_5=cov_matrix[12,12]

#4-year yield interpolation:
y0_us_4=y0_us_3+(4-3)/(5-3)*(y0_us_5-y0_us_3)
var_delta_y0_us_4=var_delta_y0_us_3+((4-3)/(5-3)*(var_delta_y0_us_5-var_delta_y0_us_3))
std_delta_y0_us_4=np.sqrt(var_delta_y0_us_4)

#Define simulation parameters:
np.random.seed(42)
num_simulations=10000
time_horizon=52
tau_1=4

#Simulation of weekly changes in 4-year yield:
delta_yt=np.random.normal(loc=0, scale=std_delta_y0_us_4, size=(time_horizon, num_simulations))
total_change=np.sum(delta_yt, axis=0)

#4-year yield at t=1:
y1_us_4_sim=y0_us_4+total_change

#4-year bond price at t=1 (Simulated):
z1_us_4_sim=np.exp(-y1_us_4_sim*tau_1)


#Analytical distribution parameters:
mean_log_z1_us_4=-tau_1*y0_us_4
std_log_z1_us_4=np.sqrt(tau_1**2*52*std_delta_y0_us_4**2)

#Analytical PDF for log-normal distribution:
x=np.linspace(np.min(z1_us_4_sim), np.max(z1_us_4_sim), 1000)
an_pdf=lognorm.pdf(x, s=std_log_z1_us_4, scale=np.exp(mean_log_z1_us_4))

#PLOTTING THE DISTRIBUTIONS:
plt.figure(figsize=(10, 6))

#Plot the histogram of simulated values:
plt.hist(z1_us_4_sim, bins=50, density=True, color='skyblue', edgecolor='black', label='Simulated Distribution')

#Plot the analytical PDF:
plt.plot(x, an_pdf, 'r-', lw=2, label='Analytical PDF (Log-normal)')

#Combining the plots:
plt.xlabel('Bond Price (Z_t1) at t=1')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()


