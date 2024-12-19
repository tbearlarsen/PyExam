import numpy as np
import matplotlib.pyplot as plt
from Data.Data import cov_matrix, x0, mean_vector
from scipy.stats import lognorm

mu=mean_vector
time_horizon=52

#Define yield parameters for interpolation:
y0_us_3=x0[11]
y0_us_5=x0[12]
var_delta_y0_us_3=cov_matrix[11,11]
var_delta_y0_us_5=cov_matrix[12,12]

#4-year yield interpolation at t=0:
y0_us_4=y0_us_3+(4-3)/(5-3)*(y0_us_5-y0_us_3)
var_delta_y0_us_4=var_delta_y0_us_3+((4-3)/(5-3)*(var_delta_y0_us_5-var_delta_y0_us_3))
std_delta_y0_us_4=np.sqrt(var_delta_y0_us_4)

#4-year yield interpolation at t=1:
y1_us_3=x0[11]+mu[11]*time_horizon
y1_us_5=x0[12]+mu[12]*time_horizon
var_delta_y1_us_3=cov_matrix[11,11]*52
var_delta_y1_us_5=cov_matrix[12,12]*52

y1_us_4=y1_us_3+(4-3)/(5-3)*(y1_us_5-y1_us_3)
var_delta_y1_us_4=var_delta_y1_us_3+((4-3)/(5-3)*(var_delta_y1_us_5-var_delta_y1_us_3))
std_delta_y1_us_4=np.sqrt(var_delta_y1_us_4)

#Define simulation parameters:
np.random.seed(42)
num_simulations=100000
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
mean_log_z1_us_4=-tau_1*y1_us_4
var_log_z1_us_4=tau_1**2*52*var_delta_y0_us_4
std_log_z1_us_4=np.sqrt(var_log_z1_us_4)
#std_log_z1_us_4=tau_1*np.sqrt(52*var_delta_y0_us_4)
#var_log_z1_us_4=std_log_z1_us_4**2

#Real-scale values:
an_mean_log_z1_us_4=np.exp(mean_log_z1_us_4+(var_log_z1_us_4)/2)
an_var_z1_us_4=(np.exp(var_log_z1_us_4) - 1) * np.exp(2 * mean_log_z1_us_4 + var_log_z1_us_4)
an_std_z1_us_4=np.sqrt(an_var_z1_us_4)


#Analytical PDF for log-normal distribution:
x=np.linspace(np.min(z1_us_4_sim), np.max(z1_us_4_sim), 1000)
an_pdf=lognorm.pdf(x, s=std_log_z1_us_4, scale=np.exp(mean_log_z1_us_4))


#PLOTTING THE DISTRIBUTIONS:
plt.figure(figsize=(10, 6))

#Plot the histogram of simulated values:
plt.hist(z1_us_4_sim, bins=100, density=True, color='skyblue', edgecolor='black', label='Simulated Distribution')

#Plot the analytical PDF:
plt.plot(x, an_pdf, 'r-', lw=2, label='Analytical PDF (Log-normal)')

#Combining the plots:
plt.xlabel('Bond Price (Z_t1) at t=1')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()






"""
# Parameters for simulation
tau_0 = 5  # Maturity of the bond (5 years)

std_delta_y0_us_5 = np.sqrt(cov_matrix[12, 12])  # Standard deviation of yield changes

# Simulate the evolution of the 5-year yield
np.random.seed(42)  # Ensure reproducibility
simulated_yields = np.zeros((time_horizon + 1, num_simulations))
simulated_yields[0] = y0_us_5  # Initial yield

for t in range(1, time_horizon + 1):
    # Weekly changes in yield
    weekly_changes = np.random.normal(0, std_delta_y0_us_5, num_simulations)
    simulated_yields[t] = simulated_yields[t - 1] + weekly_changes

# Calculate the 5-year bond price evolution
simulated_prices = np.exp(-simulated_yields * tau_0)

# Plot one path of the 5-year bond yield evolution
plt.figure(figsize=(10, 6))
plt.plot(range(time_horizon + 1), simulated_yields[:, 0], label="Simulation 1", linewidth=2, color="blue")
plt.title("5-Year Yield Evolution (Single Path)")
plt.xlabel("Weeks")
plt.ylabel("Yield")
plt.grid(True)
plt.show()

# Plot multiple paths for the yield evolution
plt.figure(figsize=(10, 6))
for i in range(min(10, num_simulations)):  # Show first 10 simulations
    plt.plot(range(time_horizon + 1), simulated_yields[:, i], alpha=0.7)
plt.title("5-Year Yield Evolution (Multiple Paths)")
plt.xlabel("Weeks")
plt.ylabel("Yield")
plt.grid(True)
plt.show()

# Plot the bond price distribution at t=1
plt.figure(figsize=(10, 6))
plt.hist(simulated_prices[-1], bins=50, density=True, color="lightgreen", edgecolor="black", label="Simulated Prices")
plt.title("5-Year Bond Price Distribution at t=1")
plt.xlabel("Price")
plt.ylabel("Density")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.show()"""