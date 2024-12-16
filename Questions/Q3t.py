import numpy as np

"""Interpolate values"""
# Given data (example values — replace with actual exam data)
y0_3 = 0.038206506  # Initial 3-year yield (2.5%)
y0_5 = 0.037408587  # Initial 5-year yield (3.5%)

# Variance (diagonal) of changes in 3-year and 5-year yields from the covariance matrix
var_delta_y_t_3 = 1.09022E-06  # Variance of change in 3-year yield
var_delta_y_t_5 = 1.16941E-06  # Variance of change in 5-year yield

# Step 1: Calculate the initial 4-year yield using linear interpolation
y0_4 = y0_3 + ((4 - 3) / (5 - 3)) * (y0_5 - y0_3)

# Step 2: Calculate the variance of the change in the 4-year yield using linear interpolation of variances
var_delta_y_t_4 = var_delta_y_t_3 + ((4 - 3) / (5 - 3)) * (var_delta_y_t_5 - var_delta_y_t_3)

# Step 3: Calculate the standard deviation of the change in the 4-year yield
sigma_delta_y_t_4 = np.sqrt(var_delta_y_t_4)

print(y0_4)
print(sigma_delta_y_t_4)


# Set parameters for the simulation
np.random.seed(42)  # For reproducibility
num_simulations = 1000  # Number of paths
time_steps = 52  # 52 weekly steps in a year
sigma_delta_y_4 = sigma_delta_y_t_4  # Weekly standard deviation for the 4-year yield change
y0_4 = y0_4  # Initial 4-year yield

#Simulate the weekly changes in the 4-year yield (for each path)
weekly_changes = np.random.normal(loc=0, scale=sigma_delta_y_4, size=(time_steps, num_simulations)) #loc=0 because mean is 0.

#Calculate the total change in yield for each path over 1 year
total_change_y_4 = np.sum(weekly_changes, axis=0)

#Calculate the yield at t=1 for each path
y_t1_4 = y0_4 + total_change_y_4

#Calculate the bond price at t=1 for each path
Z_t1 = np.exp(-y_t1_4 * 4)



#Calculate the analytical log-normal distribution parameters
mu_log_Z_t1 = -4 * y0_4  #   log of Z_t1, is necessary for the lognorm.pdf function
sigma_log_Z_t1 = np.sqrt(16 * 52 * sigma_delta_y_4**2)  # Std of the log of Z_t1, is necessary for the lognorm.pdf function

#Calculate the analytical PDF for the log-normal distribution
x_vals = np.linspace(np.min(Z_t1), np.max(Z_t1), 1000)
pdf_analytical = lognorm.pdf(x_vals, s=sigma_log_Z_t1, scale=np.exp(mu_log_Z_t1))



# Plot histogram of the simulated bond prices at t=1
plt.figure(figsize=(10, 6))
plt.hist(Z_t1, bins=50, density=True, color='skyblue', edgecolor='black', label='Simulated Distribution')

# Plot the analytical PDF of the log-normal distribution
plt.plot(x_vals, pdf_analytical, 'r-', linewidth=2, label='Analytical PDF (Log-normal)')

# Formatting the plot
plt.xlabel('Bond Price (Z_t1) at t=1')
plt.ylabel('Density')
plt.title('Simulated vs Analytical Distribution of Bond Price at t=1 (with Weekly Time Steps)')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()