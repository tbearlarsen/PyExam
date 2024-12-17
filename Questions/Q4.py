import pandas as pd
import numpy as np
from Data.Data import covariance_matrix, cov_matrix, x0, mean_vector


#Given parameters:
mu = mean_vector
time_steps = 52

#Calculate covariance matrix at t=1:
cov_1 = covariance_matrix * time_steps

#Define indices for P1 components in X1:
log_FX_index = 0
log_V_US_index = 1
log_V_EUR_index = 2
y_US_3_index = 11
y_US_5_index = 12
y_EUR_3_index = 5
y_EUR_5_index = 6

#Extract values for the P1 vector at t=0:
log_fx_0=x0[log_FX_index]
log_v0_us=x0[log_V_US_index]
log_v0_eu=x0[log_V_EUR_index]
y0_us_3=x0[y_US_3_index]
y0_us_5=x0[y_US_5_index]
y0_eu_3=x0[y_EUR_3_index]
y0_eu_5=x0[y_EUR_5_index]

#Interpolate 4-year yields at t=0:
y0_us_4 = y0_us_3 + (4 - 3) / (5 - 3) * (y0_us_5 - y0_us_3)
y0_eu_4 = y0_eu_3 + (4 - 3) / (5 - 3) * (y0_eu_5 - y0_eu_3)

#Calculate log bond prices at t=0:
log_z0_us_4 = -4 * y0_us_4
log_z0_eu_4 = -4 * y0_eu_4

#Define the constants vector b:
b = np.array([log_fx_0, log_v0_us, log_v0_eu, log_z0_us_4, log_z0_eu_4])

#Define the affine transformation matrix A:
A = np.array([
    [1, 0, 0,  0,  0,  0,  0],
    [0, 1, 0,  0,  0,  0,  0],
    [0, 0, 1,  0,  0,  0,  0],
    [0, 0, 0, -2, -2,  0,  0],
    [0, 0, 0,  0,  0, -2, -2],
])

#Annual drift vector:
annual_mu_with_drift = np.array([0, 0.07, 0.06, 0, 0, 0, 0])

#Calculate the mean of log(P1) using the affine transformation:
mean_log_P1 = A @ annual_mu_with_drift + b
print(mean_log_P1)

#Extract the covariance matrix:
cov_1_ext = cov_1.loc[
    ["fx_spot", "EQV US", "EQV EUR", "3Y USD", "5Y USD", "3Y EUR", "5Y EUR"],
    ["fx_spot", "EQV US", "EQV EUR", "3Y USD", "5Y USD", "3Y EUR", "5Y EUR"]
].values

#Calculate the covariance of P1 using the affine transformation:
cov_log_P1=A @ cov_1_ext @ A.T
columns = ['log_FX_1', 'log_V_USD_1', 'log_V_EUR_1', 'log_Z_4Y_USD', 'log_Z_4Y_EUR']
cov_log_p1_df = pd.DataFrame(cov_log_P1, index=columns, columns=columns)
print(cov_log_p1_df)

#Bringing mean and covariance matrix out of log-space:
var_log_p1=np.diag(cov_log_P1)
mean_real_p1=np.exp(mean_log_P1+var_log_p1/2)

exp_cov_log_p1 = np.exp(cov_log_P1)
cov_real_p1 = (exp_cov_log_p1 - 1) * (mean_real_p1[:, None] @ mean_real_p1[None, :])

#Labelling:
columns = ['FX_1', 'V_USD_1', 'V_EUR_1', 'Z_4Y_USD', 'Z_4Y_EUR']
cov_real_p1_df = pd.DataFrame(cov_real_p1, index=columns, columns=columns)
mean_real_p1_df = pd.DataFrame(mean_real_p1, index=columns)
print(f"Mean P1:\n {mean_real_p1_df}"
      f"\n\nCov Mat P1:\n {cov_real_p1_df}")
