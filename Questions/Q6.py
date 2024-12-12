import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data.Data import covariance_matrix, init_values, cov_matrix, x0, mean_vector, delta_t
from scipy.stats import norm, lognorm
import seaborn as sns
from Questions.Q4 import mean_log_FX, var_log_FX, mean_log_V_US, var_log_V_US, mean_Z_US_4Y, var_Z_US_4Y, mean_log_V_EUR, var_log_V_EUR,mean_Z_EUR_4Y, var_Z_EUR_4Y
from Questions.Q5 import mean_1_FX, var_1_FX, mean_V1_US_EUR, var_V1_US_EUR, mean_Z1_US_4Y_EUR, var_Z1_US_4Y_EUR, mean_P1_EUR

# Forward price calculation
fx0=np.exp(x0[0])
F0_1=fx0*np.exp(1*(x0[10]-x0[4]))

# Mean and variance of PnL components
# FX Forward PnL
E_PnL_FX = (1/F0_1)-mean_1_FX

# V1_US PnL in EUR
E_PnL_V1_US = mean_P1_EUR[1]-np.exp(x0[1])

# EUR Equities PnL
mean_V1_EUR = np.exp(mean_log_V_EUR)  # Expected value of EUR equities
E_PnL_V1_EUR = mean_V1_EUR - np.exp(x0[2])

#Z1_US_4Y-Z0_US_5Y
Z0_US_5Y=np.exp(x0[12]*5)
E_PnL_Z1_US = mean_Z1_US_4Y_EUR - Z0_US_5Y

#Z1_EUR_4Y-Z0_EUR_5Y
Z0_EUR_5Y=np.exp(x0[6]*5)
E_PnL_Z1_EUR = mean_P1_EUR[4] - Z0_US_5Y


# Combine into the PnL vector
PnL_mean_vector = np.array([E_PnL_FX, E_PnL_V1_US, E_PnL_V1_EUR, E_PnL_Z1_US, E_PnL_Z1_EUR])