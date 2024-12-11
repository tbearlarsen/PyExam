import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data.Data import covariance_matrix, init_values, cov_matrix, x0, mean_vector, delta_t
from scipy.stats import norm, lognorm
import seaborn as sns
from Questions.Q5 import FX_1, V1_US, V1_EUR, Z1_US_4Y, Z1_EUR_4Y


#Calculate forward rate:
y0_us_1=x0[10]
y0_eur_1=x0[4]
F0_1 = np.exp(x0[0])*np.exp(y0_us_1-y0_eur_1)

#Indexing:
V0_US=np.exp(x0[1])
V0_EUR=np.exp([2])
Z0_US_5Y=np.exp(x0[12])
Z0_EUR_5Y=np.exp(x0[6])

#Calculate instruments of PnL_1:
#1:
PnL_FX = 1 / F0_1 - 1 / FX_1

#2:
PnL_V1_US=V1_US-V0_US

#3:
PnL_V1_EUR = V1_EUR - V0_EUR

#4:
PnL_Z1_US = Z1_US_4Y - Z0_US_5Y

#5:
PnL_Z1_EUR = Z1_EUR_4Y - Z0_EUR_5Y


#Combine into PnL vector:
PnL_1 = np.column_stack((PnL_FX, PnL_V1_US, PnL_V1_EUR, PnL_Z1_US, PnL_Z1_EUR))

print(PnL_1)




