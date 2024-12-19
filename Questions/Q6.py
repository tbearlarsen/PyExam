import numpy as np
import pandas as pd
from Data.Data import cov_matrix, x0, mean_vector
from Questions.Q5 import cov_real_p1_eur_df, mean_real_p1_eur_df


# Forward price calculation
fx0=np.exp(x0[0])
F0_1=fx0*np.exp(1*(x0[10]-x0[4]))

#Expected value of the PnL1 vector:
#FX Forward PnL
E_pnl_fx=(1/F0_1)-mean_real_p1_eur_df.iloc[0]

#EQV US in EUR:
E_pnl_v_us=mean_real_p1_eur_df.iloc[1]-(np.exp(x0[1])/fx0)

#EQV EUR:
E_pnl_v_eur=mean_real_p1_eur_df.iloc[2]-np.exp(x0[2])

#Price of US bond in EUR:
    #Calculating Z0:
mean_x0=x0+mean_vector
t=5
mean_y0_us_5=mean_x0[12]
var_y0_us_5=cov_matrix[12,12]

mean_z0_us_5_local=np.exp(-mean_y0_us_5*t+(t**2*var_y0_us_5)/2) #This formula is different from the one that the others use (it produces a slightly differetn result)
var_z0_us_5_local=np.exp(-2*t*mean_y0_us_5+(t**2)*var_y0_us_5)*(np.exp((t**2)*(var_y0_us_5))-1)

mean_z0_us_5=mean_z0_us_5_local/fx0

    #PnL:
E_pnl_z_us=mean_real_p1_eur_df.iloc[3]-mean_z0_us_5

#Price of EUR bond:
    #Calculating Z0:
mean_y0_eur_5=mean_x0[6]
var_y0_eur_5=cov_matrix[6,6]

mean_z0_eur_5=np.exp(-mean_y0_eur_5*t+(t**2*var_y0_eur_5)/2)
var_z0_eur_5=np.exp(-2*t*mean_y0_eur_5+(t**2)*var_y0_eur_5)*(np.exp((t**2)*(var_y0_eur_5))-1)

    #PnL:
E_pnl_z_eur=mean_real_p1_eur_df.iloc[4]-mean_z0_eur_5

#Expected value of the PnL1 vector:
E_pnl1=np.array([E_pnl_fx,E_pnl_v_us,E_pnl_v_eur,E_pnl_z_us,E_pnl_z_eur])

# Convert P1 EUR to PnL
# Transformation matrix C
C = np.array([
    [-1,  0,  0,  0,  0],  # -(1/FX)
    [ 0,  1,  0,  0,  0],  # V_US
    [ 0,  0,  1,  0,  0],  # V_EUR
    [ 0,  0,  0,  1,  0],  # Z_USD (4Y)
    [ 0,  0,  0,  0,  1],  # Z_EUR (4Y)
])

""" # Compute the transformed mean vector
mean_log_p1_eur = C @ mean_log_p1
print(mean_log_p1_eur) """

# Transform the covariance matrix
cov_pnl1 = C @ cov_real_p1_eur_df @ C.T
cov_pnl1.shape

print(E_pnl1)
print(cov_pnl1)
print(fx0)

"""h=np.array([1,1,1,1,1])
E_pnl1_p=np.dot(h,E_pnl1)
var_pnl1_p=np.dot(h.T,np.dot(cov_pnl1,h))

print(E_pnl1_p)
print(var_pnl1_p)
"""

