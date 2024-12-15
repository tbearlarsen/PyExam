import numpy as np
import pandas as pd
from Data.Data import cov_matrix, x0, mean_vector
from Questions.Q5 import mean_P1_eur, cov_P1_eur

# Forward price calculation
fx0=np.exp(x0[0])
F0_1=fx0*np.exp(1*(x0[10]-x0[4]))

#Expected value of the PnL1 vector:
#FX Forward PnL
E_pnl_fx=(1/F0_1)-mean_P1_eur[0]

#EQV US in EUR:
E_pnl_v_us=mean_P1_eur[1]-(np.exp(x0[1])/fx0)

#EQV EUR:
E_pnl_v_eur=mean_P1_eur[2]-np.exp(x0[2])

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
E_pnl_z_us=mean_P1_eur[3]-mean_z0_us_5

#Price of EUR bond:
    #Calculating Z0:
mean_y0_eur_5=mean_x0[6]
var_y0_eur_5=cov_matrix[6,6]

mean_z0_eur_5=np.exp(-mean_y0_eur_5*t+(t**2*var_y0_eur_5)/2)
var_z0_eur_5=np.exp(-2*t*mean_y0_eur_5+(t**2)*var_y0_eur_5)*(np.exp((t**2)*(var_y0_eur_5))-1)

    #PnL:
E_pnl_z_eur=mean_P1_eur[4]-mean_z0_eur_5

#Expected value of the PnL1 vector:
E_pnl1=np.array([E_pnl_fx,E_pnl_v_us,E_pnl_v_eur,E_pnl_z_us,E_pnl_z_eur])

#Covariance matrix of the PnL1 vector:
#cov_pnl1=cov_P1_eur

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
cov_pnl1 = C @ cov_P1_eur @ C.T

columns = ['-1/FX_1', 'V_USD_1 in EUR', 'V_EUR_1', 'Z_4Y_USD in EUR', 'Z_4Y_EUR']
cov_pnl_df = pd.DataFrame(cov_pnl1, index=columns, columns=columns)
print(cov_pnl_df)




