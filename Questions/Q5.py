import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data.Data import covariance_matrix, init_values, cov_matrix, x0, mean_vector, delta_t
from scipy.stats import norm, lognorm
from Questions.Q4 import mean_P1, cov_P1

#Converting the covariance matrix to EUR:
def convert_covariance_to_EUR(mean, covariance):
    FX_1=mean[0]
    v1_us_local=mean[1]
    z4_us=mean[3]

    n=len(mean)
    J=np.eye(n)

    J[0,0]=-1/(FX_1**2)

    J[1,0]=-v1_us_local/(FX_1**2)
    J[1,1]=1/FX_1

    J[3,0]=-z4_us/(FX_1**2)
    J[3,3]=1/FX_1

    cov_P1_eur= J @ covariance @ J.T

    return cov_P1_eur

cov_P1_eur=convert_covariance_to_EUR(mean_P1, cov_P1)

#Verifying the covariance matrix is positive semi-definite:
eigenvalues=np.linalg.eigvals(cov_P1_eur.values)
is_positive_semi_definite=np.all(eigenvalues>=0)
is_positive_semi_definite, eigenvalues


#Converting the mean vector to EUR:
    # Extract individual means from the P1 vector
fx_mean = mean_P1[0]  # FX mean
v_us_mean = mean_P1[1]  # USD stock mean
v_eur_mean = mean_P1[2]  # EUR stock mean (unchanged)
z_usd_4y_mean = mean_P1[3]  # USD bond mean
z_eur_4y_mean = mean_P1[4]  # EUR bond mean (unchanged)

    # Convert each component
mu_1 = 1 / fx_mean  # 1 / FX_1
mu_2 = v_us_mean / fx_mean  # V_US_LOCAL / FX_1
mu_3 = v_eur_mean  # V_EUR (unchanged)
mu_4 = z_usd_4y_mean / fx_mean  # Z_4Y_USD / FX_1
mu_5 = z_eur_4y_mean  # Z_4Y_EUR (unchanged)

mean_P1_eur=np.array([mu_1, mu_2, mu_3, mu_4, mu_5])


#The distribution of the P1 EUR:
mean_P1_eur, cov_P1_eur