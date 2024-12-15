import pandas as pd
import numpy as np
from Data.Data import covariance_matrix, cov_matrix, x0, mean_vector


#Given parameters:
mu = mean_vector
Sigma = cov_matrix
time_steps = 52

#Calculate mean and covariance matrix of X1:
mean_X1 = x0 + time_steps * mu
cov_X1 = time_steps * Sigma

#Define indices for P1 components in X1:
log_FX_index = 0
log_V_US_index = 1
log_V_EUR_index = 2
y_US_3_index = 11
y_US_5_index = 12
y_EUR_3_index = 5
y_EUR_5_index = 6

##Analytical transformations for P1 components##
#The EUR/USD FX rate:
    #The log FX rate:
mean_log_FX1 = mean_X1[log_FX_index]
var_log_FX1 = cov_X1[log_FX_index, log_FX_index] #The log variance of the EUR/USD FX rate at time 1 = 0.005864534599574461

    #The lognormal FX rate:
mean_FX1=np.exp(mean_log_FX1+(var_log_FX1/2)) #The mean of the EUR/USD FX rate at time 1 = 1.0599535146393084
var_FX1=(np.exp(var_log_FX1)-1)*np.exp(2*mean_log_FX1+var_log_FX1) #The variance of the EUR/USD FX rate at time 1 = 0.006608171129626716


#The value of the US equity (local):
    #log mean and variance:
mean_log_V1_us_local = mean_X1[log_V_US_index]
var_log_V1_us_local = cov_X1[log_V_US_index, log_V_US_index]

    #lognormal mean and variance:
mean_V1_us_local=np.exp(mean_log_V1_us_local+(var_log_V1_us_local/2))
var_V1_us_local=(np.exp(var_log_V1_us_local)-1)*np.exp(2*mean_log_V1_us_local+var_log_V1_us_local)


#The value of the EU equity:
    #log mean and variance:
mean_log_V1_eur = mean_X1[log_V_EUR_index]
var_log_V1_eur = cov_X1[log_V_EUR_index, log_V_EUR_index]

    #lognormal mean and variance:
mean_v1_eur=np.exp(mean_log_V1_eur+(var_log_V1_eur/2))
var_v1_eur=(np.exp(var_log_V1_eur)-1)*np.exp(2*mean_log_V1_eur+var_log_V1_eur)
#print(mean_v1_eur,var_v1_eur)


#Yield and price for 4-year bond:
#US bond yield:
    #mean:
mean_y1_us_4 = mean_X1[y_US_3_index] + (4 - 3) / (5 - 3) * (mean_X1[y_US_5_index] - mean_X1[y_US_3_index])

    #variance:
var_y1_us_3=cov_X1[y_US_3_index, y_US_3_index]
var_y1_us_5=cov_X1[y_US_5_index, y_US_5_index]
var_y1_us_4=var_y1_us_3+(4-3)/(5-3)*(var_y1_us_5-var_y1_us_3)

#US bond price:
t=4 #Time to maturity
mean_z1_us_4=np.exp(-mean_y1_us_4*t+(t**2*var_y1_us_4)/2)
var_z1_us_4=np.exp(-2*t*mean_y1_us_4+(t**2)*var_y1_us_4)*(np.exp((t**2)*(var_y1_us_4))-1)


#EU bond yield:
    #mean:
mean_y1_eu_4=mean_X1[y_EUR_3_index]+(4-3)/(5-3)*(mean_X1[y_EUR_5_index]-mean_X1[y_EUR_3_index])

    #variance:
var_y1_eu_3=cov_X1[y_EUR_3_index, y_EUR_3_index]
var_y1_eu_5=cov_X1[y_EUR_5_index, y_EUR_5_index]
var_y1_eu_4=var_y1_eu_3+(4-3)/(5-3)*(var_y1_eu_5-var_y1_eu_3)

#EU bond price:
t=4
mean_z1_eu_4=np.exp(-mean_y1_eu_4*t+(t**2*var_y1_eu_4)/2)
var_z1_eu_4=np.exp(-2*t*mean_y1_eu_4+(t**2)*var_y1_eu_4)*(np.exp((t**2)*(var_y1_eu_4))-1)


#Construct H1 vector:
mean_log=[mean_log_FX1, mean_log_V1_us_local, mean_log_V1_eur, mean_y1_us_4, mean_y1_eu_4]
print(mean_log)


#Construct the mean P1 vector:
mean_P1 = np.array([mean_FX1, mean_V1_us_local, mean_v1_eur, mean_z1_us_4, mean_z1_eu_4])
print(mean_P1)


#Interpolating and Filtering the covariance matrix:
usd_3y_5y=covariance_matrix.loc[["3Y USD", "5Y USD"]]
eur_3y_5y=covariance_matrix.loc[["3Y EUR", "5Y EUR"]]

#Interpolating the 4-year yields:
usd_4y=usd_3y_5y.loc["3Y USD"]+(usd_3y_5y.loc["5Y USD"]-usd_3y_5y.loc["3Y USD"])*(4-3)/(5-3)
eur_4y=eur_3y_5y.loc["3Y EUR"]+(eur_3y_5y.loc["5Y EUR"]-eur_3y_5y.loc["3Y EUR"])*(4-3)/(5-3)

#Add to covariance matrix:
    #rows
covariance_matrix.loc["4Y USD"]=usd_4y
covariance_matrix.loc["4Y EUR"]=eur_4y

    #columns
covariance_matrix["4Y USD"]=usd_4y
covariance_matrix["4Y EUR"]=eur_4y

#Interpolate the 4-year bond yield variables:
covariance_matrix.loc["4Y USD","4Y USD"]=usd_3y_5y.loc["3Y USD", "3Y USD"]+(usd_3y_5y.loc["5Y USD", "5Y USD"]-usd_3y_5y.loc["3Y USD", "3Y USD"])*(4-3)/(5-3)
covariance_matrix.loc["4Y EUR","4Y EUR"]=eur_3y_5y.loc["3Y EUR", "3Y EUR"]+(eur_3y_5y.loc["5Y EUR", "5Y EUR"]-eur_3y_5y.loc["3Y EUR", "3Y EUR"])*(4-3)/(5-3)

#Interpolate cross-covariances:
cov_4y_us_eur=covariance_matrix.loc["3Y USD", "3Y EUR"]+(covariance_matrix.loc["5Y USD", "5Y EUR"]-covariance_matrix.loc["3Y USD", "3Y EUR"])*(4-3)/(5-3)
covariance_matrix.loc["4Y USD", "4Y EUR"]=cov_4y_us_eur
covariance_matrix.loc["4Y EUR", "4Y USD"]=cov_4y_us_eur

#Filtering:
required_variables=["fx_spot","EQV US", "EQV EUR", "4Y USD", "4Y EUR"]
filtered_covariance_matrix=covariance_matrix.loc[required_variables, required_variables]

#Covariance matrix to t=1:
cov_mat_fil1=filtered_covariance_matrix*52

#Making the covariance matrix lognormal:
log_cov=cov_mat_fil1.to_numpy()

def covariance_conversion(mean, covariance, T=4):
    n=len(mean)
    J=np.eye(n)

    #log to lognormal for variables 0-2:
    for i in range(3):
        J[i,i]=np.exp(mean[i]+0.5*covariance[i,i])

    #Yield to bond price for variables 3-4:
    z_us=np.exp(-mean[3]*T)
    z_eur=np.exp(-mean[4]*T)
    J[3,3]=-T*z_us
    J[4,4]=-T*z_eur

    #Full transformation of the covariance matrix:
    cov_P1 = J @ covariance @ J.T

    return cov_P1

cov_P1=covariance_conversion(mean_log, log_cov, 4)

variables=["FX t1", "EQV US Local t1", "EQV EUR t1", "Z4 USD Local t1", "Z4 EUR t1"]
cov_P1=pd.DataFrame(cov_P1, index=variables, columns=variables)

#Verify that the covariance matrix is positice semi-definite:
eigenvalues=np.linalg.eigvals(cov_P1.values)
is_positive_semi_definite=np.all(eigenvalues>=0)
is_positive_semi_definite, eigenvalues

#The final distribution of the P1 vector:
mean_P1, cov_P1


"""#Export to Excel:
mean_P1_excel=pd.DataFrame(mean_P1, index=variables, columns=["Mean"])
mean_P1_excel.to_excel("mean_P1.xlsx")
cov_P1.to_excel("cov_P1.xlsx")"""



