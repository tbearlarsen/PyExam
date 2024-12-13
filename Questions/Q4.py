import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data.Data import covariance_matrix, init_values, cov_matrix, x0, mean_vector, delta_t
from scipy.stats import norm, lognorm
import seaborn as sns

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
required_variables=["fx_spot","EQV US", "EQV EUR", "4Y EUR", "4Y USD"]
filtered_covariance_matrix=covariance_matrix.loc[required_variables, required_variables]

#Covariance matrix to t=1:
cov_mat_fil1=filtered_covariance_matrix*52

print(cov_mat_fil1)





# Mean and variance for lognormal bond prices
mean_Z_US_4Y = -mean_y_US_4 * 4
var_Z_US_4Y = (4 ** 2) * var_y_US_4

mean_Z_EUR_4Y = -mean_y_EUR_4 * 4
var_Z_EUR_4Y = (4 ** 2) * var_y_EUR_4




# Jacobian matrix for transformations
J = np.zeros((5, len(x0)))

# Fill Jacobian entries for each transformation
# FX_1 = exp(log FX)
J[0, log_FX_index] = np.exp(mean_log_FX1)

# V1_US = exp(log V_US)
J[1, log_V_US_index] = np.exp(mean_log_V_US)

# V1_EUR = exp(log V_EUR)
J[2, log_V_EUR_index] = np.exp(mean_log_V_EUR)

# Z1_US_4Y = exp(-y_US_4 * 4)
J[3, y_US_3_index] = -4 * (1 - (4 - 3) / (5 - 3)) * np.exp(mean_Z_US_4Y)
J[3, y_US_5_index] = -4 * (4 - 3) / (5 - 3) * np.exp(mean_Z_US_4Y)

# Z1_EUR_4Y = exp(-y_EUR_4 * 4)
J[4, y_EUR_3_index] = -4 * (1 - (4 - 3) / (5 - 3)) * np.exp(mean_Z_EUR_4Y)
J[4, y_EUR_5_index] = -4 * (4 - 3) / (5 - 3) * np.exp(mean_Z_EUR_4Y)

# Covariance matrix of P1
cov_P1 = J @ cov_X1 @ J.T

# Output mean vector and covariance matrix of P1
mean_P1 = np.array([
    mean_FX1,
    np.exp(mean_log_V_US),
    np.exp(mean_log_V_EUR),
    np.exp(mean_Z_US_4Y),
    np.exp(mean_Z_EUR_4Y),
])











"""#Calcualting the covariances for the 4Y yields and adding to the covariance matrix:
cov_matrix_df=pd.read_excel(r"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/PyExam/Data/covariance_matrix.xlsx")

#Covariance matrix processing:
labels = cov_matrix_df.iloc[:, 0].tolist()
eur_index = labels.index('3Y EUR') + 1
usd_index = labels.index('3Y USD') + 1
labels.insert(eur_index, '4Y EUR')
labels.insert(usd_index, '4Y USD')

n = len(labels)
expanded_cov_matrix = np.zeros((n, n))
old_cov_matrix = cov_matrix_df.drop('Unnamed: 0', axis=1).values

#Fill old covariance values:
for i in range(len(labels)):
    for j in range(len(labels)):
        old_i = cov_matrix_df.columns[1:].tolist().index(labels[i]) if labels[i] in cov_matrix_df.columns[1:] else -1
        old_j = cov_matrix_df.columns[1:].tolist().index(labels[j]) if labels[j] in cov_matrix_df.columns[1:] else -1
        if old_i != -1 and old_j != -1:
            expanded_cov_matrix[i, j] = old_cov_matrix[old_i, old_j]

#Interpolate and insert new covariance values:
for label in ['EUR', 'USD']:
    i3 = labels.index('3Y ' + label)
    i5 = labels.index('5Y ' + label)
    i4 = labels.index('4Y ' + label)
    for i in range(n):
        expanded_cov_matrix[i, i4] = (expanded_cov_matrix[i, i3] + expanded_cov_matrix[i, i5]) / 2
        expanded_cov_matrix[i4, i] = expanded_cov_matrix[i, i4]

#Set variance for 4Y positions:
expanded_cov_matrix[i4, i4] = (expanded_cov_matrix[i3, i3] + 2 * expanded_cov_matrix[i3, i5] + expanded_cov_matrix[i5, i5]) / 4

#Convert to DataFrame:
expanded_cov_matrix_df = pd.DataFrame(expanded_cov_matrix, index=labels, columns=labels)

#--------------------------------------------------------#
#Calculate the 4-year yield:
y0_eur_3=x0[5]
y0_eur_5=x0[6]
y0_us_3=x0[11]
y0_us_5=x0[12]

y0_US_4=y0_us_3+(y0_us_5-y0_eur_3)/2
y0_EUR_4=y0_eur_3+(y0_eur_5-y0_eur_3)/2

#--------------------------------------------------------#
#Filtering the covariance matrix:
required_variables=["fx_spot","EQV US", "EQV EUR", "4Y EUR", "4Y USD"]
filtered_cov_matrix = expanded_cov_matrix_df.loc[required_variables, required_variables]


#Calculate the mean and covariance of X1:
mean_X1=x0+delta_t*mean_vector
cov_X1=expanded_cov_matrix_df*52

#Indexing:
log_FX_index = 0
log_V_US_index = 1
log_V_EUR_index = 2

#Analytical transformations for P1 components:
mean_log_FX=mean_X1[log_FX_index]
var_log_FX=cov_X1.iloc[log_FX_index,log_FX_index]

mean_log_V_US = mean_X1[log_V_US_index]
var_log_V_US = cov_X1.iloc[log_V_US_index, log_V_US_index]

mean_log_V_EUR = mean_X1[log_V_EUR_index]
var_log_V_EUR = cov_X1.iloc[log_V_EUR_index, log_V_EUR_index]




#Extracting means
log_FX=mean_vector[0]
log_V_US_mean=mean_vector[1]
log_V_EUR=mean_vector[2]

#Taking the values to time = 1
log_FX_t1=log_FX*52
log_V_US_mean_t1=log_V_US_mean*52
log_V_EUR_t1=log_V_EUR*52

cov_matrix_t1=filtered_cov_matrix*52


#Zero-coupon bond prices at t=1:
Z1_US_4Y = np.exp(-y1_US_4 * 4)
Z1_EUR_4Y = np.exp(-y1_EUR_4 * 4)




#Simulate the joint distribution of P1:
np.random.seed(42)
num_simulations = 10000
simulated_shocks = np.random.multivariate_normal(mean_vector, cov_matrix, num_simulations)
X_1 = x0 + simulated_shocks

#Indexing
FX_t_index=0
V_US_t_index=1
V_EUR_t_index=2

# Calculate components of P1
FX_1 = np.exp(X_1[:, FX_t_index])  # Exchange rate
V1_US_local = np.exp(X_1[:, V_US_t_index])  # US equities
V1_EUR = np.exp(X_1[:, V_EUR_t_index])  # EUR equities

#Interpolate 4-year yields
#Existing yields
y1_US_3=X_1[:,11]
y1_US_5=X_1[:,12]
y1_EUR_3=X_1[:,5]
y1_EUR_5=X_1[:,6]
y1_US_4 = y1_US_3 + (4 - 3) / (5 - 3) * (y1_US_5 - y1_US_3)
y1_EUR_4 = y1_EUR_3 + (4 - 3) / (5 - 3) * (y1_EUR_5 - y1_EUR_3)


# Zero-coupon bond prices at t = 1
Z1_US_4Y = np.exp(-y1_US_4 * 4)
Z1_EUR_4Y = np.exp(-y1_EUR_4 * 4)

# Combine into the joint vector P1
P1 = np.column_stack((FX_1, V1_US_local, V1_EUR, Z1_US_4Y, Z1_EUR_4Y))

# Visualise pairwise relationships
df = pd.DataFrame(P1, columns=["FX_1", "V1_US_local", "V1_EUR", "Z1_US_4Y", "Z1_EUR_4Y"])
sns.pairplot(df)
plt.show()"""

