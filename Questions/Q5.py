import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm
from Questions.Q4 import mean_real_p1, cov_real_p1_df, mean_log_P1, cov_log_P1

C = np.array([
    [-1,  0,  0,  0,  0],  # Log(1/FX) = -Log FX
    [-1,  1,  0,  0,  0],  # Log V_US - Log FX
    [ 0,  0,  1,  0,  0],  # Log V_EUR
    [-1,  0,  0,  1,  0],  # Log Z_USD (4Y) - Log FX
    [ 0,  0,  0,  0,  1],  # Log Z_EUR (4Y)
])

#Compute transformed mean vector:
mean_log_p1_eur=C@mean_log_P1
print(f"Mean log P1 EUR:\n {mean_log_p1_eur}")

#Compute transformed covariance matrix:
cov_log_p1_eur=C@cov_log_P1@C.T
columns = ['log(1/FX_1)', 'log_V_USD_1 in EUR', 'log_V_EUR_1', 'log_Z_4Y_USD in EUR', 'log_Z_4Y_EUR']
cov_log_p1_eur_df = pd.DataFrame(cov_log_p1_eur, index=columns, columns=columns)
print(f"Cov Mat log P1 EUR:\n {cov_log_p1_eur_df}")

#Bringing mean and covariance matrix out of log-space:
var_log_p1_eur=np.diag(cov_log_p1_eur)
mean_real_p1_eur=np.exp(mean_log_p1_eur+var_log_p1_eur/2)

exp_cov_log_p1_eur = np.exp(cov_log_p1_eur)
cov_real_p1_eur = (np.exp(cov_log_p1_eur) - 1) * np.exp(np.add.outer(mean_log_p1_eur, mean_log_p1_eur) + cov_log_p1_eur)




#Labelling:
columns = ['1/FX_1', 'V_USD_1 in EUR', 'V_EUR_1', 'Z_4Y_USD in EUR', 'Z_4Y_EUR']
cov_real_p1_eur_df = pd.DataFrame(cov_real_p1_eur, index=columns, columns=columns)
mean_real_p1_eur_df = pd.DataFrame(mean_real_p1_eur, index=columns)
print(f"Mean P1 EUR:\n {mean_real_p1_eur_df}"
      f"\n\nCov Mat P1 EUR:\n {cov_real_p1_eur_df}")

#Compare analytical distribution for V_US with the simulated distribution:
v1_us_mean=mean_real_p1_eur_df.iloc[1]
v1_us_var=cov_real_p1_eur_df.loc["V_USD_1 in EUR", "V_USD_1 in EUR"]


#Number of simulations:
num_simulations = 10000
np.random.seed(42)

#Simulate V1^US in EUR:
simulated_v1_us_eur = np.random.lognormal(
    mean=np.log(v1_us_mean) - 0.5 * v1_us_var,
    sigma=np.sqrt(v1_us_var),
    size=num_simulations
)

#Analytical distribution parameters:
x_values = np.linspace(min(simulated_v1_us_eur), max(simulated_v1_us_eur), 1000)
pdf_values = lognorm.pdf(x_values, s=np.sqrt(v1_us_var), scale=v1_us_mean)

#Plotting the distributions:
plt.figure(figsize=(10, 6))
plt.hist(simulated_v1_us_eur, bins=50, alpha=0.5, density=True, label='Simulated Distribution')
plt.plot(x_values, pdf_values, 'r', label='Analytical Distribution', linewidth=2)
plt.title('Comparison of Simulated and Analytical Distributions for $V_{1}^{US}$ in EUR')
plt.xlabel('$V_{1}^{US}$ in EUR')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
