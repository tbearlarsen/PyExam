import pandas as pd
covariance_matrix=pd.read_excel(r"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/PyExam/Data/covariance_matrix.xlsx", index_col=0)
init_values=pd.read_excel(r"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/PyExam/Data/init_values.xlsx", index_col=0)

#Transform data
import numpy as np
cov_matrix = covariance_matrix.to_numpy()
x0 = init_values["initial values"].to_numpy()

#General parameters
delta_t = 1 / 52
mean_vector = np.array([0.07 * delta_t, 0.06 * delta_t] + [0] * (len(covariance_matrix) - 2))



