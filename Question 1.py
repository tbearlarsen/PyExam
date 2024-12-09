import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Data.Data import data_cov,data_init

#Extract the initial values as a vector
x0=data_init["initial values"].values

#Defining the time parameters
h=1 #1 year
dt=1/52 #Weekly time steps
steps=int(h/dt) #Number of steps

#Mean vector
mu = np.zeros(len(x0))
mu[1]=0.07*dt
mu[2]=0.06*dt


#Covariance matrix
sigma=data_cov.values
















