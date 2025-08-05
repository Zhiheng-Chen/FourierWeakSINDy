# ---packages---
from SINDyFunctions import *
import numpy as np
from scipy.io import savemat

# ---simulate Lorenz system and add noise---
# system parameters
sigma = 10
rho = 28
beta = 8/3

# true coefficient matrix (with 1, linear, and quadratic basis functions)
w_true = np.zeros((10,3))
w_true[1,0] = -sigma
w_true[2,0] = sigma
w_true[1,1] = rho
w_true[2,1] = -1
w_true[9,1] = -1
w_true[3,2] = -beta
w_true[7,2] = 1

# simulate
X0 = np.array([20,12,-30])  # initial conditions
t_sim = 10
gridDensity = 1000  # number of steps in one second
[t_out,X_clean,X_dot_out] = simulateLorenzSystem(X0,t_sim,gridDensity,sigma,rho,beta)

# ---Fourier weak SINDy (manual threshold)---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
N_freq = 200
w_ident = WSINDy_Fourier_Lorenz(t_out,X_clean,N_freq,params_regression)
print(w_ident)