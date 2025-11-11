import numpy as np
from scipy.io import savemat
import time

from SINDyFunctions import simulateLorenzSystem
from paramEstFunctions import *

# ---simulate Lorenz system and add noise---
# system parameters
sigma = 10
rho = 28
beta = 8/3

# simulate
X0 = np.array([20,12,-30])  # initial conditions
t_sim = 10
gridDensity = 1000  # number of steps in one second
[t_out,X_clean,X_dot_out] = simulateLorenzSystem(X0,t_sim,gridDensity,sigma,rho,beta)

# add noise
MD = t_sim*gridDensity*3
arr_sig_NR = np.hstack((np.logspace(-6,-2,5),np.linspace(0.05,1,15)))   # array of noise ratios
N_noise = 20   # number of noises to try at each noise level
arr_X_noisy = np.zeros((X_clean.shape[0],X_clean.shape[1],len(arr_sig_NR),N_noise)) # allocte a 4-way array for storing noisy trajectory data

rng = np.random.default_rng(seed=0)
for i in range(0,len(arr_sig_NR)):
    sig_NR = arr_sig_NR[i]
    sig = sig_NR*np.linalg.norm(X_clean,"fro")/np.sqrt(MD)
    for j in range(0,N_noise):
        X_noisy = X_clean+rng.normal(0,sig,size=X_clean.shape)
        arr_X_noisy[:,:,i,j] = X_noisy

# true coefficient vector (concatonated)
w_x_true = np.array([-sigma,sigma])
w_y_true = np.array([rho,-1,-1])
w_z_true = np.array([-beta,1])
w_true = np.hstack((w_x_true,w_y_true,w_z_true))

# ---parameter estimation---
arr_w = np.zeros((len(w_true),len(arr_sig_NR),N_noise))
startTime = time.perf_counter()
for i in range(0,len(arr_sig_NR)):
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        _,_,_,w_ident = FourierParamEst_Lorenz(t_out,X_noisy)
        arr_w[:,i,j] = w_ident
        print(f"Progress: {i*N_noise+(j+1)}/{len(arr_sig_NR)*N_noise}")
endTime = time.perf_counter()
time_FourierWSINDyFFT = endTime-startTime

data = dict()
data["w_true"] = w_true
data["arr_w"] = arr_w

savemat("Lorenz_paramEst.mat",data)