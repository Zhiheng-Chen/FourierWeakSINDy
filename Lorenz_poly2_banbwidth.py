# ---packages---
from SINDyFunctions import *
import numpy as np
from scipy.io import savemat
import time

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

# true coefficient matrix (up to fifth-order polynomials)
Theta,exps = calcTheta_poly_3D(t_out[1:-1],X_clean[1:-1,0],X_clean[1:-1,1],X_clean[1:-1,2],order=2)
w_true = trueCoeffMatrix_Lorenz(exps,sigma,rho,beta)

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


# ---Fourier weak SINDy (BW = 5)---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
N_freq = 100
arr_w_5 = np.zeros((w_true.shape[0],w_true.shape[1],len(arr_sig_NR),N_noise))

startTime = time.perf_counter()
for i in range(0,len(arr_sig_NR)):
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_FFT_PSD_3D(t_out,X_noisy,N_freq,params_regression,polyOrder=2,bandwidth=5)
        arr_w_5[:,:,i,j] = w_ident
        print(f"Progress 1/4: {i*N_noise+(j+1)}/{len(arr_sig_NR)*N_noise}")
endTime = time.perf_counter()
time_5 = endTime-startTime
print("!")

# ---Fourier weak SINDy (BW = 15)---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
N_freq = 100
arr_w_15 = np.zeros((w_true.shape[0],w_true.shape[1],len(arr_sig_NR),N_noise))

startTime = time.perf_counter()
for i in range(0,len(arr_sig_NR)):
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_FFT_PSD_3D(t_out,X_noisy,N_freq,params_regression,polyOrder=2,bandwidth=15)
        arr_w_15[:,:,i,j] = w_ident
        print(f"Progress 2/4: {i*N_noise+(j+1)}/{len(arr_sig_NR)*N_noise}")
endTime = time.perf_counter()
time_15 = endTime-startTime
print("!")

# ---Fourier weak SINDy (BW = 25)---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
N_freq = 100
arr_w_25 = np.zeros((w_true.shape[0],w_true.shape[1],len(arr_sig_NR),N_noise))

startTime = time.perf_counter()
for i in range(0,len(arr_sig_NR)):
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_FFT_PSD_3D(t_out,X_noisy,N_freq,params_regression,polyOrder=2,bandwidth=25)
        arr_w_25[:,:,i,j] = w_ident
        print(f"Progress 3/4: {i*N_noise+(j+1)}/{len(arr_sig_NR)*N_noise}")
endTime = time.perf_counter()
time_25 = endTime-startTime
print("!")

# ---log data---
data = dict()
data["w_true"] = w_true
data["arr_sig_NR"] = arr_sig_NR
data["arr_w_5"] = arr_w_5
data["arr_w_15"] = arr_w_15
data["arr_w_25"] = arr_w_25
data["time_5"] = time_5
data["time_15"] = time_15
data["time_25"] = time_25

savemat("Lorenz_poly2_bandwidth.mat",data)