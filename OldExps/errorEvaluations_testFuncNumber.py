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
_,exps = calcTheta_poly_3D(t_out[1:-1],X_clean[1:-1,0],X_clean[1:-1,1],X_clean[1:-1,2],order=5)
w_true_5 = trueCoeffMatrix_Lorenz(exps,sigma,rho,beta)

_,exps = calcTheta_poly_3D(t_out[1:-1],X_clean[1:-1,0],X_clean[1:-1,1],X_clean[1:-1,2],order=2)
w_true_2 = trueCoeffMatrix_Lorenz(exps,sigma,rho,beta)

# add noise
MD = t_sim*gridDensity*3
arr_testFuncNum_Fourier = np.array([20,40,60,100,200,300,400,500,800,1000,1500,2000])   # array of number of test functions for bump weak SINDy
N_noise = 200   # number of noises to try at each noise level
arr_X_noisy = np.zeros((X_clean.shape[0],X_clean.shape[1],len(arr_testFuncNum_Fourier),N_noise)) # allocte a 4-way array for storing noisy trajectory data

for i in range(0,len(arr_testFuncNum_Fourier)):
    sig_NR = 1e-6
    sig = sig_NR*np.linalg.norm(X_clean,"fro")/np.sqrt(MD)
    rng = np.random.default_rng(seed=0)
    for j in range(0,N_noise):
        X_noisy = X_clean+rng.normal(0,sig,size=X_clean.shape)
        arr_X_noisy[:,:,i,j] = X_noisy

# ---FFT WSINDy, poly 5---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error with different numbers of test functions, poly 5
arr_w_poly5 = np.zeros((w_true_5.shape[0],w_true_5.shape[1],len(arr_testFuncNum_Fourier),N_noise))
for i in range(0,len(arr_testFuncNum_Fourier)):
    N_freq = arr_testFuncNum_Fourier[i]
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_Fourier_FFT_Lorenz(t_out,X_noisy,N_freq,params_regression,polyOrder=5)
        arr_w_poly5[:,:,i,j] = w_ident
        print(f"Progress 1/2: {i*N_noise+(j+1)}/{len(arr_testFuncNum_Fourier)*N_noise}")
print("!")

# evaluate error with different numbers of test functions, poly 2
arr_w_poly2 = np.zeros((w_true_2.shape[0],w_true_2.shape[1],len(arr_testFuncNum_Fourier),N_noise))
for i in range(0,len(arr_testFuncNum_Fourier)):
    N_freq = arr_testFuncNum_Fourier[i]
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_Fourier_FFT_Lorenz(t_out,X_noisy,N_freq,params_regression,polyOrder=2)
        arr_w_poly2[:,:,i,j] = w_ident
        print(f"Progress 2/2: {i*N_noise+(j+1)}/{len(arr_testFuncNum_Fourier)*N_noise}")
print("!")

# ---log data---
data = dict()
data["arr_testFuncNum_Fourier"] = arr_testFuncNum_Fourier
data["arr_w_poly5"] = arr_w_poly5
data["arr_w_poly2"] = arr_w_poly2
data["w_true_5"] = w_true_5
data["w_true_2"] = w_true_2

savemat("results_testFuncNum.mat",data)