# ---packages---
from SINDyFunctions import *
import numpy as np
from scipy.io import savemat
import time

# ---simulate Lorenz system and add noise---
# system parameters
a = 10
b = 2.667
c = 28
d = 1.1

# simulate
X0 = np.array([5,8,12,21])
t_sim = 10
gridDensity = 1000
t_span = [0,t_sim]
t_out,X_clean = simulateHyperLorenz(X0,t_sim,gridDensity,a,b,c,d)

# true coefficient matrix (up to fifth-order polynomials)
Theta,exps = calcTheta_poly_4D(t_out[1:-1],X_clean[1:-1,0],X_clean[1:-1,1],X_clean[1:-1,2],X_clean[1:-1,3],order=2)
w_true = trueCoeffMatrix_hyperLorenz(exps,a,b,c,d)

# add noise
MD = t_sim*gridDensity*4
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

# ---Fourier weak SINDy (FFT accelerated)---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
N_freq = 500
arr_w_FFTWSINDy = np.zeros((w_true.shape[0],w_true.shape[1],len(arr_sig_NR),N_noise))

startTime = time.perf_counter()
for i in range(0,len(arr_sig_NR)):
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_Fourier_FFT_4D(t_out,X_noisy,N_freq,params_regression,polyOrder=2)
        arr_w_FFTWSINDy[:,:,i,j] = w_ident
        print(f"Progress: {i*N_noise+(j+1)}/{len(arr_sig_NR)*N_noise}")
endTime = time.perf_counter()
time_FourierWSINDyFFT = endTime-startTime
print("!")

# ---log data---
data = dict()
data["arr_sig_NR"] = arr_sig_NR
data["arr_w_FFTWSINDy"] = arr_w_FFTWSINDy

savemat("hyperLorenz_poly2_bruteFFT.mat",data)