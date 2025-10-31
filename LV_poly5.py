# ---packages---
from SINDyFunctions import *
import numpy as np
from scipy.io import savemat
import time

# ---simulate VDP system and add noise---
# system parameters
beta = 1

# simulate
X0 = np.array([1,1])
t_sim = 10
gridDensity = 1000
t_span = [0,t_sim]
t_out,X_clean,XD_out = simulateLV(X0,t_sim,gridDensity,beta)

# true coefficient matrix
_,exps = calcTheta_poly_2D(t_out[1:-1],X_clean[1:-1,0],X_clean[1:-1,1])
w_true = trueCoeffMatrix_LV(exps,beta)

# add noise
MD = t_sim*gridDensity*2
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

# ---regular SINDy (manual threshold)---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
arr_w_SINDy = np.zeros((w_true.shape[0],w_true.shape[1],len(arr_sig_NR),N_noise))

startTime = time.perf_counter()
for i in range(0,len(arr_sig_NR)):
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = SINDy_LV(t_out,X_noisy,params_regression,polyOrder=5)
        arr_w_SINDy[:,:,i,j] = w_ident
        print(f"Progress 1/3: {i*N_noise+(j+1)}/{len(arr_sig_NR)*N_noise}")
endTime = time.perf_counter()
time_SINDy = endTime-startTime
print("!")

# ---bump weak SINDy (manual threshold)---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.3
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
arr_w_bumpWSINDy = np.zeros((w_true.shape[0],w_true.shape[1],len(arr_sig_NR),N_noise))

startTime = time.perf_counter()
for i in range(0,len(arr_sig_NR)):
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_bump_LV(t_out,X_noisy,20,20,params_regression,polyOrder=5)
        arr_w_bumpWSINDy[:,:,i,j] = w_ident
        print(f"Progress 2/3: {i*N_noise+(j+1)}/{len(arr_sig_NR)*N_noise}")
endTime = time.perf_counter()
time_bumpWSINDy = endTime-startTime
print("!")

# ---Fourier weak SINDy (FFT accelerated)---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
N_freq = 100
arr_w_FFTWSINDy = np.zeros((w_true.shape[0],w_true.shape[1],len(arr_sig_NR),N_noise))

startTime = time.perf_counter()
for i in range(0,len(arr_sig_NR)):
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_FFT_PSD_2D(t_out,X_noisy,N_freq,params_regression,polyOrder=5)
        arr_w_FFTWSINDy[:,:,i,j] = w_ident
        print(f"Progress 3/3: {i*N_noise+(j+1)}/{len(arr_sig_NR)*N_noise}")
dTime = time.perf_counter()
time_FourierWSINDyFFT = endTime-startTime
print("!")

# ---log data---
data = dict()
data["arr_sig_NR"] = arr_sig_NR
data["arr_w_SINDy"] = arr_w_SINDy
data["arr_w_bumpWSINDy"] = arr_w_bumpWSINDy
data["arr_w_FFTWSINDy"] = arr_w_FFTWSINDy
data["time_SINDy"] = time_SINDy
data["time_bumpWSINDy"] = time_bumpWSINDy
data["time_FourierWSINDyFFT"] = time_FourierWSINDyFFT
data["w_true"] = w_true

savemat("LV_poly5.mat",data)