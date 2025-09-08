# ---packages---
from SINDyFunctions import *
import numpy as np
from scipy.io import savemat
import time

# ---simulate Lorenz system and add noise---
# system parameters
a = 0.2
b = 0.2
c = 5.7

# true coefficient matrix (with 1, linear, and quadratic basis functions)
w_true = np.zeros((10,3))
w_true[2,0] = -1
w_true[3,0] = -1
w_true[1,1] = 1
w_true[2,1] = a
w_true[0,2] = b
w_true[3,2] = -c
w_true[9,2] = 1

# simulate
X0 = np.array([1,1,1])
t_sim = 30
gridDensity = 1000
t_span = [0,t_sim]
t_out,X_clean,X_dot_out = simulateRosslerSystem(X0,t_sim,gridDensity,a,b,c)

# add noise
MD = t_sim*gridDensity*3
arr_sig_NR = np.hstack((np.logspace(-6,-2,5),np.linspace(0.05,0.4,8)))   # array of noise ratios
N_noise = 200   # number of noises to try at each noise level
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
params_regression["lambda_sparse"] = 0.01
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
arr_relError_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean and standard dev. of relative error norms for each noise level
arr_relError_std = np.zeros((1,len(arr_sig_NR)))
arr_TPR_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean and standard dev. of TPR for each noise level
arr_TPR_std = np.zeros((1,len(arr_sig_NR)))

startTime = time.perf_counter()
for i in range(0,len(arr_sig_NR)):
    arr_relError = np.zeros((1,N_noise))   # array of relative error norms for N noises with current noise level
    arr_TPR = np.zeros((1,N_noise)) # array of TPRs for N noises with current noise level
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = SINDy_Lorenz(t_out,X_noisy,params_regression)
        errorNorm_rel,TPR = errorEval(w_true,w_ident)
        arr_relError[0,j] = errorNorm_rel
        arr_TPR[0,j] = TPR
        print(f"Progress 1/4: {i*N_noise+(j+1)}/{len(arr_sig_NR)*N_noise}")
    arr_relError_mean[0,i] = np.average(arr_relError)
    arr_relError_std[0,i] = np.std(arr_relError)
    arr_TPR_mean[0,i] = np.average(arr_TPR)
    arr_TPR_std[0,i] = np.std(arr_TPR)
errorMean_SINDy_Rossler = arr_relError_mean.flatten()
errorStD_SINDy_Rossler = arr_relError_std
TPRMean_SINDy_Rossler = arr_TPR_mean.flatten()
TPRStD_SINDy_Rossler = arr_TPR_std.flatten()

endTime = time.perf_counter()
time_SINDy = endTime-startTime
print("!")

# ---bump weak SINDy (manual threshold)---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.01
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
arr_relError_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean and standard dev. of relative error norms for each noise level
arr_relError_std = np.zeros((1,len(arr_sig_NR)))
arr_TPR_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean and standard dev. of TPR for each noise level
arr_TPR_std = np.zeros((1,len(arr_sig_NR)))

startTime = time.perf_counter()
for i in range(0,len(arr_sig_NR)):
    arr_relError = np.zeros((1,N_noise))   # array of relative error norms for N noises with current noise level
    arr_TPR = np.zeros((1,N_noise)) # array of TPRs for N noises with current noise level
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_bump_Lorenz(t_out,X_noisy,20,20,params_regression)
        errorNorm_rel,TPR = errorEval(w_true,w_ident)
        arr_relError[0,j] = errorNorm_rel
        arr_TPR[0,j] = TPR
        print(f"Progress 2/4: {i*N_noise+(j+1)}/{len(arr_sig_NR)*N_noise}")
    arr_relError_mean[0,i] = np.average(arr_relError)
    arr_relError_std[0,i] = np.std(arr_relError)
    arr_TPR_mean[0,i] = np.average(arr_TPR)
    arr_TPR_std[0,i] = np.std(arr_TPR)
errorMean_bumpWSINDy_Rossler = arr_relError_mean.flatten()
errorStD_bumpWSINDy_Rossler = arr_relError_std
TPRMean_bumpWSINDy_Rossler = arr_TPR_mean.flatten()
TPRStD_bumpWSINDy_Rossler = arr_TPR_std.flatten()

endTime = time.perf_counter()
time_bumpWSINDy = endTime-startTime
print("!")

# ---Fourier weak SINDy (manual threshold)---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.01
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
N_freq = 60

arr_relError_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean and standard dev. of relative error norms for each noise level
arr_relError_std = np.zeros((1,len(arr_sig_NR)))
arr_TPR_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean and standard dev. of TPR for each noise level
arr_TPR_std = np.zeros((1,len(arr_sig_NR)))

startTime = time.perf_counter()
for i in range(0,len(arr_sig_NR)):
    arr_relError = np.zeros((1,N_noise))   # array of relative error norms for N noises with current noise level
    arr_TPR = np.zeros((1,N_noise)) # array of TPRs for N noises with current noise level
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_Fourier_Lorenz(t_out,X_noisy,N_freq,params_regression)
        errorNorm_rel,TPR = errorEval(w_true,w_ident)
        arr_relError[0,j] = errorNorm_rel
        arr_TPR[0,j] = TPR
        print(f"Progress 3/4: {i*N_noise+(j+1)}/{len(arr_sig_NR)*N_noise}")
    arr_relError_mean[0,i] = np.average(arr_relError)
    arr_relError_std[0,i] = np.std(arr_relError)
    arr_TPR_mean[0,i] = np.average(arr_TPR)
    arr_TPR_std[0,i] = np.std(arr_TPR)
errorMean_FourierWSINDy_Rossler = arr_relError_mean.flatten()
errorStD_FourierWSINDy_Rossler = arr_relError_std
TPRMean_FourierWSINDy_Rossler = arr_TPR_mean.flatten()
TPRStD_FourierWSINDy_Rossler = arr_TPR_std.flatten()

endTime = time.perf_counter()
time_FourierWSINDy = endTime-startTime
print("!")

# ---Fourier weak SINDy (FFT accelerated)---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.01
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
N_freq = 60

arr_relError_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean and standard dev. of relative error norms for each noise level
arr_relError_std = np.zeros((1,len(arr_sig_NR)))
arr_TPR_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean and standard dev. of TPR for each noise level
arr_TPR_std = np.zeros((1,len(arr_sig_NR)))

startTime = time.perf_counter()
for i in range(0,len(arr_sig_NR)):
    arr_relError = np.zeros((1,N_noise))   # array of relative error norms for N noises with current noise level
    arr_TPR = np.zeros((1,N_noise)) # array of TPRs for N noises with current noise level
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_Fourier_FFT_Lorenz(t_out,X_noisy,N_freq,params_regression)
        errorNorm_rel,TPR = errorEval(w_true,w_ident)
        arr_relError[0,j] = errorNorm_rel
        arr_TPR[0,j] = TPR
        print(f"Progress 4/4: {i*N_noise+(j+1)}/{len(arr_sig_NR)*N_noise}")
    arr_relError_mean[0,i] = np.average(arr_relError)
    arr_relError_std[0,i] = np.std(arr_relError)
    arr_TPR_mean[0,i] = np.average(arr_TPR)
    arr_TPR_std[0,i] = np.std(arr_TPR)
errorMean_FourierWSINDyFFT_Rossler = arr_relError_mean.flatten()
errorStD_FourierWSINDyFFT_Rossler = arr_relError_std
TPRMean_FourierWSINDyFFT_Rossler = arr_TPR_mean.flatten()
TPRStD_FourierWSINDyFFT_Rossler = arr_TPR_std.flatten()

endTime = time.perf_counter()
time_FourierWSINDyFFT = endTime-startTime
print("!")

# ---log data---
data = dict()
data["arr_sig_NR"] = arr_sig_NR
data["errorMean_SINDy_Rossler"] = errorMean_SINDy_Rossler
data["errorStD_SINDy_Rossler"] = errorStD_SINDy_Rossler
data["TPRMean_SINDy_Rossler"] = TPRMean_SINDy_Rossler
data["TPRStD_SINDy_Rossler"] = TPRStD_SINDy_Rossler
data["errorMean_bumpWSINDy_Rossler"] = errorMean_bumpWSINDy_Rossler
data["errorStD_bumpWSINDy_Rossler"] = errorStD_bumpWSINDy_Rossler
data["TPRMean_bumpWSINDy_Rossler"] = TPRMean_bumpWSINDy_Rossler
data["TPRStD_bumpWSINDy_Rossler"] = TPRStD_bumpWSINDy_Rossler
data["errorMean_FourierWSINDy_Rossler"] = errorMean_FourierWSINDy_Rossler
data["errorStD_FourierWSINDy_Rossler"] = errorStD_FourierWSINDy_Rossler
data["TPRMean_FourierWSINDy_Rossler"] = TPRMean_FourierWSINDy_Rossler
data["TPRStD_FourierWSINDy_Rossler"] = TPRStD_FourierWSINDy_Rossler
data["errorMean_FourierWSINDyFFT_Rossler"] = errorMean_FourierWSINDyFFT_Rossler
data["errorStD_FourierWSINDyFFT_Rossler"] = errorStD_FourierWSINDyFFT_Rossler
data["TPRMean_FourierWSINDyFFT_Rossler"] = TPRMean_FourierWSINDyFFT_Rossler
data["TPRStD_FourierWSINDyFFT_Rossler"] = TPRStD_FourierWSINDyFFT_Rossler
data["time_SINDy"] = time_SINDy
data["time_bumpWSINDy"] = time_bumpWSINDy
data["time_FourierWSINDy"] = time_FourierWSINDy
data["time_FourierWSINDyFFT"] = time_FourierWSINDyFFT

savemat("results_noiseLevel_Rossler.mat",data)