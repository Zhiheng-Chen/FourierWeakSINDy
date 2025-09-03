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

# add noise
MD = t_sim*gridDensity
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
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
arr_relError_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean relative error norms for each noise level
arr_TPR_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean TPR for each noise level
for i in range(0,len(arr_sig_NR)):
    arr_relError = np.zeros((1,N_noise))   # array of relative error norms for N noises with current noise level
    arr_TPR = np.zeros((1,N_noise)) # array of TPRs for N noises with current noise level
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = SINDy_Lorenz(t_out,X_noisy,params_regression)
        errorNorm_rel,TPR = errorEval(w_true,w_ident)
        arr_relError[0,j] = errorNorm_rel
        arr_TPR[0,j] = TPR
    arr_relError_mean[0,i] = np.average(arr_relError)
    arr_TPR_mean[0,i] = np.average(arr_TPR)
error_SINDy_manual = arr_relError_mean.flatten()
TPR_SINDy_manual = arr_TPR_mean.flatten()
print("!")

# ---bump weak SINDy (manual threshold)---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
arr_relError_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean relative error norms for each noise level
arr_TPR_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean TPR for each noise level
for i in range(0,len(arr_sig_NR)):
    arr_relError = np.zeros((1,N_noise))   # array of relative error norms for N noises with current noise level
    arr_TPR = np.zeros((1,N_noise)) # array of TPRs for N noises with current noise level
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_bump_Lorenz(t_out,X_noisy,20,20,params_regression)
        errorNorm_rel,TPR = errorEval(w_true,w_ident)
        arr_relError[0,j] = errorNorm_rel
        arr_TPR[0,j] = TPR
    arr_relError_mean[0,i] = np.average(arr_relError)
    arr_TPR_mean[0,i] = np.average(arr_TPR)
error_bumpWSINDy_manual = arr_relError_mean.flatten()
TPR_bumpWSINDy_manual = arr_TPR_mean.flatten()
print("!")

# ---Fourier weak SINDy (manual threshold)---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
N_freq = 30
arr_relError_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean relative error norms for each noise level
arr_TPR_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean TPR for each noise level
for i in range(0,len(arr_sig_NR)):
    arr_relError = np.zeros((1,N_noise))   # array of relative error norms for N noises with current noise level
    arr_TPR = np.zeros((1,N_noise)) # array of TPRs for N noises with current noise level
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_Fourier_Lorenz(t_out,X_noisy,N_freq,params_regression)
        errorNorm_rel,TPR = errorEval(w_true,w_ident)
        arr_relError[0,j] = errorNorm_rel
        arr_TPR[0,j] = TPR
    arr_relError_mean[0,i] = np.average(arr_relError)
    arr_TPR_mean[0,i] = np.average(arr_TPR)
error_FourierWSINDy_manual = arr_relError_mean.flatten()
TPR_FourierWSINDy_manual = arr_TPR_mean.flatten()
print("!")

# ---Fourier weak SINDy (FFT accelerated)---
# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
N_freq = 60
arr_relError_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean relative error norms for each noise level
arr_TPR_mean = np.zeros((1,len(arr_sig_NR))) # allocate array of mean TPR for each noise level
for i in range(0,len(arr_sig_NR)):
    arr_relError = np.zeros((1,N_noise))   # array of relative error norms for N noises with current noise level
    arr_TPR = np.zeros((1,N_noise)) # array of TPRs for N noises with current noise level
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_Fourier_FFT_Lorenz(t_out,X_noisy,N_freq,params_regression)
        errorNorm_rel,TPR = errorEval(w_true,w_ident)
        arr_relError[0,j] = errorNorm_rel
        arr_TPR[0,j] = TPR
    arr_relError_mean[0,i] = np.average(arr_relError)
    arr_TPR_mean[0,i] = np.average(arr_TPR)
error_FourierWSINDy_FFT = arr_relError_mean.flatten()
TPR_FourierWSINDy_FFT = arr_TPR_mean.flatten()
print("!")

# ---log data---
data = dict()
data["arr_sig_NR"] = arr_sig_NR
data["error_SINDy_manual"] = error_SINDy_manual
data["TPR_SINDy_manual"] = TPR_SINDy_manual
data["error_bumpWSINDy_manual"] = error_bumpWSINDy_manual
data["TPR_bumpWSINDy_manual"] = TPR_bumpWSINDy_manual
data["error_FourierWSINDy_manual"] = error_FourierWSINDy_manual
data["TPR_FourierWSINDy_manual"] = TPR_FourierWSINDy_manual
data["error_FourierWSINDy_FFT"] = error_FourierWSINDy_FFT
data["TPR_FourierWSINDy_FFT"] = TPR_FourierWSINDy_FFT

savemat("results_noiseLevel.mat",data)