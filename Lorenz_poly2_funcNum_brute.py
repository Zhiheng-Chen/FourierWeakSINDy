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

# ---0.2 noise level---
# generate trajectories
MD = t_sim*gridDensity*3
arr_num_func = np.linspace(20,500,25,dtype=int)
N_noise = 20   # number of noises to try at each test function number
arr_X_noisy = np.zeros((X_clean.shape[0],X_clean.shape[1],len(arr_num_func),N_noise)) # allocte a 4-way array for storing noisy trajectory data

rng = np.random.default_rng(seed=0)
for i in range(0,len(arr_num_func)):
    sig_NR = 0.2
    sig = sig_NR*np.linalg.norm(X_clean,"fro")/np.sqrt(MD)
    for j in range(0,N_noise):
        X_noisy = X_clean+rng.normal(0,sig,size=X_clean.shape)
        arr_X_noisy[:,:,i,j] = X_noisy

# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
arr_w_0P2 = np.zeros((w_true.shape[0],w_true.shape[1],len(arr_num_func),N_noise))

startTime = time.perf_counter()
for i in range(0,len(arr_num_func)):
    N_freq = arr_num_func[i]
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_Fourier_FFT_Lorenz(t_out,X_noisy,N_freq,params_regression,polyOrder=2)
        arr_w_0P2[:,:,i,j] = w_ident
        print(f"Progress 1/3: {i*N_noise+(j+1)}/{len(arr_num_func)*N_noise}")
endTime = time.perf_counter()
time_FourierWSINDyFFT = endTime-startTime
print("!")

# ---0.5 noise level---
# generate trajectories
MD = t_sim*gridDensity*3
arr_num_func = np.linspace(20,500,25,dtype=int)
N_noise = 20   # number of noises to try at each test function number
arr_X_noisy = np.zeros((X_clean.shape[0],X_clean.shape[1],len(arr_num_func),N_noise)) # allocte a 4-way array for storing noisy trajectory data

rng = np.random.default_rng(seed=0)
for i in range(0,len(arr_num_func)):
    sig_NR = 0.5
    sig = sig_NR*np.linalg.norm(X_clean,"fro")/np.sqrt(MD)
    for j in range(0,N_noise):
        X_noisy = X_clean+rng.normal(0,sig,size=X_clean.shape)
        arr_X_noisy[:,:,i,j] = X_noisy

# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
arr_w_0P5 = np.zeros((w_true.shape[0],w_true.shape[1],len(arr_num_func),N_noise))

startTime = time.perf_counter()
for i in range(0,len(arr_num_func)):
    N_freq = arr_num_func[i]
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_Fourier_FFT_Lorenz(t_out,X_noisy,N_freq,params_regression,polyOrder=2)
        arr_w_0P5[:,:,i,j] = w_ident
        print(f"Progress 2/3: {i*N_noise+(j+1)}/{len(arr_num_func)*N_noise}")
endTime = time.perf_counter()
time_FourierWSINDyFFT = endTime-startTime
print("!")

# ---0.8 noise level---
# generate trajectories
MD = t_sim*gridDensity*3
arr_num_func = np.linspace(20,500,25,dtype=int)
N_noise = 20   # number of noises to try at each test function number
arr_X_noisy = np.zeros((X_clean.shape[0],X_clean.shape[1],len(arr_num_func),N_noise)) # allocte a 4-way array for storing noisy trajectory data

rng = np.random.default_rng(seed=0)
for i in range(0,len(arr_num_func)):
    sig_NR = 0.8
    sig = sig_NR*np.linalg.norm(X_clean,"fro")/np.sqrt(MD)
    for j in range(0,N_noise):
        X_noisy = X_clean+rng.normal(0,sig,size=X_clean.shape)
        arr_X_noisy[:,:,i,j] = X_noisy

# sparse regression settings
params_regression = dict()
params_regression["method"] = "ridge"
params_regression["lambda_sparse"] = 0.5
params_regression["lambda_ridge"] = 0.001
params_regression["N_loops"] = 100

# evaluate error at different noise levels
arr_w_0P8 = np.zeros((w_true.shape[0],w_true.shape[1],len(arr_num_func),N_noise))

startTime = time.perf_counter()
for i in range(0,len(arr_num_func)):
    N_freq = arr_num_func[i]
    for j in range(0,N_noise):
        X_noisy = arr_X_noisy[:,:,i,j]
        w_ident = WSINDy_Fourier_FFT_Lorenz(t_out,X_noisy,N_freq,params_regression,polyOrder=2)
        arr_w_0P8[:,:,i,j] = w_ident
        print(f"Progress 3/3: {i*N_noise+(j+1)}/{len(arr_num_func)*N_noise}")
endTime = time.perf_counter()
time_FourierWSINDyFFT = endTime-startTime
print("!")

# ---log data---
data = dict()
data["w_true"] = w_true
data["arr_num_func"] = arr_num_func
data["arr_w_0P2"] = arr_w_0P2
data["arr_w_0P5"] = arr_w_0P5
data["arr_w_0P8"] = arr_w_0P8

savemat("Lorenz_poly2_funcNum_brute.mat",data)