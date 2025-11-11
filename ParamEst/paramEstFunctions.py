import numpy as np
import itertools
import scipy
from scipy.signal import welch
from mne.time_frequency import psd_array_multitaper
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

from SINDyFunctions import getFreqInds_PSD_multiTaper,constructLS_FFT_PSD

def calcTheta_Lorenz(arr_t,arr_x,arr_y,arr_z):
    N = len(arr_t)
    x = arr_x
    y = arr_y
    z = arr_z
    
    Theta_x = np.zeros((N,2))
    Theta_x[:,0] = x
    Theta_x[:,1] = y

    Theta_y = np.zeros((N,3))
    Theta_y[:,0] = x
    Theta_y[:,1] = y
    Theta_y[:,2] = x*z

    Theta_z = np.zeros((N,2))
    Theta_z[:,0] = z
    Theta_z[:,1] = x*y
    
    return Theta_x,Theta_y,Theta_z

def FourierParamEst_Lorenz(t_out,X_out,N_freq=100,bandwidth=15,lambda_ridge=0.001):
    omega0 = 2*np.pi/(t_out[-1]-t_out[0])
    arr_x = X_out[:,0]
    arr_y = X_out[:,1]
    arr_z = X_out[:,2]
    # set n-values
    arr_n_x = getFreqInds_PSD_multiTaper(t_out,arr_x,N_freq,bandwidth=bandwidth)
    arr_n_y = getFreqInds_PSD_multiTaper(t_out,arr_y,N_freq,bandwidth=bandwidth)
    arr_n_z = getFreqInds_PSD_multiTaper(t_out,arr_z,N_freq,bandwidth=bandwidth)
    # library
    Theta_x,Theta_y,Theta_z = calcTheta_Lorenz(t_out,arr_x,arr_y,arr_z)
    # A and b
    A1,b1 = constructLS_FFT_PSD(t_out,arr_x,arr_n_x,Theta_x)
    A2,b2 = constructLS_FFT_PSD(t_out,arr_y,arr_n_y,Theta_y)
    A3,b3 = constructLS_FFT_PSD(t_out,arr_z,arr_n_z,Theta_z)
    # ridge regression
    w_x = np.linalg.solve(A1.T@A1+lambda_ridge*np.eye(A1.shape[1]),A1.T@b1)
    w_x = w_x.flatten()
    w_y = np.linalg.solve(A2.T@A2+lambda_ridge*np.eye(A2.shape[1]),A2.T@b2)
    w_y = w_y.flatten()
    w_z = np.linalg.solve(A3.T@A3+lambda_ridge*np.eye(A3.shape[1]),A3.T@b3)
    w_z = w_z.flatten()
    w_ident = np.hstack((w_x,w_y,w_z))
    return w_x,w_y,w_z,w_ident

def errorEval(w_true,w_ident):
    e = w_true-w_ident
    errorNorm = np.linalg.norm(e)
    errorNorm_rel = errorNorm/np.linalg.norm(w_true)
    return errorNorm_rel

def batchErrorEval(w_true,arr_w):
    n_noiseLevels = arr_w.shape[1]
    N_noise = arr_w.shape[2]

    error_all = np.zeros((n_noiseLevels,N_noise))

    # loop over noise levels and trials
    for i in range(n_noiseLevels):
        for j in range(N_noise):
            w_ident = arr_w[:,i,j]
            eNorm = errorEval(w_true,w_ident)
            error_all[i,j] = eNorm

    # compute statistics
    error_mean = np.median(error_all,axis=1)
    error_q1 = np.percentile(error_all,25,axis=1)
    error_q3 = np.percentile(error_all,75,axis=1)

    results = {
        "error_mean": error_mean,
        "error_q1": error_q1,
        "error_q3": error_q3,
    }

    return results

def plotWithQuartiles(x,mean,q1,q3,label,color,ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(x,mean,label=label,color=color)
    ax.fill_between(x,q1,q3,color=color,alpha=0.5)
