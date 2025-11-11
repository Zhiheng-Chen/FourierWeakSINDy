from SINDyFunctions import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def rgb(r,g,b):
    return (r/255,g/255,b/255)

c = [rgb(235, 172, 35), rgb(184, 0, 88), rgb(0, 140, 249),  rgb(89, 84, 214),rgb(0, 187, 173), rgb(209, 99, 230), rgb(178, 69, 2), rgb(255, 146, 135), rgb(0, 110, 0), rgb(0, 198, 248), rgb(135, 133, 0), rgb(0, 167, 108), rgb(189, 189, 189)]
s = ["-","--","-.",":"]

# ---Lorenz 5, hyperLorenz 3, hyperJha 3---
plt.rcParams.update({'font.size': 12,'legend.fontsize': 11})
fig,axs = plt.subplots(2,3,figsize=(15,7))

# Lorenz 5
## extract data
data = loadmat("ResultsLog/Lorenz_poly5.mat")
data_ps = loadmat("ResultsLog/Lorenz_poly5_ps.mat")
# data_brute = loadmat("ResultsLog/Lorenz_poly2_bruteFFT")
arr_sig_NR = data["arr_sig_NR"].flatten()
arr_w_SINDy = data_ps["arr_w_SINDy"]
arr_w_bump = data_ps["arr_w_WSINDy"]
arr_w_fft = data["arr_w_FFTWSINDy"]
# arr_w_brute = data_brute["arr_w_FFTWSINDy"]
w_true = data["w_true"]
w_true_ps = data_ps["w_true"]

## evaluate error
res_SINDy = batchErrorEval(w_true_ps, arr_w_SINDy)
res_bump = batchErrorEval(w_true_ps, arr_w_bump)
res_fft = batchErrorEval(w_true, arr_w_fft)
# res_brute = batchErrorEval(w_true,arr_w_brute)

## plot error
ax = axs[0,0]
plotWithQuartiles(arr_sig_NR, res_SINDy["error_mean"], res_SINDy["error_q1"], res_SINDy["error_q3"], "SINDy (PySINDy)", c[7], linestyle = s[0], ax=ax)
plotWithQuartiles(arr_sig_NR, res_bump["error_mean"], res_bump["error_q1"], res_bump["error_q3"], "WSINDy (PySINDy)", c[1], linestyle = s[1], ax=ax)
# plotWithQuartiles(arr_sig_NR, res_brute["error_mean"], res_brute["error_q1"], res_brute["error_q3"], "Fourier WSINDy (Sweep)", c[2], linestyle = s[2], ax=ax)
plotWithQuartiles(arr_sig_NR, res_fft["error_mean"], res_fft["error_q1"], res_fft["error_q3"], "Fourier WSINDy (SDE)", c[3], linestyle = s[3], ax=ax)
ax.set_ylim(10**-3,10**1.5)
ax.set_yscale("log")
ax.set_xticks([10**-6, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
# ax.set_xlabel("Noise Level")
ax.set_ylabel("Relative Coefficient Error",fontsize=14)
ax.grid(True)
ax.legend()
ax.set_title("(a)")

## plot TPR
ax = axs[1,0]
ax.set_ylim(0,1.0)
plotWithQuartiles(arr_sig_NR, res_SINDy["TPR_mean"], res_SINDy["TPR_q1"], res_SINDy["TPR_q3"], "SINDy (PySINDy)", c[7], linestyle = s[0], ax=ax)
plotWithQuartiles(arr_sig_NR, res_bump["TPR_mean"], res_bump["TPR_q1"], res_bump["TPR_q3"], "WSINDy (PySINDy)", c[1], linestyle = s[1], ax=ax)
# plotWithQuartiles(arr_sig_NR, res_brute["TPR_mean"], res_brute["TPR_q1"], res_brute["TPR_q3"], "Fourier WSINDy (Sweep)", c[2], linestyle = s[2], ax=ax)
plotWithQuartiles(arr_sig_NR, res_fft["TPR_mean"], res_fft["TPR_q1"], res_fft["TPR_q3"], "Fourier WSINDy (SDE)", c[3], linestyle = s[3], ax=ax)
ax.set_xticks([10**-6, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.set_xlabel("Noise Level",fontsize=14)
ax.set_ylabel("True Positive Ratio (TPR)",fontsize=14)
ax.grid(True)
# ax.legend()

# hyper Lorenz 3
## extract data
data = loadmat("ResultsLog/hyperLorenz_poly3.mat")
data_ps = loadmat("ResultsLog/hyperLorenz_poly3_ps.mat")
# data_brute = loadmat("ResultsLog/Lorenz_poly2_bruteFFT")
arr_sig_NR = data["arr_sig_NR"].flatten()[:13]
arr_w_SINDy = data_ps["arr_w_SINDy"][:,:,:13,:]
arr_w_bump = data_ps["arr_w_WSINDy"][:,:,:13,:]
arr_w_fft = data["arr_w_FFTWSINDy"][:,:,:13,:]
# arr_w_brute = data_brute["arr_w_FFTWSINDy"]
w_true = data["w_true"]
w_true_ps = data_ps["w_true"]

## evaluate error
res_SINDy = batchErrorEval(w_true_ps, arr_w_SINDy)
res_bump = batchErrorEval(w_true_ps, arr_w_bump)
res_fft = batchErrorEval(w_true, arr_w_fft)
# res_brute = batchErrorEval(w_true,arr_w_brute)

## plot error
ax = axs[0,1]
plotWithQuartiles(arr_sig_NR, res_SINDy["error_mean"], res_SINDy["error_q1"], res_SINDy["error_q3"], "SINDy (PySINDy)", c[7], linestyle = s[0], ax=ax)
plotWithQuartiles(arr_sig_NR, res_bump["error_mean"], res_bump["error_q1"], res_bump["error_q3"], "WSINDy (PySINDy)", c[1], linestyle = s[1], ax=ax)
# plotWithQuartiles(arr_sig_NR, res_brute["error_mean"], res_brute["error_q1"], res_brute["error_q3"], "Fourier WSINDy (Sweep)", c[2], linestyle = s[2], ax=ax)
plotWithQuartiles(arr_sig_NR, res_fft["error_mean"], res_fft["error_q1"], res_fft["error_q3"], "Fourier WSINDy (SDE)", c[3], linestyle = s[3], ax=ax)
ax.set_ylim(10**-3,10**1.5)
ax.set_yscale("log")
ax.set_xticks([10**-6, 0.25, 0.5])
ax.set_xticklabels(['0%', '25%', '50%'])
# ax.set_xlabel("Noise Level")
# ax.set_ylabel("Relative Coefficient Error",fontsize=14)
ax.grid(True)
ax.legend()
ax.set_title("(b)")

## plot TPR
ax = axs[1,1]
ax.set_ylim(0,1.0)
plotWithQuartiles(arr_sig_NR, res_SINDy["TPR_mean"], res_SINDy["TPR_q1"], res_SINDy["TPR_q3"], "SINDy (PySINDy)", c[7], linestyle = s[0], ax=ax)
plotWithQuartiles(arr_sig_NR, res_bump["TPR_mean"], res_bump["TPR_q1"], res_bump["TPR_q3"], "WSINDy (PySINDy)", c[1], linestyle = s[1], ax=ax)
# plotWithQuartiles(arr_sig_NR, res_brute["TPR_mean"], res_brute["TPR_q1"], res_brute["TPR_q3"], "Fourier WSINDy (Sweep)", c[2], linestyle = s[2], ax=ax)
plotWithQuartiles(arr_sig_NR, res_fft["TPR_mean"], res_fft["TPR_q1"], res_fft["TPR_q3"], "Fourier WSINDy (SDE)", c[3], linestyle = s[3], ax=ax)
ax.set_xticks([10**-6, 0.25, 0.5])
ax.set_xticklabels(['0%', '25%', '50%'])
ax.set_xlabel("Noise Level",fontsize=14)
# ax.set_ylabel("True Positive Ratio (TPR)",fontsize=14)
ax.grid(True)
# ax.legend()

# hyper Jha 3
## extract data
data = loadmat("ResultsLog/hyperJha_poly3.mat")
data_ps = loadmat("ResultsLog/hyperJha_poly3_ps.mat")
# data_brute = loadmat("ResultsLog/Lorenz_poly2_bruteFFT")
arr_sig_NR = data["arr_sig_NR"].flatten()[:13]
arr_w_SINDy = data_ps["arr_w_SINDy"][:,:,:13,:]
arr_w_bump = data_ps["arr_w_WSINDy"][:,:,:13,:]
arr_w_fft = data["arr_w_FFTWSINDy"][:,:,:13,:]
# arr_w_brute = data_brute["arr_w_FFTWSINDy"]
w_true = data["w_true"]
w_true_ps = data_ps["w_true"]

## evaluate error
res_SINDy = batchErrorEval(w_true_ps, arr_w_SINDy)
res_bump = batchErrorEval(w_true_ps, arr_w_bump)
res_fft = batchErrorEval(w_true, arr_w_fft)
# res_brute = batchErrorEval(w_true,arr_w_brute)

## plot error
ax = axs[0,2]
plotWithQuartiles(arr_sig_NR, res_SINDy["error_mean"], res_SINDy["error_q1"], res_SINDy["error_q3"], "SINDy (PySINDy)", c[7], linestyle = s[0], ax=ax)
plotWithQuartiles(arr_sig_NR, res_bump["error_mean"], res_bump["error_q1"], res_bump["error_q3"], "WSINDy (PySINDy)", c[1], linestyle = s[1], ax=ax)
# plotWithQuartiles(arr_sig_NR, res_brute["error_mean"], res_brute["error_q1"], res_brute["error_q3"], "Fourier WSINDy (Sweep)", c[2], linestyle = s[2], ax=ax)
plotWithQuartiles(arr_sig_NR, res_fft["error_mean"], res_fft["error_q1"], res_fft["error_q3"], "Fourier WSINDy (SDE)", c[3], linestyle = s[3], ax=ax)
ax.set_ylim(10**-3,10**1.5)
ax.set_yscale("log")
ax.set_xticks([10**-6, 0.25, 0.5])
ax.set_xticklabels(['0%', '25%', '50%'])
# ax.set_xlabel("Noise Level")
# ax.set_ylabel("Relative Coefficient Error",fontsize=14)
ax.grid(True)
ax.legend()
ax.set_title("(c)")

## plot TPR
ax = axs[1,2]
ax.set_ylim(0,1.0)
plotWithQuartiles(arr_sig_NR, res_SINDy["TPR_mean"], res_SINDy["TPR_q1"], res_SINDy["TPR_q3"], "SINDy (PySINDy)", c[7], linestyle = s[0], ax=ax)
plotWithQuartiles(arr_sig_NR, res_bump["TPR_mean"], res_bump["TPR_q1"], res_bump["TPR_q3"], "WSINDy (PySINDy)", c[1], linestyle = s[1], ax=ax)
# plotWithQuartiles(arr_sig_NR, res_brute["TPR_mean"], res_brute["TPR_q1"], res_brute["TPR_q3"], "Fourier WSINDy (Sweep)", c[2], linestyle = s[2], ax=ax)
plotWithQuartiles(arr_sig_NR, res_fft["TPR_mean"], res_fft["TPR_q1"], res_fft["TPR_q3"], "Fourier WSINDy (SDE)", c[3], linestyle = s[3], ax=ax)
ax.set_xticks([10**-6, 0.25, 0.5])
ax.set_xticklabels(['0%', '25%', '50%'])
ax.set_xlabel("Noise Level",fontsize=14)
# ax.set_ylabel("True Positive Ratio (TPR)",fontsize=14)
ax.grid(True)
# ax.legend()

plt.tight_layout()
plt.savefig("HigherOrderPoly.pdf", bbox_inches="tight")