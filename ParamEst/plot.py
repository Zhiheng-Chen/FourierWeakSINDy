from paramEstFunctions import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

t_sim = 10
gridDensity = 1000  # number of steps in one second
MD = t_sim*gridDensity*3
arr_sig_NR = np.hstack((np.logspace(-6,-2,5),np.linspace(0.05,1,15)))   # array of noise ratios

data = loadmat("ResultsLog/Lorenz_paramEst.mat")
arr_w = data["arr_w"]
w_true = data["w_true"]

res = batchErrorEval(w_true, arr_w)

plt.rcParams.update({'font.size': 15,'legend.fontsize': 13})
fig,ax = plt.subplots(1,1,figsize=(15,7))
plotWithQuartiles(arr_sig_NR, res["error_mean"], res["error_q1"], res["error_q3"], "SINDy (PySINDy)", "C0", ax=ax)
ax.set_ylim(10**-3,10**1.5)
ax.set_yscale("log")
ax.set_xticks([10**-6, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.set_xlabel("Noise Level")
ax.set_ylabel("Relative Coefficient Error",fontsize=14)
ax.grid(True)

plt.tight_layout()
plt.savefig("ParameterEstimation_Lorenz.pdf", bbox_inches="tight")