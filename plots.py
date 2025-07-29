import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat("results.mat")
arr_sig_NR = data["arr_sig_NR"]
error_SINDy_manual = data["error_SINDy_manual"]
TPR_SINDy_manual = data["TPR_SINDy_manual"]
error_SINDy_auto = data["error_SINDy_auto"]
TPR_SINDy_auto = data["TPR_SINDy_auto"]
error_bumpWSINDy_manual = data["error_bumpWSINDy_manual"]
TPR_bumpWSINDy_manual = data["TPR_bumpWSINDy_manual"]
error_bumpWSINDy_auto = data["error_bumpWSINDy_auto"]
TPR_bumpWSINDy_auto = data["TPR_bumpWSINDy_auto"]
error_FourierWSINDy_manual = data["error_FourierWSINDy_manual"]
TPR_FourierWSINDy_manual = data["TPR_FourierWSINDy_manual"]
error_FourierWSINDy_auto = data["error_FourierWSINDy_auto"]
TPR_FourierWSINDy_auto = data["TPR_FourierWSINDy_auto"]

plt.figure()
plt.semilogx(arr_sig_NR.flatten(),error_SINDy_auto.flatten())
plt.semilogx(arr_sig_NR.flatten(),error_bumpWSINDy_auto.flatten())
plt.semilogx(arr_sig_NR.flatten(),error_FourierWSINDy_auto.flatten())
plt.show()