import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.size"] = 6
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

nobs = np.arange(1, 21)

# control case
ctrl = 0.585
# spread-based method
sprd = 0.523
sprd = (sprd - ctrl) / ctrl * 100.0

mean = np.load("../data/rmse_fig2.npy")
std = np.load("../data/std_fig2.npy")

fig = plt.figure(constrained_layout=True, figsize=(3.5, 2.0), dpi=300)
ax = fig.add_subplot(1, 1, 1)
ax.plot([1], sprd, "o", color="k", ms=2, label="Spread method")
ax.plot(nobs, mean[0], "-", lw=1.0, color="C0", label="UNIF")
ax.fill_between(nobs, mean[0] - std[0], mean[0] +
                std[0], color="C0", lw=0, alpha=0.2)
ax.plot(nobs, mean[1], "-", lw=1.0, color="C1", label="SSPE (D-optimization)")
ax.fill_between(nobs, mean[1] - std[1], mean[1] +
                std[1], color="C1", lw=0, alpha=0.2)
ax.plot(nobs, mean[2], "-", lw=1.0, color="C2", label="SSPE (A-optimization)")
ax.fill_between(nobs, mean[2] - std[2], mean[2] +
                std[2], color="C2", lw=0, alpha=0.2)
ax.plot(nobs, mean[3], "-", lw=1.0, color="C3", label="SSPE (E-optimization)")
ax.fill_between(nobs, mean[3] - std[3], mean[3] +
                std[3], color="C3", lw=0, alpha=0.2)
ax.set_xlim(0, 20)
ax.set_ylim(-30, 0)
ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
ax.set_xticks([0, 5, 10, 15, 20])
ax.set(
    xlabel="Number of observations",
    ylabel="Relative change of RMSE (%)",
)
ax.legend(loc=0, fontsize=6)
plt.savefig("../figure/fig2_obsnumber.png")
plt.show()
