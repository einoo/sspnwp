import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.size"] = 8
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

radius = np.array([300, 500, 600, 700, 900, 1200,
                  2000, 3000, 4000, 5000, 6000])
mean = np.load("../data/rmse_fig4a.npy")
std = np.load("../data/std_fig4a.npy")

fig = plt.figure(constrained_layout=True, figsize=(6.5, 2.0), dpi=300)
ax = fig.add_subplot(1, 2, 1)
ax.plot(radius, mean[0], "-", lw=1.0, color="C0", label="UNIF")
ax.fill_between(radius, mean[0] - std[0], mean[0] +
                std[0], color="C0", lw=0, alpha=0.2)
ax.plot(radius, mean[1], "-", lw=1.0, color="C1",
        label="SSPE (D-optimization)")
ax.fill_between(radius, mean[1] - std[1], mean[1] +
                std[1], color="C1", lw=0, alpha=0.2)
ax.plot(radius, mean[2], "-", lw=1.0, color="C2",
        label="SSPE (A-optimization)")
ax.fill_between(radius, mean[2] - std[2], mean[2] +
                std[2], color="C2", lw=0, alpha=0.2)
ax.plot(radius, mean[3], "-", lw=1.0, color="C3",
        label="SSPE (E-optimization)")
ax.fill_between(radius, mean[3] - std[3], mean[3] +
                std[3], color="C3", lw=0, alpha=0.2)
ax.set_xlim(0, 6000)
ax.set_ylim(-30, 0)
ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
ax.set_xticks([1000, 2000, 3000, 4000, 5000, 6000])
ax.set(
    xlabel=r"Localization factor $\sigma$ (km)",
    ylabel="Relative change of RMSE (%)",
)
ax.legend(loc=0, fontsize=6)

########################################################
# NOBS = 20
mean = np.load("../data/rmse_fig4b.npy")
std = np.load("../data/std_fig4b.npy")
# second figure
ax = fig.add_subplot(1, 2, 2)
ax.plot(radius, mean[0], "-", lw=1.0, color="C0", label="UNIF")
ax.fill_between(radius, mean[0] - std[0], mean[0] +
                std[0], color="C0", lw=0, alpha=0.2)
ax.plot(radius, mean[1], "-", lw=1.0, color="C1",
        label="SSPE (D-optimization)")
ax.fill_between(radius, mean[1] - std[1], mean[1] +
                std[1], color="C1", lw=0, alpha=0.2)
ax.plot(radius, mean[2], "-", lw=1.0, color="C2",
        label="SSPE (A-optimization)")
ax.fill_between(radius, mean[2] - std[2], mean[2] +
                std[2], color="C2", lw=0, alpha=0.2)
ax.plot(radius, mean[3], "-", lw=1.0, color="C3",
        label="SSPE (E-optimization)")
ax.fill_between(radius, mean[3] - std[3], mean[3] +
                std[3], color="C3", lw=0, alpha=0.2)
ax.set_xlim(0, 6000)
ax.set_ylim(-50, 0)
ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
ax.set_xticks([1000, 2000, 3000, 4000, 5000, 6000])
ax.set(
    xlabel=r"Localization factor $\sigma$ (km)",
    ylabel="Relative change of RMSE (%)",
)
ax.legend(loc=0, fontsize=6)
plt.savefig("../figure/fig4_locfactor.png")
plt.show()
