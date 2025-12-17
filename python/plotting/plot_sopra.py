
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({"font.size": 9})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'})#, "font.serif": ['Computer Modern']})
mm = 1 / 25.4

dataFolder = "output/sopra/plots"

# Sopra
sopraPath = f"{dataFolder}/simPos.csv"
sopraPathGT = f"{dataFolder}/realPos.csv"
sopraData = np.loadtxt(sopraPath)[:, 1:]
sopraDataGT = np.loadtxt(sopraPathGT)[:, 1:]
nSopra = sopraData.shape[0]
sopraDt = 1e-2

# Plotting
fig, ax = plt.subplots(figsize=(0.6*88*mm, 0.6*88*mm))
s = 50
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
for i, (sp, rp) in enumerate(zip(sopraData, sopraDataGT)):
    # ax.scatter(1e2*rp[0], 1e2*rp[1], marker='o', s=s, c=colors[i%6], zorder=2, alpha=0.5)
    # ax.scatter(1e2*sp[0], 1e2*sp[1], marker='x', s=s, c=colors[i%6], zorder=2)
    ax.scatter(1e2*rp[0], 1e2*rp[1], marker='o', s=s, c="tab:orange", zorder=2, alpha=0.75)
    ax.scatter(1e2*sp[0], 1e2*sp[1], marker='x', s=s, c="tab:blue", zorder=2)
# ax.scatter([], [], marker='x', s=s, c='k', label="Simulation", zorder=2)
# ax.scatter([], [], marker='o', s=s, c='k', label="Reality", zorder=2)
ax.scatter([], [], marker='x', s=s, c="tab:blue", label="Simulation", zorder=2)
ax.scatter([], [], marker='o', s=s, c="tab:orange", label="Reality", zorder=2)
ax.set_xlabel("X Position (cm)")
ax.set_ylabel("Y Position (cm)", labelpad=2)
# ax.set_xlim(-0.08, 0.06)
# ax.set_ylim(-0.07, 0.07)
ax.set_xlim(-8, 6)
ax.set_ylim(-7, 7)

ax.legend(loc='lower center', bbox_to_anchor=(0.37, 0.68), ncol=1, handlelength=2)
ax.grid()
# ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))

fig.savefig(f"{dataFolder}/sim2real_sopra.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{dataFolder}/sim2real_sopra.pdf", bbox_inches="tight")
plt.close()