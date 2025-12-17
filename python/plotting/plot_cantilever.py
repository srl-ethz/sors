
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({"font.size": 9})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'})#, "font.serif": ['Computer Modern']})
mm = 1 / 25.4

dataFolder = "output/cantilever/plots"

# Cantilever
cantileverPath = f"{dataFolder}/simMarker.csv"
cantileverPathGT = f"{dataFolder}/realMarker.csv"
cantileverData = np.loadtxt(cantileverPath, delimiter=',')
cantileverDataGT = np.loadtxt(cantileverPathGT, delimiter=',')

# Choose the trajectory to plot
trajIdx = 0
timeIdx = 1
zIdx = [4, 7, 10, 13, 16, 19, 22, 25, 28, 31]
nCantilever = cantileverData.shape[0]
tCantilever = cantileverData.shape[1]
cantileverDt = 1e-2
trajChoice = 16
cantileverData = cantileverData[cantileverData[:, trajIdx] == trajChoice]
cantileverDataGT = cantileverDataGT[cantileverDataGT[:, trajIdx] == trajChoice]

# Plotting
fig, ax = plt.subplots(figsize=(0.6*88*mm, 33*mm))

# Cantilever
ax.plot(cantileverData[:, timeIdx], 1e2*cantileverData[:, zIdx].mean(-1), label='Simulation')
ax.plot(cantileverDataGT[:, timeIdx], 1e2*cantileverDataGT[:, zIdx].mean(-1), label='Reality', linestyle='--')
ax.grid()
ax.set_xlim(0, cantileverData[:, timeIdx].max())
ax.set_xlabel('Time (s)')
ax.set_ylabel('Z Position (cm)', labelpad=5)
# ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
ax.legend(loc='lower center', bbox_to_anchor=(0.67, -0.025), ncol=1, handlelength=1.5, columnspacing=0.5)

fig.savefig(f"{dataFolder}/sim2real_cantilever.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{dataFolder}/sim2real_cantilever.pdf", bbox_inches="tight")
plt.close()
