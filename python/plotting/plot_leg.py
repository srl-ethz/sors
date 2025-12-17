import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'})#, "font.serif": ['Computer Modern']})
mm = 1 / 25.4

dataFolderUnactuated = "output/hopping_leg_unactuated/plots"
dataFolder = "output/hopping_leg/plots"

# Leg Data
actuationPath = f"{dataFolder}/actuationSignal.csv"
optLegPath = f"{dataFolder}/minPos.csv"
ghostLegPath = f"{dataFolderUnactuated}/minPos.csv"
actuationData = np.loadtxt(actuationPath, delimiter=',')
ghostLegData = np.loadtxt(ghostLegPath, delimiter=',')
optLegData = np.loadtxt(optLegPath, delimiter=',')

fig, axs = plt.subplots(1, 2, figsize=(88*mm, 32*mm))
fig.subplots_adjust(wspace=0.55)
axs[0].plot(actuationData[:len(ghostLegData),0], actuationData[:len(ghostLegData),1], label='Dorsal')
axs[0].plot(actuationData[:len(ghostLegData),0], actuationData[:len(ghostLegData),2], label='Ventral')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Actuation (-)')
axs[0].grid()
axs[0].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
axs[0].legend(loc='lower center', bbox_to_anchor=(0.7, 0.4), ncol=1, handlelength=0.02, columnspacing=0.5)

axs[1].plot(ghostLegData[:, 0], ghostLegData[:, 3], label="Passive")
axs[1].plot(optLegData[:, 0], optLegData[:, 3], label="Optimized")
axs[1].set_xlabel('Time (s)')
axs[1].set_xlim([0, ghostLegData[-1,0]])
axs[1].set_ylabel('Z Position (m)')
axs[1].grid()
axs[1].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
axs[1].legend(loc='lower center', bbox_to_anchor=(0.615, 0.52), ncol=1, handlelength=0.05, columnspacing=0.5)

fig.savefig(f"{dataFolder}/hopping_leg.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{dataFolder}/hopping_leg.pdf", bbox_inches='tight')
plt.close(fig)