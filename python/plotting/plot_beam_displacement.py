
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif', "font.serif": ['Computer Modern']})

mm = 1 / 25.4
figsize = (60 * mm, 32 * mm)

### Read CSV
dt = 1e-2
data = np.loadtxt("output/tip_displacement.csv", delimiter=",")
times = data[:,1]
tipPos = data[:,-1]
tipPos -= tipPos[0]

# DiffPD comparison
realdt = 1e-2
# tipPosDiffPD = np.loadtxt("vertical_movement.txt")
tipPosDiffPD = np.array([0])
tipPosDiffPD -= tipPosDiffPD[0]
# times = np.arange(0, len(tipPosDiffPD)*realdt, realdt)

# Load real data
# realPosData = np.load("qs_real.npy")
# realPosData = realPosData[:,1,2]
realPosData = np.array([0])
realPos = realPosData - realPosData[0]
realPos = realPos[:len(tipPosDiffPD)]

# Load COMSOL result
q_comsol=np.array([
    [1.0147, 15.001, 23.226],
    [0.51624, -0.0009737,    17.248],
    [-0.96361, 30.000, -0.69132]])

q_comsol=q_comsol*0.001

### Start plotting
fig, ax = plt.subplots(figsize=figsize)

line_width = 1
ax.plot(times, tipPos, linestyle="-", linewidth=line_width, label="SORS")
# ax.plot(times, tipPosDiffPD, linestyle="-", linewidth=line_width, label="DiffPD")
# ax.plot(times, realPos, linestyle="--", linewidth=line_width, label="Real")
# ax.scatter(times[-5], q_comsol[1,2]-realPosData[0], s=10, c='k', marker='x', label='COMSOL')
ax.legend(loc="lower center", ncol=4, fancybox=True, fontsize=6, bbox_to_anchor=[0.5, -0.6])

ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (m)")
ax.set_xlim(times.min(), times.max())
#ax.set_ylim(qs_gt.min()-offset-2e-3, qs_gt.max()-offset+1e-3)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.grid()
ax.set_axisbelow(True)

fig.savefig("tip_displacement.png", dpi=300, bbox_inches="tight", pad_inches=1*mm)
# fig.savefig("fig6_sim2realbeam.pdf", bbox_inches="tight", pad_inches=1*mm)
plt.close()

