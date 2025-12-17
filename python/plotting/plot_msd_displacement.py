
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif', "font.serif": ['Computer Modern']})

mm = 1 / 25.4
figsize = (120 * mm, 64 * mm)

### Read CSV
data = np.loadtxt("data/msd_displacement.csv", delimiter=",")
time = data[:, 0]
comPos = data[:, 1]
force = data[:, 2]
#tipPos -= tipPos[0]

### Start plotting
fig, ax1 = plt.subplots(figsize=figsize)

line_width = 1
graph1 = ax1.plot(time, comPos, linestyle="-", linewidth=line_width, label="center of mass")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Position (m)")
ax1.set_xlim(time.min(), time.max())
ax1.grid()
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax2 = ax1.twinx()
graph2 = ax2.plot(time, force, linestyle="--", linewidth=line_width, color = "orange",label="force")
ax2.set_ylabel("Force (N)")
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


#ax.legend(loc="lower center", ncol=4, fancybox=True, fontsize=6, bbox_to_anchor=[0.5, -0.6])
graphs = graph1 + graph2
ax1.legend(graphs, [graph.get_label() for graph in graphs])

#ax.set_axisbelow(True)

fig.savefig("data/msd_tip_displacement_plot.png", dpi=300, bbox_inches="tight", pad_inches=1*mm)
plt.close()

