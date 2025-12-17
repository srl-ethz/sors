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
tipPos = np.loadtxt("output_arm/tip_displacement.csv", delimiter=",")
tipPos -= tipPos[0]
timesms = np.arange(0, len(tipPos)*dt, dt)

### Start plotting
fig, ax = plt.subplots(figsize=figsize)

line_width = 1
coords = ["x", "y", "z"]
for i in range(3):
    ax.plot(timesms, tipPos[:,i], linestyle="-", linewidth=line_width, label=f"SORS {coords[i]}")
ax.legend(loc="lower center", ncol=4, fancybox=True, fontsize=6, bbox_to_anchor=[0.5, -0.6])

ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (m)")
ax.set_xlim(timesms.min(), timesms.max())
#ax.set_ylim(qs_gt.min()-offset-2e-3, qs_gt.max()-offset+1e-3)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.grid()
ax.set_axisbelow(True)

fig.savefig("output_arm/tip_displacement.png", dpi=300, bbox_inches="tight", pad_inches=1*mm)
# fig.savefig("fig6_sim2realbeam.pdf", bbox_inches="tight", pad_inches=1*mm)
plt.close()

