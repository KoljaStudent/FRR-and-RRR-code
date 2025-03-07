import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Computer Modern'

# plt.rcParams['mathtext.fontset'] = 'dejavuserif'
# plt.rcParams['mathtext.fontset'] = 'cm'

ts_data = np.loadtxt("./data/thesis/ts/ts_LIF_white.txt")

time = ts_data[:, 0]
voltage = ts_data[:, 1]
spike_train = ts_data[:, 3]

dt = time[1]

# plot voltage, spike-train with clamped voltage during refractory period
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex = ax1)

t_min = 0
t_max = 10

ax1.plot(time, voltage, "o", markersize = 3, color = (0, 0.290, 0.565))
ax1.set_ylabel(r"$v$", fontsize=35)
ax1.set_xlim(t_min, t_max)
ax1.set_ylim(-0.4, 1.1)
ax1.tick_params(axis = "x", labelbottom=False, bottom = False)
ax1.tick_params(axis = "y", labelleft=False, left = False)

ax2.plot(time, spike_train * dt, "-", color = (0.961, 0.314, 0.114), linewidth = 3)
ax2.set_ylabel(r"$x$", fontsize=35)
ax2.set_xlabel(r"$t$", fontsize=35)
ax2.set_xlim(t_min, t_max)
ax2.set_ylim(0, 1.1)
ax2.tick_params(axis = "x", labelbottom=False, bottom = False)
ax2.tick_params(axis = "y", labelleft=False, left = False)

fig.align_ylabels()

plt.tight_layout()
plt.savefig("./pictures/paper/0_ts.eps") 
