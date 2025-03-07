import numpy as np
import matplotlib.pyplot as plt
from scipy import special

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Computer Modern'

# plt.rcParams['mathtext.fontset'] = 'dejavuserif'
# plt.rcParams['mathtext.fontset'] = 'cm'


# load data 
spec_data_1_a = np.loadtxt("./data/thesis/LIF_white/res_LIF_white_4.txt")
sus_data_1_a = np.loadtxt("./data/thesis/LIF_white/res_s_LIF_white_4.txt")
sus_data_1_a_alt = np.loadtxt("./data/thesis/LIF_white/res_s_LIF_white_9.txt")

spec_data_1_b = np.loadtxt("./data/thesis/LIF_white/res_LIF_white_6.txt")
sus_data_1_b = np.loadtxt("./data/thesis/LIF_white/res_s_LIF_white_6.txt")
sus_data_1_b_alt = np.loadtxt("./data/thesis/LIF_white/res_s_LIF_white_10.txt")


mathematica_data_1_a = np.genfromtxt("./data/parabolicCylinder/parabolicCylinder08.csv", delimiter = ",")
mathematica_data_1_b = np.genfromtxt("./data/parabolicCylinder/parabolicCylinder12.csv", delimiter = ",")

""" 
    1) LIF + white noise plots for vFRR in two cases
"""

""" 
    a) mu = 0.8
"""

frequency_spon = spec_data_1_a[:,0]
frequency_08 = spec_data_1_a[:,0]
S_xv_real = spec_data_1_a[:,3]
S_xv_imag = spec_data_1_a[:,4]
S_vv_real = spec_data_1_a[:,5]
S_vv_08 = spec_data_1_a[:,5]
S_vv_imag = spec_data_1_a[:,6]
frequency_stim = sus_data_1_a[:,0]
sus_v_real = sus_data_1_a[:,9]
sus_v_imag = sus_data_1_a[:,10]

sus_v_real_alt = sus_data_1_a_alt[:,9]
sus_v_imag_alt = sus_data_1_a_alt[:,10]

Re_D1_08 = mathematica_data_1_a[:,1]
Im_D1_08 = mathematica_data_1_a[:,2]
Re_D2_08 = mathematica_data_1_a[:,3]
Im_D2_08 = mathematica_data_1_a[:,4]
Re_D3_08 = mathematica_data_1_a[:,5]
Im_D3_08 = mathematica_data_1_a[:,6]
Re_D4_08 = mathematica_data_1_a[:,7]
Im_D4_08 = mathematica_data_1_a[:,8]

omega_spon = 2 * np.pi * frequency_spon
omega_stim = 2 * np.pi * frequency_stim

# simulation parameters
df = frequency_spon[1]
N_frq = frequency_spon.shape[0]
steps = 2 * (N_frq - 1) # number of simulation steps
T = 1 / df # simulation time
dt = T / steps 
f_nyquist_white = 0.5 / dt # maximal measurable frequency

# physical parameters
D = 0.1 # noise intensity
v_R = 0 # reset voltage
v_T = 1 # threshold voltage
mu = 0.8 # mean input
r0 = 0.369689 # firing rate from count statistics 
N_frq_max = Re_D1_08.shape[0] # maximal stimulus frequency

# calculate voltage FRR for LIF with white noise 
frrv_real = (S_vv_real + (v_T - v_R) * S_xv_real) / (2 * D)
frrv_imag = (omega_spon * S_vv_real + (v_T - v_R) * S_xv_imag) / (2 * D)

# calculate expected statinary firing rate for white noise
z_min = (mu - v_T) / (np.sqrt(2 * D))
z_max = (mu - v_R) / (np.sqrt(2 * D))

num_sup = 100000

dz = (z_max - z_min) / num_sup

integral = 0.

for i in range(0, num_sup):
    integral +=  np.exp((z_min + i * dz) ** 2) * special.erfc(z_min + i * dz)
integral = integral * np.sqrt(np.pi) * dz

r_0 = 1 / integral

# 0) set up abreviations 
z_T = (mu - v_T) / np.sqrt(D)
z_R = (mu - v_R) / np.sqrt(D)

e2 = np.exp((z_R ** 2 - z_T ** 2) / 2)
e4 = np.exp((z_R ** 2 - z_T ** 2) / 4)

R1 = Re_D1_08 - e4 * Re_D3_08
R2 = Re_D2_08 - e4 * Re_D4_08
I1 = Im_D1_08 - e4 * Im_D3_08
I2 = Im_D2_08 - e4 * Im_D4_08

A = R2 ** 2 + I2 ** 2

# 1) calculate S_xx(f) 
S_xx_theory = r_0 * (Re_D2_08 ** 2 + Im_D2_08 ** 2 - e2 * (Re_D4_08 ** 2 + Im_D4_08 ** 2)) / A

# 2) calculate Re(X) and Im(X)
sus_x_real_theory = r_0 * omega_spon[:N_frq_max] * (omega_spon[:N_frq_max] * (R1 * R2 + I1 * I2) + I1 * R2 - R1 * I2) / (np.sqrt(D) * (1 + (omega_spon[:N_frq_max]) ** 2) * A)
sus_x_imag_theory = r_0 * omega_spon[:N_frq_max] * (omega_spon[:N_frq_max] * (I1 * R2 - R1 * I2) - R1 * R2 - I1 * I2) / (np.sqrt(D) * (1 + (omega_spon[:N_frq_max]) ** 2) * A)

# 3) calculate Re(S_vx) and Im(S_vx)
S_xv_real_theory = (2 * D *(sus_x_real_theory + omega_spon[:N_frq_max] * sus_x_imag_theory) - (v_T - v_R) * S_xx_theory) / (1 + (omega_spon[:N_frq_max]) ** 2)
S_xv_imag_theory = (2 * D * sus_x_imag_theory - omega_spon[:N_frq_max] * (2 * D * sus_x_real_theory - (v_T - v_R) * S_xx_theory)) / (1 + (omega_spon[:N_frq_max]) ** 2)

# 4) calculate analytical result for X_v based on RRR
sus_v_real_theory = (1 - (v_T-v_R) * (sus_x_real_theory - omega_spon[:N_frq_max] * sus_x_imag_theory)) / (1 + (omega_spon[:N_frq_max]) ** 2)
sus_v_imag_theory = (omega_spon[:N_frq_max] - (v_T - v_R) * (sus_x_imag_theory + omega_spon[:N_frq_max] * sus_x_real_theory)) / (1 + (omega_spon[:N_frq_max]) ** 2)

# plot voltage FRR for white noise
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


f_min = df
f_max = 60

ax1.plot(frequency_stim, sus_v_real, "o", markersize = 6, color = (0, 0.290, 0.565))
# ax1.plot(frequency_stim, sus_v_real_alt, "o", markersize = 6, color = (0.290,0.565,0))
ax1.plot(frequency_spon, frrv_real, "o", markersize = 6, color = (0.961, 0.314, 0.114))
ax1.plot(frequency_spon[:N_frq_max], sus_v_real_theory, "-k", linewidth = 2)
ax1.set_ylabel(r"$\mathrm{Re}(\chi_v)$", fontsize=35)
ax1.set_xscale("log")
ax1.set_xlim(f_min, f_max)
ax1.set_ylim(-0.02, 0.2)
ax1.tick_params(axis = "y", labelsize = 24)
ax1.tick_params(axis = "x", labelbottom = False)
ax1.text(4.2e-3, 0.18, r"$\mathbf{A}$", fontsize = 35)
ax1.legend(["susceptibility","fluctuations (FRR)", "theory (RRR)"], fontsize=24)
# ax1.legend([r"$\varepsilon = 1$", r"$\varepsilon = 0.1$","fluctuations (FRR)", "theory (RRR)"], fontsize=24)

ax2.plot(frequency_stim, -sus_v_imag, "o", markersize = 6, color = (0, 0.290, 0.565))
# ax2.plot(frequency_stim, -sus_v_imag_alt, "o", markersize = 6, color = (0.290,0.565,0))
ax2.plot(frequency_spon, frrv_imag, "o", markersize = 6, color = (0.961, 0.314, 0.114))
ax2.plot(frequency_spon[:N_frq_max], sus_v_imag_theory, "-k", linewidth = 2)
ax2.set_xlabel(r"$f \tau _m$", fontsize=35)
ax2.set_ylabel(r"$\mathrm{Im}(\chi_v)$", fontsize=35)
ax2.set_xscale("log")
ax2.set_xlim(f_min, f_max)
ax2.set_ylim(-0.03, 0.12)
ax2.tick_params(axis = "both", labelsize = 24)

fig.align_ylabels()

plt.tight_layout()
plt.savefig("./pictures/paper/1_a_vFRR_LIF_white.pdf") 


"""
    b) mu = 1.2
"""

frequency_spon = spec_data_1_b[:,0]
frequency_12 = spec_data_1_b[:,0]
S_xv_real = spec_data_1_b[:,3]
S_xv_imag = spec_data_1_b[:,4]
S_vv_real = spec_data_1_b[:,5]
S_vv_12 = spec_data_1_b[:,5]
S_vv_imag = spec_data_1_b[:,6]
frequency_stim = sus_data_1_b[:,0]
sus_v_real = sus_data_1_b[:,9]
sus_v_imag = sus_data_1_b[:,10]

sus_v_real_alt = sus_data_1_b_alt[:,9]
sus_v_imag_alt = sus_data_1_b_alt[:,10]

Re_D1 = mathematica_data_1_b[:,1]
Im_D1 = mathematica_data_1_b[:,2]
Re_D2 = mathematica_data_1_b[:,3]
Im_D2 = mathematica_data_1_b[:,4]
Re_D3 = mathematica_data_1_b[:,5]
Im_D3 = mathematica_data_1_b[:,6]
Re_D4 = mathematica_data_1_b[:,7]
Im_D4 = mathematica_data_1_b[:,8]

omega_spon = 2 * np.pi * frequency_spon
omega_stim = 2 * np.pi * frequency_stim

# simulation parameters
df = frequency_spon[1]
N_frq = frequency_spon.shape[0]
steps = 2 * (N_frq - 1) # number of simulation steps
T = 1 / df # simulation time
dt = T / steps 
f_nyquist_white = 0.5 / dt # maximal measurable frequency

# physical parameters
D = 0.1 # noise intensity
v_R = 0 # reset voltage
v_T = 1 # threshold voltage
mu = 1.2 # mean input
r0 = 0.73 # firing rate from count statistics 
N_frq_max = Re_D1.shape[0] # maximal stimulus frequency

# calculate voltage FRR for LIF with white noise 
frrv_real = (S_vv_real + (v_T - v_R) * S_xv_real) / (2 * D)
frrv_imag = (omega_spon * S_vv_real + (v_T - v_R) * S_xv_imag) / (2 * D)

# calculate expected statinary firing rate for white noise
z_min = (mu - v_T) / (np.sqrt(2 * D))
z_max = (mu - v_R) / (np.sqrt(2 * D))

num_sup = 100000

dz = (z_max - z_min) / num_sup

integral = 0.

for i in range(0, num_sup):
    integral +=  np.exp((z_min + i * dz) ** 2) * special.erfc(z_min + i * dz)
integral = integral * np.sqrt(np.pi) * dz

r_0 = 1 / integral

# 0) set up abreviations 
z_T = (mu - v_T) / np.sqrt(D)
z_R = (mu - v_R) / np.sqrt(D)

e2 = np.exp((z_R ** 2 - z_T ** 2) / 2)
e4 = np.exp((z_R ** 2 - z_T ** 2) / 4)

R1 = Re_D1 - e4 * Re_D3
R2 = Re_D2 - e4 * Re_D4
I1 = Im_D1 - e4 * Im_D3
I2 = Im_D2 - e4 * Im_D4

A = R2 ** 2 + I2 ** 2

# 1) calculate S_xx(f) 
S_xx_theory = r_0 * (Re_D2 ** 2 + Im_D2 ** 2 - e2 * (Re_D4 ** 2 + Im_D4 ** 2)) / A

# 2) calculate Re(X) and Im(X)
sus_x_real_theory = r_0 * omega_spon[:N_frq_max] * (omega_spon[:N_frq_max] * (R1 * R2 + I1 * I2) + I1 * R2 - R1 * I2) / (np.sqrt(D) * (1 + (omega_spon[:N_frq_max]) ** 2) * A)
sus_x_imag_theory = r_0 * omega_spon[:N_frq_max] * (omega_spon[:N_frq_max] * (I1 * R2 - R1 * I2) - R1 * R2 - I1 * I2) / (np.sqrt(D) * (1 + (omega_spon[:N_frq_max]) ** 2) * A)

# 3) calculate Re(S_vx) and Im(S_vx)
S_xv_real_theory = (2 * D *(sus_x_real_theory + omega_spon[:N_frq_max] * sus_x_imag_theory) - (v_T - v_R) * S_xx_theory) / (1 + (omega_spon[:N_frq_max]) ** 2)
S_xv_imag_theory = (2 * D * sus_x_imag_theory - omega_spon[:N_frq_max] * (2 * D * sus_x_real_theory - (v_T - v_R) * S_xx_theory)) / (1 + (omega_spon[:N_frq_max]) ** 2)

# 4) calculate analytical result for X_v based on RRR
sus_v_real_theory = (1 - (v_T-v_R) * (sus_x_real_theory - omega_spon[:N_frq_max] * sus_x_imag_theory)) / (1 + (omega_spon[:N_frq_max]) ** 2)
sus_v_imag_theory = (omega_spon[:N_frq_max] - (v_T - v_R) * (sus_x_imag_theory + omega_spon[:N_frq_max] * sus_x_real_theory)) / (1 + (omega_spon[:N_frq_max]) ** 2)

# plot voltage FRR for white noise
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

f_min = df
f_max = 60

ax1.plot(frequency_stim, sus_v_real, "o", markersize = 6, color = (0, 0.290, 0.565))
# ax1.plot(frequency_stim, sus_v_real_alt, "o", markersize = 6, color = (0.290,0.565,0))
ax1.plot(frequency_spon, frrv_real, "o", markersize = 6, color = (0.961, 0.314, 0.114))
ax1.plot(frequency_spon[:N_frq_max], sus_v_real_theory, "-k", linewidth = 2)
ax1.set_ylabel(r"$\mathrm{Re}(\chi_v)$", fontsize=35)
ax1.set_xscale("log")
ax1.set_xlim(f_min, f_max)
ax1.set_ylim(-0.02, 0.2)
ax1.tick_params(axis = "y", labelsize = 24)
ax1.tick_params(axis = "x", labelbottom = False)
ax1.text(4.2e-3, 0.18, r"$\mathbf{B}$", fontsize = 35)

ax2.plot(frequency_stim, -sus_v_imag, "o", markersize = 6, color = (0, 0.290, 0.565))
# ax2.plot(frequency_stim, -sus_v_imag_alt, "o", markersize = 6, color = (0.290,0.565,0))
ax2.plot(frequency_spon, frrv_imag, "o", markersize = 6, color = (0.961, 0.314, 0.114))
ax2.plot(frequency_spon[:N_frq_max], sus_v_imag_theory, "-k", linewidth = 2)
ax2.set_xlabel(r"$f \tau _m$", fontsize=35)
ax2.set_ylabel(r"$\mathrm{Im}(\chi_v)$", fontsize=35)
ax2.set_xscale("log")
ax2.set_xlim(f_min, f_max)
ax2.set_ylim(-0.03, 0.12)
ax2.tick_params(axis = "both", labelsize = 24)

fig.align_ylabels()

plt.tight_layout()
plt.savefig("./pictures/paper/1_b_vFRR_LIF_white.pdf") 