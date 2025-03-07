import numpy as np
import matplotlib.pyplot as plt

def calc_tada_ratio(Delta, omega):
    return (Delta * tau_a)/(1 + (omega * tau_a) ** 2)

def calc_vFRR(tada_ratio, omega, S_vv_real, S_xv_real, S_xv_imag, S_vf_real, S_vf_imag, S_noise):
    frrv_real = (-S_vf_real + (v_T - v_R + tada_ratio) * S_xv_real + tau_a * omega * tada_ratio * S_xv_imag) / S_noise
    frrv_imag = (omega * S_vv_real + S_vf_imag + (v_T - v_R + tada_ratio) * S_xv_imag - tau_a * omega * tada_ratio * S_xv_real) / S_noise
    return frrv_real, frrv_imag

def calc_sus_x(omega, tada_ratio, sus_v_real, sus_v_imag, sus_f_real, sus_f_imag):
    tada_ratio2 = 1 / ( (v_T - v_R + tada_ratio) ** 2 + (omega * tau_a * tada_ratio) ** 2)

    # calculate chi_x using RRR
    rrr_real = tada_ratio2 * ((v_T - v_R + tada_ratio) * (sus_f_real + omega * sus_v_imag + 1) + omega * tau_a * tada_ratio * (omega * sus_v_real - sus_f_imag))
    rrr_imag = tada_ratio2 * ((v_T - v_R + tada_ratio) * (omega * sus_v_real - sus_f_imag) - omega * tau_a * tada_ratio * (sus_f_real + omega * sus_v_imag + 1))
    return rrr_real, rrr_imag

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Computer Modern'

# plt.rcParams['mathtext.fontset'] = 'dejavuserif'
# plt.rcParams['mathtext.fontset'] = 'cm'


# load data 
spec_data_a = np.loadtxt("./data/thesis/EIF_adapt_OU/res_EIF_adapt_OU_53.txt")
sus_data_a = np.loadtxt("./data/thesis/EIF_adapt_OU/res_s_EIF_adapt_OU_53.txt")

# spec_data_b = np.loadtxt("./data/thesis/EIF_adapt_OU/res_EIF_adapt_OU_54.txt")
# sus_data_b = np.loadtxt("./data/thesis/EIF_adapt_OU/res_s_EIF_adapt_OU_54.txt")

spec_data_c = np.loadtxt("./data/thesis/EIF_adapt_OU/res_EIF_adapt_OU_55.txt")
sus_data_c = np.loadtxt("./data/thesis/EIF_adapt_OU/res_s_EIF_adapt_OU_55.txt")

# physical parameters for all cases
v_R = 0 # reset voltage
v_T = 1.5 # threshold voltage
tau_a = 100 # adaption time constant

""" 
    3) EIF + colored noise + adaptation voltage FRR + 4) EIF + colored noise + adaptation RRR
"""

""" 
    b) Delta_a = 1
"""

frequency = spec_data_a[:,0]

S_xv_real = spec_data_a[:,3]
S_xv_imag = spec_data_a[:,4]
S_vv_real = spec_data_a[:,5]
S_vv_imag = spec_data_a[:,6]
S_vf_real = spec_data_a[:,9]
S_vf_imag = spec_data_a[:,10]
S_noise_real = spec_data_a[:,13]

sus_x_real = sus_data_a[:,7]
sus_x_imag = sus_data_a[:,8]
sus_v_real = sus_data_a[:,9]
sus_v_imag = sus_data_a[:,10]
sus_f_real = sus_data_a[:,11]
sus_f_imag = sus_data_a[:,12]

omega = 2 * np.pi * frequency

# simulation parameters
df = frequency[1]
N_frq = frequency.shape[0]
steps = 2 * (N_frq - 1) # number of simulation steps
T = 1 / df # simulation time
dt = T / steps 
f_nyquist_white = 0.5 / dt # maximal measurable frequency

Delta_a = 1 # adaption jump size

# calculate X_v based on vFRR
tada_ratio = calc_tada_ratio(Delta_a, omega)

# calculate voltage FRR with OU noise
frrv_real, frrv_imag = calc_vFRR(tada_ratio, omega, S_vv_real, S_xv_real, S_xv_imag, S_vf_real, S_vf_imag, S_noise_real)

# calculate RRR with OU noise
rrr_real, rrr_imag = calc_sus_x(omega, tada_ratio, sus_v_real, sus_v_imag, sus_f_real, sus_f_imag)

# response of the Ornstein-Uhlenbeck process as reference
sus_OUP_real = 1 / (1 + omega ** 2)
sus_OUP_imag = omega / (1 + omega ** 2)

# plot voltage FRR for OU noise
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

f_min = df
f_max = 9.5

ax1.plot(frequency, sus_v_real, "o", markersize = 6, color = (0, 0.290, 0.565))
ax1.plot(frequency, frrv_real, "o", markersize = 6, color = (0.961, 0.314, 0.114))
ax1.plot(frequency, sus_OUP_real, "-k", linewidth = 3)
ax1.set_ylabel(r"$\mathrm{Re}(\chi_v)$", fontsize=35)
ax1.set_xscale("log")
ax1.set_xlim(f_min, f_max)
ax1.set_ylim(-0.1, 1.5)
ax1.tick_params(axis = "y", labelsize = 24)
ax1.tick_params(axis = "x", labelbottom = False)
ax1.text(1.75e-4, 1.35, r"$\mathbf{A}$", fontsize = 35)
ax1.legend(["susceptibility", "voltage FRR", "OUP susceptibility"], fontsize=24)

ax2.plot(frequency, -sus_v_imag, "o", markersize = 6, color = (0, 0.290, 0.565))
ax2.plot(frequency, frrv_imag, "o", markersize = 6, color = (0.961, 0.314, 0.114))
ax2.plot(frequency, sus_OUP_imag, "-k", linewidth = 3)
ax2.set_xlabel(r"$f \tau _m$", fontsize=35)
ax2.set_ylabel(r"$\mathrm{Im}(\chi_v)$", fontsize=35)
ax2.set_xscale("log")
ax2.set_xlim(f_min, f_max)
ax2.set_ylim(-0.7, 0.7)
ax2.tick_params(axis = "both", labelsize = 24)

fig.align_ylabels()

plt.tight_layout()
plt.savefig("./pictures/paper/4_a_vFRR_EIF_adapt_OU.pdf") 


# plot RRR for OU noise
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

f_min = df
f_max = 9.5

ax1.plot(frequency, sus_x_real, "o", markersize = 6, color = (0, 0.290, 0.565))
ax1.plot(frequency, rrr_real, "o", markersize = 6, color = (0.961, 0.314, 0.114))
ax1.set_ylabel(r"$\mathrm{Re}(\chi_x)$", fontsize=35)
ax1.set_xscale("log")
ax1.set_xlim(f_min, f_max)
ax1.set_ylim(-0.05, 0.1)
ax1.tick_params(axis = "y", labelsize = 24)
ax1.tick_params(axis = "x", labelbottom = False)
ax1.text(2e-4, 0.1, r"$\mathbf{A}$", fontsize = 35)

ax2.plot(frequency, -sus_x_imag, "o", markersize = 6, color = (0, 0.290, 0.565))
ax2.plot(frequency, rrr_imag, "o", markersize = 6, color = (0.961, 0.314, 0.114))
ax2.set_xlabel(r"$f \tau _m$", fontsize=35)
ax2.set_ylabel(r"$\mathrm{Im}(\chi_x)$", fontsize=35)
ax2.set_xscale("log")
ax2.set_xlim(f_min, f_max)
ax2.set_ylim(-0.05, 0.1)
ax2.tick_params(axis = "both", labelsize = 24)

fig.align_ylabels()

plt.tight_layout()
plt.savefig("./pictures/paper/5_a_RRR_EIF_adapt_OU.pdf") 


"""
    b) Delta_a = 0.025
"""

# frequency = spec_data_1_b[:,0]

# S_xv_real = spec_data_1_b[:,3]
# S_xv_imag = spec_data_1_b[:,4]
# S_vv_real = spec_data_1_b[:,5]
# S_vv_imag = spec_data_1_b[:,6]
# S_vf_real = spec_data_1_b[:,9]
# S_vf_imag = spec_data_1_b[:,10]
# S_noise_real = spec_data_1_b[:,13]

# sus_v_real = sus_data_1_b[:,9]
# sus_v_imag = sus_data_1_b[:,10]

# omega = 2 * np.pi * frequency

# # simulation parameters
# df = frequency[1]
# N_frq = frequency.shape[0]
# steps = 2 * (N_frq - 1) # number of simulation steps
# T = 1 / df # simulation time
# dt = T / steps 
# f_nyquist_white = 0.5 / dt # maximal measurable frequency

# Delta_a = 0.025 # adaption jump size

# # calculate X_v based on vFRR
# tada_ratio = calc_tada_ratio(Delta_a, omega)

# #calculate voltage FRR with OU noise
# frrv_real, frrv_imag = calc_vFRR(tada_ratio, omega, S_vv_real, S_xv_real, S_xv_imag, S_vf_real, S_vf_imag, S_noise_real)

# # plot voltage FRR for white noise
# fig = plt.figure(figsize=(14, 6))

# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)

# f_min = df
# f_max = 9.5

# ax1.plot(frequency, sus_v_real, "o", markersize = 6, color = (0, 0.290, 0.565))
# ax1.plot(frequency, frrv_real, "o", markersize = 6, color = (0.961, 0.314, 0.114))
# ax1.set_ylabel(r"$\mathrm{Re}(\chi_v)$", fontsize=35)
# ax1.set_xscale("log")
# ax1.set_xlim(f_min, f_max)
# ax1.set_ylim(-0.1, 1.5)
# ax1.tick_params(axis = "y", labelsize = 24)
# ax1.tick_params(axis = "x", labelbottom = False)
# ax1.text(1.75e-4, 1.35, r"$\mathbf{B}$", fontsize = 35)

# ax2.plot(frequency, -sus_v_imag, "o", markersize = 6, color = (0, 0.290, 0.565))
# ax2.plot(frequency, frrv_imag, "o", markersize = 6, color = (0.961, 0.314, 0.114))
# ax2.set_xlabel(r"$f \tau _m$", fontsize=35)
# ax2.set_ylabel(r"$\mathrm{Im}(\chi_v)$", fontsize=35)
# ax2.set_xscale("log")
# ax2.set_xlim(f_min, f_max)
# ax2.set_ylim(-0.7, 0.7)
# ax2.tick_params(axis = "both", labelsize = 24)

# fig.align_ylabels()

# plt.tight_layout()
# plt.savefig("./pictures/paper/3_b_vFRR_EIF_adapt_OU.png") 

"""
    a) Delta_a = 0.005
"""

frequency = spec_data_c[:,0]

S_xv_real = spec_data_c[:,3]
S_xv_imag = spec_data_c[:,4]
S_vv_real = spec_data_c[:,5]
S_vv_imag = spec_data_c[:,6]
S_vf_real = spec_data_c[:,9]
S_vf_imag = spec_data_c[:,10]
S_noise_real = spec_data_c[:,13]

sus_x_real = sus_data_c[:,7]
sus_x_imag = sus_data_c[:,8]
sus_v_real = sus_data_c[:,9]
sus_v_imag = sus_data_c[:,10]
sus_f_real = sus_data_c[:,11]
sus_f_imag = sus_data_c[:,12]

omega = 2 * np.pi * frequency

# simulation parameters
df = frequency[1]
N_frq = frequency.shape[0]
steps = 2 * (N_frq - 1) # number of simulation steps
T = 1 / df # simulation time
dt = T / steps 
f_nyquist_white = 0.5 / dt # maximal measurable frequency

Delta_a = 0.005 # adaption jump size

# calculate X_v based on vFRR
tada_ratio = calc_tada_ratio(Delta_a, omega)

#calculate voltage FRR with OU noise
frrv_real, frrv_imag = calc_vFRR(tada_ratio, omega, S_vv_real, S_xv_real, S_xv_imag, S_vf_real, S_vf_imag, S_noise_real)

# calculate RRR with OU noise
rrr_real, rrr_imag = calc_sus_x(omega, tada_ratio, sus_v_real, sus_v_imag, sus_f_real, sus_f_imag)

# plot voltage FRR for white noise
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

f_min = df
f_max = 9.5

ax1.plot(frequency, sus_v_real, "o", markersize = 6, color = (0, 0.290, 0.565))
ax1.plot(frequency, frrv_real, "o", markersize = 6, color = (0.961, 0.314, 0.114))
ax1.plot(frequency, sus_OUP_real, "-k", linewidth = 3)
ax1.set_ylabel(r"$\mathrm{Re}(\chi_v)$", fontsize=35)
ax1.set_xscale("log")
ax1.set_xlim(f_min, f_max)
ax1.set_ylim(-0.1, 1.5)
ax1.tick_params(axis = "y", labelsize = 24)
ax1.tick_params(axis = "x", labelbottom = False)
ax1.text(1.75e-4, 1.35, r"$\mathbf{B}$", fontsize = 35)

ax2.plot(frequency, -sus_v_imag, "o", markersize = 6, color = (0, 0.290, 0.565))
ax2.plot(frequency, frrv_imag, "o", markersize = 6, color = (0.961, 0.314, 0.114))
ax2.plot(frequency, sus_OUP_imag, "-k", linewidth = 3)
ax2.set_xlabel(r"$f \tau _m$", fontsize=35)
ax2.set_ylabel(r"$\mathrm{Im}(\chi_v)$", fontsize=35)
ax2.set_xscale("log")
ax2.set_xlim(f_min, f_max)
ax2.set_ylim(-0.7, 0.7)
ax2.tick_params(axis = "both", labelsize = 24)

fig.align_ylabels()

plt.tight_layout()
plt.savefig("./pictures/paper/4_b_vFRR_EIF_adapt_OU.pdf") 

# plot RRR for OU noise
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

f_min = df
f_max = 9.5

ax1.plot(frequency, sus_x_real, "o", markersize = 6, color = (0, 0.290, 0.565))
ax1.plot(frequency, rrr_real, "o", markersize = 6, color = (0.961, 0.314, 0.114))
ax1.set_ylabel(r"$\mathrm{Re}(\chi_x)$", fontsize=35)
ax1.set_xscale("log")
ax1.set_xlim(f_min, f_max)
ax1.set_ylim(-0.1, 0.6)
ax1.tick_params(axis = "y", labelsize = 24)
ax1.tick_params(axis = "x", labelbottom = False)
ax1.text(2e-4, 0.54, r"$\mathbf{B}$", fontsize = 35)
ax1.legend(["susceptibility", "RRR"], fontsize=24)

ax2.plot(frequency, -sus_x_imag, "o", markersize = 6, color = (0, 0.290, 0.565))
ax2.plot(frequency, rrr_imag, "o", markersize = 6, color = (0.961, 0.314, 0.114))
ax2.set_xlabel(r"$f \tau _m$", fontsize=35)
ax2.set_ylabel(r"$\mathrm{Im}(\chi_x)$", fontsize=35)
ax2.set_xscale("log")
ax2.set_xlim(f_min, f_max)
ax2.set_ylim(-0.2, 0.4)
ax2.tick_params(axis = "both", labelsize = 24)

fig.align_ylabels()

plt.tight_layout()
plt.savefig("./pictures/paper/5_b_RRR_EIF_adapt_OU.pdf") 