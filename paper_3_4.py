import numpy as np
import matplotlib.pyplot as plt
from scipy import special

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Computer Modern'

def calc_sus_theory(mu, Re_D1, Im_D1, Re_D2, Im_D2, Re_D3, Im_D3, Re_D4, Im_D4, omega):
    z_min = (mu - v_T) / (np.sqrt(2 * D))
    z_max = (mu - v_R) / (np.sqrt(2 * D))

    num_sup = 100000

    dz = (z_max - z_min) / num_sup

    integral = 0.

    for i in range(0, num_sup):
        integral +=  np.exp((z_min + i * dz) ** 2) * special.erfc(z_min + i * dz)
    integral = integral * np.sqrt(np.pi) * dz

    r_0 = 1 / integral

    z_T = (mu - v_T) / np.sqrt(D)
    z_R = (mu - v_R) / np.sqrt(D)

    e2 = np.exp((z_R ** 2 - z_T ** 2) / 2)
    e4 = np.exp((z_R ** 2 - z_T ** 2) / 4)

    R1 = Re_D1 - e4 * Re_D3
    R2 = Re_D2 - e4 * Re_D4
    I1 = Im_D1 - e4 * Im_D3
    I2 = Im_D2 - e4 * Im_D4

    A = R2 ** 2 + I2 ** 2

    # 1) calculate Re(X) and Im(X)
    sus_x_real_theory = r_0 * omega * (omega * (R1 * R2 + I1 * I2) + I1 * R2 - R1 * I2) / (np.sqrt(D) * (1 + (omega) ** 2) * A)
    sus_x_imag_theory = r_0 * omega * (omega * (I1 * R2 - R1 * I2) - R1 * R2 - I1 * I2) / (np.sqrt(D) * (1 + (omega) ** 2) * A)

    # 2) calculate analytical result for X_v based on RRR
    sus_v_real_theory = (1 - (v_T-v_R) * (sus_x_real_theory - omega * sus_x_imag_theory)) / (1 + (omega) ** 2)
    sus_v_imag_theory = (omega - (v_T - v_R) * (sus_x_imag_theory + omega * sus_x_real_theory)) / (1 + (omega) ** 2)

    return sus_x_real_theory, sus_x_imag_theory, sus_v_real_theory, sus_v_imag_theory

def calc_sus_theory_ref(mu, Re_D1, Im_D1, Re_D2, Im_D2, Re_D3, Im_D3, Re_D4, Im_D4, omega):
    B_ft_real =  np.sin(omega * tau_ref) / (omega)
    B_ft_real[0] =  tau_ref # correct zero-frequency limit
    B_ft_imag =  (np.cos(omega * tau_ref) - 1) / (omega)
    B_ft_imag[0] = 0 # correct zero-frequency limit
   
    z_min = (mu - v_T) / (np.sqrt(2 * D))
    z_max = (mu - v_R) / (np.sqrt(2 * D))

    num_sup = 100000

    dz = (z_max - z_min) / num_sup

    integral = 0.

    for i in range(0, num_sup):
        integral +=  np.exp((z_min + i * dz) ** 2) * special.erfc(z_min + i * dz)
    integral = integral * np.sqrt(np.pi) * dz

    r_0 = 1 / (tau_ref + integral)

    # 0) set up abreviations 
    z_T = (mu - v_T) / np.sqrt(D)
    z_R = (mu - v_R) / np.sqrt(D)

    e2 = np.exp((z_R ** 2 - z_T ** 2) / 2)
    e4 = np.exp((z_R ** 2 - z_T ** 2) / 4)


    R1 = Re_D1 - e4 * Re_D3
    R2 = Re_D2 - e4 * (np.cos(tau_ref * omega) * Re_D4 - np.sin(tau_ref * omega) * Im_D4)
    I1 = Im_D1 - e4 * Im_D3
    I2 = Im_D2 - e4 * (np.sin(tau_ref * omega) * Re_D4 + np.cos(tau_ref * omega) * Im_D4)

    A = R2 ** 2 + I2 ** 2

    # 1) calculate Re(X) and Im(X)
    sus_x_real_theory = r_0 * omega * (omega * (R1 * R2 + I1 * I2) + I1 * R2 - R1 * I2) / (np.sqrt(D) * (1 + (omega) ** 2) * A)
    sus_x_imag_theory = -r_0 * omega * (omega * (I1 * R2 - R1 * I2) - R1 * R2 - I1 * I2) / (np.sqrt(D) * (1 + (omega) ** 2) * A)

    # calcuxlate RRR for LIF with white noise and refractory period
    sin = np.sin(omega * tau_ref)
    cos = np.cos(omega * tau_ref)

    sus_v_real_theory = (1 - r_0 * tau_ref + sus_x_real_theory * (v_R * (cos - omega * sin) - v_T - mu * (B_ft_real + omega * B_ft_imag)) + sus_x_imag_theory * (v_R * (sin + omega * cos) - omega * v_T + mu * (B_ft_imag - omega * B_ft_real))) / (1 + omega ** 2)
    sus_v_imag_theory = (omega * (1 - r_0 * tau_ref) + sus_x_real_theory * (v_R * (sin + omega * cos) - omega * v_T - mu * (-B_ft_imag + omega * B_ft_real)) - sus_x_imag_theory * (v_R * (cos - omega * sin) - v_T - mu * (B_ft_real + omega * B_ft_imag))) / (1 + omega ** 2)

    return sus_x_real_theory, sus_x_imag_theory, sus_v_real_theory, sus_v_imag_theory

# load data
sus_data_08 = np.loadtxt("./data/thesis/LIF_ref/res_s_LIF_ref_1.txt")
sus_data_08_alt = np.loadtxt("./data/thesis/LIF_ref/res_s_LIF_ref_7.txt")
mathematica_data_0 = np.genfromtxt("./data/parabolicCylinder/parabolicCylinder0.csv", delimiter = ",")
mathematica_data_05 = np.genfromtxt("./data/parabolicCylinder/parabolicCylinder05.csv", delimiter = ",")
mathematica_data_08 = np.genfromtxt("./data/parabolicCylinder/parabolicCylinder08.csv", delimiter = ",")
mathematica_data_12 = np.genfromtxt("./data/parabolicCylinder/parabolicCylinder12.csv", delimiter = ",")

sus_v_real_08 = sus_data_08[:,9]
sus_v_imag_08 = sus_data_08[:,10]

sus_v_real_08_alt = sus_data_08_alt[:,9]
sus_v_imag_08_alt = sus_data_08_alt[:,10]

Re_D1_0 = mathematica_data_0[:,1]
Im_D1_0 = mathematica_data_0[:,2]
Re_D2_0 = mathematica_data_0[:,3]
Im_D2_0 = mathematica_data_0[:,4]
Re_D3_0 = mathematica_data_0[:,5]
Im_D3_0 = mathematica_data_0[:,6]
Re_D4_0 = mathematica_data_0[:,7]
Im_D4_0 = mathematica_data_0[:,8]

Re_D1_05 = mathematica_data_05[:,1]
Im_D1_05 = mathematica_data_05[:,2]
Re_D2_05 = mathematica_data_05[:,3]
Im_D2_05 = mathematica_data_05[:,4]
Re_D3_05 = mathematica_data_05[:,5]
Im_D3_05 = mathematica_data_05[:,6]
Re_D4_05 = mathematica_data_05[:,7]
Im_D4_05 = mathematica_data_05[:,8]

Re_D1_08 = mathematica_data_08[:,1]
Im_D1_08 = mathematica_data_08[:,2]
Re_D2_08 = mathematica_data_08[:,3]
Im_D2_08 = mathematica_data_08[:,4]
Re_D3_08 = mathematica_data_08[:,5]
Im_D3_08 = mathematica_data_08[:,6]
Re_D4_08 = mathematica_data_08[:,7]
Im_D4_08 = mathematica_data_08[:,8]

Re_D1_12 = mathematica_data_12[:,1]
Im_D1_12 = mathematica_data_12[:,2]
Re_D2_12 = mathematica_data_12[:,3]
Im_D2_12 = mathematica_data_12[:,4]
Re_D3_12 = mathematica_data_12[:,5]
Im_D3_12 = mathematica_data_12[:,6]
Re_D4_12 = mathematica_data_12[:,7]
Im_D4_12 = mathematica_data_12[:,8]

# physical parameters
D = 0.1 # noise intensity
v_R = 0 # reset voltage
v_T = 1 # threshold voltage
mu_0 = 0 # mean input
mu_05 = 0.5 # mean input
mu_08 = 0.8 # mean input
mu_12 = 1.2 # mean input
N_frq_max = Re_D1_05.shape[0] # maximal stimulus frequenc

frequency = sus_data_08[:N_frq_max,0]
df = frequency[1]
omega = 2 * np.pi * frequency

N_frq = sus_data_08[:,0].shape[0]
steps = 2 * (N_frq - 1) # number of simulation steps
T = 1/df # simulation time
dt = T / steps 
tau_ref = 5000 * dt # refractory period

sus_x_real_theory_0, sus_x_imag_theory_0, sus_v_real_theory_0, sus_v_imag_theory_0 = calc_sus_theory(mu_0, Re_D1_0, Im_D1_0, Re_D2_0, Im_D2_0, Re_D3_0, Im_D3_0, Re_D4_0, Im_D4_0, omega)
sus_x_real_theory_05, sus_x_imag_theory_05, sus_v_real_theory_05, sus_v_imag_theory_05 = calc_sus_theory(mu_05, Re_D1_05, Im_D1_05, Re_D2_05, Im_D2_05, Re_D3_05, Im_D3_05, Re_D4_05, Im_D4_05, omega)
sus_x_real_theory_08, sus_x_imag_theory_08, sus_v_real_theory_08, sus_v_imag_theory_08 = calc_sus_theory(mu_08, Re_D1_08, Im_D1_08, Re_D2_08, Im_D2_08, Re_D3_08, Im_D3_08, Re_D4_08, Im_D4_08, omega)
sus_x_real_theory_12, sus_x_imag_theory_12, sus_v_real_theory_12, sus_v_imag_theory_12 = calc_sus_theory(mu_12, Re_D1_12, Im_D1_12, Re_D2_12, Im_D2_12, Re_D3_12, Im_D3_12, Re_D4_12, Im_D4_12, omega)

sus_x_real_theory_08_ref, sus_x_imag_theory_08_ref, sus_v_real_theory_08_ref, sus_v_imag_theory_08_ref = calc_sus_theory_ref(mu_08, Re_D1_08, Im_D1_08, Re_D2_08, Im_D2_08, Re_D3_08, Im_D3_08, Re_D4_08, Im_D4_08, omega)

sus_OUP_real = 1 / (1 + omega ** 2)
sus_OUP_imag = omega / (1 + omega ** 2)

# compare chi_x and chi_v for differen mu
fig = plt.figure(figsize=(14, 6.28))
# fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(234, sharex = ax1)
ax3 = fig.add_subplot(232, sharey = ax1)
ax4 = fig.add_subplot(235, sharex = ax3, sharey = ax2)
ax5 = fig.add_subplot(233, sharey = ax1)
ax6 = fig.add_subplot(236, sharex = ax5, sharey = ax2)


f_min = 1.5 * df
f_max = 60

ax1.plot(frequency, sus_x_real_theory_0, "-", color = (0, 0.290, 0.565), linewidth = 3)
ax1.plot(frequency, sus_v_real_theory_0, "-", color = (0.961, 0.314, 0.114), linewidth = 3)
ax1.plot(frequency, sus_OUP_real, "-k", linewidth = 3)
ax1.set_xscale("log")
ax1.set_xlim(f_min, f_max)
ax1.set_ylim(-0.1, 1.1)
ax1.tick_params(axis = "y", labelsize = 24)
ax1.tick_params(axis = "x", labelbottom=False)
ax1.set_ylabel(r"Re$(\chi$)", fontsize=35)
# ax1.text(1.7e1, 0.85, r"$\mathbf{A_1}$", fontsize = 35)
ax1.set_title(r"$\mathbf{A}$", fontsize=35)

ax2.plot(frequency, sus_x_imag_theory_0, "-", color = (0, 0.290, 0.565), linewidth = 3)
ax2.plot(frequency, sus_v_imag_theory_0, "-", color = (0.961, 0.314, 0.114), linewidth = 3)
ax2.plot(frequency, sus_OUP_imag, "-k", linewidth = 3)
ax2.set_xscale("log")
ax2.set_xlim(f_min, f_max)
ax2.set_ylim(-0.05, 0.6)
ax2.tick_params(axis = "both", labelsize = 24)
ax2.set_ylabel(r"Im$(\chi$)", fontsize=35)
ax2.set_xlabel(r"$f \tau _m$", fontsize=35)
# ax2.text(1.66e-2, 0.465, r"$\mathbf{A_2}$", fontsize = 35)
ax2.legend([r"$\chi_x$", r"$\chi_v$", "OUP"], fontsize=22)

ax3.plot(frequency, sus_x_real_theory_05, "-", color = (0, 0.290, 0.565), linewidth = 3)
ax3.plot(frequency, sus_v_real_theory_05, "-", color = (0.961, 0.314, 0.114), linewidth = 3)
ax3.set_xscale("log")
ax3.set_xlim(f_min, f_max)
ax3.set_ylim(-0.1, 1.1)
ax3.tick_params(axis = "y", labelleft = False)
ax3.tick_params(axis = "x", labelbottom=False)
# ax3.text(1.7e1, 0.85, r"$\mathbf{B_1}$", fontsize = 35)
ax3.set_title(r"$\mathbf{B}$", fontsize=35)

ax4.plot(frequency, sus_x_imag_theory_05, "-", color = (0, 0.290, 0.565), linewidth = 3)
ax4.plot(frequency, sus_v_imag_theory_05, "-", color = (0.961, 0.314, 0.114), linewidth = 3)
ax4.set_xscale("log")
ax4.set_xlim(f_min, f_max)
ax4.set_ylim(-0.05, 0.6)
ax4.tick_params(axis = "y", labelleft = False)
ax4.tick_params(axis = "x", labelsize = 24)
ax4.set_xlabel(r"$f \tau _m$", fontsize=35)
# ax4.text(1.66e-2, 0.465, r"$\mathbf{B_2}$", fontsize = 35)
# ax4.text(1.5, 0.5, r"$r_0 \approx 0.15\frac{1}{\tau_m}$", fontsize = 15)

ax5.plot(frequency, sus_x_real_theory_12, "-", color = (0, 0.290, 0.565), linewidth = 3)
ax5.plot(frequency, sus_v_real_theory_12, "-", color = (0.961, 0.314, 0.114), linewidth = 3)
ax5.set_xscale("log")
ax5.set_xlim(f_min, f_max)
ax5.set_ylim(-0.1, 1.1)
ax5.tick_params(axis = "y", labelleft = False)
ax5.tick_params(axis = "x", labelbottom=False)
# ax5.text(1.7e1, 0.85, r"$\mathbf{C_1}$", fontsize = 35)
ax5.set_title(r"$\mathbf{C}$", fontsize=35)

ax6.plot(frequency, sus_x_imag_theory_12, "-", color = (0, 0.290, 0.565), linewidth = 3)
ax6.plot(frequency, sus_v_imag_theory_12, "-", color = (0.961, 0.314, 0.114), linewidth = 3)
ax6.set_xscale("log")
ax6.set_xlim(f_min, f_max)
ax6.set_ylim(-0.05, 0.6)
ax6.tick_params(axis = "y", labelleft = False)
ax6.tick_params(axis = "x", labelsize = 24)
ax6.set_xlabel(r"$f \tau _m$", fontsize=35)
# ax6.text(1.66e-2, 0.465, r"$\mathbf{C_2}$", fontsize = 35)
# ax6.text(1.5, 0.5, r"$r_0 \approx 0.73\frac{1}{\tau_m}$", fontsize = 15)

fig.align_ylabels()

plt.tight_layout()
plt.savefig("./pictures/paper/2a_RRR_LIF_white.pdf") 


# compare simulation data with analytical result for tau_ref =/= 0

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

f_min = df
f_max = 60


ax1.plot(frequency, sus_v_real_08[:N_frq_max], "o", markersize = 6, color = (0.961, 0.314, 0.114))
# ax1.plot(frequency, sus_v_real_08_alt[:N_frq_max], "o", markersize = 6, color = (0, 0.290, 0.565))
ax1.plot(frequency, sus_v_real_theory_08_ref, "-k", linewidth = 3)
ax1.plot(frequency, sus_v_real_theory_08, "--k", linewidth = 3, alpha = 0.3)
ax1.plot(frequency, sus_v_real_08[:N_frq_max], "o", markersize = 6, color = (0.961, 0.314, 0.114))
# ax1.plot(frequency, sus_v_real_08_alt[:N_frq_max], "o", markersize = 6, color = (0, 0.290, 0.565))
ax1.plot(frequency, sus_v_real_theory_08_ref, "-k", linewidth = 3)
ax1.set_ylabel(r"$\mathrm{Re}(\chi_v)$", fontsize=35)
ax1.set_xscale("log")
ax1.set_xlim(f_min, f_max)
ax1.set_ylim(-0.02, 0.2)
ax1.tick_params(axis = "y", labelsize = 24)
ax1.tick_params(axis = "x", labelbottom = False)
# ax1.legend([r"$\varepsilon = 1$", r"$\varepsilon = \sqrt{0.1}$", "RRR"], fontsize=24)
ax1.legend(["simulation data", "RRR", r"$\tau_{\mathrm{ref}} = 0$"], fontsize=24)

ax2.plot(frequency, sus_v_imag_theory_08, "--k", linewidth = 3, alpha = 0.3)
ax2.plot(frequency, -sus_v_imag_08[:N_frq_max], "o", markersize = 6, color = (0.961, 0.314, 0.114))
# ax2.plot(frequency, -sus_v_imag_08_alt[:N_frq_max], "o", markersize = 6, color = (0, 0.290, 0.565))
ax2.plot(frequency, sus_v_imag_theory_08_ref, "-k", linewidth = 3)
ax2.set_xlabel(r"$f \tau _m$", fontsize=35)
ax2.set_ylabel(r"$\mathrm{Im}(\chi_v)$", fontsize=35)
ax2.set_xscale("log")
ax2.set_xlim(f_min, f_max)
ax2.set_ylim(-0.12, 0.12)
ax2.tick_params(axis = "both", labelsize = 24)

fig.align_ylabels()

plt.tight_layout()
plt.savefig("./pictures/paper/2b_RRR_LIF_white_ref.pdf") 