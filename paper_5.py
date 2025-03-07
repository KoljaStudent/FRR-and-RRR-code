import numpy as np
import matplotlib.pyplot as plt
from scipy import special

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Computer Modern'

# plt.rcParams['mathtext.fontset'] = 'dejavuserif'
# plt.rcParams['mathtext.fontset'] = 'cm'

def calc_S_vv_theory(mu, Re_D1, Im_D1, Re_D2, Im_D2, Re_D3, Im_D3, Re_D4, Im_D4):
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

    # 1) calculate S_xx(f) 
    S_xx_theory = r_0 * (Re_D2 ** 2 + Im_D2 ** 2 - e2 * (Re_D4 ** 2 + Im_D4 ** 2)) / A
    
    # 2) calculate Re(X) and Im(X)
    sus_x_real_theory = r_0 * omega_0[:N_frq_max] * (omega_0[:N_frq_max] * (R1 * R2 + I1 * I2) + I1 * R2 - R1 * I2) / (np.sqrt(D) * (1 + (omega_0[:N_frq_max]) ** 2) * A)

    return ((v_T - v_R) ** 2 * S_xx_theory + 2 * D * (1 - 2 * (v_T - v_R) * sus_x_real_theory)) / (1 + (omega_0[:N_frq_max]) ** 2)

# load data 
spec_data_1_a = np.loadtxt("./data/thesis/LIF_white/res_LIF_white_4.txt")

spec_data_1_b = np.loadtxt("./data/thesis/LIF_white/res_LIF_white_6.txt")

mathematica_data_1_a = np.genfromtxt("./data/parabolicCylinder/parabolicCylinder08.csv", delimiter = ",")
mathematica_data_1_b = np.genfromtxt("./data/parabolicCylinder/parabolicCylinder12.csv", delimiter = ",")

spec_data_0 = np.loadtxt("./data/thesis/LIF_white/res_LIF_white_8.txt")
mathematica_data_0 = np.genfromtxt("./data/parabolicCylinder/parabolicCylinder0.csv", delimiter = ",")
mathematica_data_12 = np.genfromtxt("./data/parabolicCylinder/parabolicCylinder12.csv", delimiter = ",")

mathematica_data_ec = np.genfromtxt("./data/parabolicCylinder/parabolicCylinder_edge_case.csv", delimiter = ",")

""" 
    2) LIF + white noise plots for voltage power-spectrum
"""


Re_D1_0 = mathematica_data_0[:,1]
Im_D1_0 = mathematica_data_0[:,2]
Re_D2_0 = mathematica_data_0[:,3]
Im_D2_0 = mathematica_data_0[:,4]
Re_D3_0 = mathematica_data_0[:,5]
Im_D3_0 = mathematica_data_0[:,6]
Re_D4_0 = mathematica_data_0[:,7]
Im_D4_0 = mathematica_data_0[:,8]

Re_D1_08 = mathematica_data_1_a[:,1]
Im_D1_08 = mathematica_data_1_a[:,2]
Re_D2_08 = mathematica_data_1_a[:,3]
Im_D2_08 = mathematica_data_1_a[:,4]
Re_D3_08 = mathematica_data_1_a[:,5]
Im_D3_08 = mathematica_data_1_a[:,6]
Re_D4_08 = mathematica_data_1_a[:,7]
Im_D4_08 = mathematica_data_1_a[:,8]


Re_D1_12 = mathematica_data_12[:,1]
Im_D1_12 = mathematica_data_12[:,2]
Re_D2_12 = mathematica_data_12[:,3]
Im_D2_12 = mathematica_data_12[:,4]
Re_D3_12 = mathematica_data_12[:,5]
Im_D3_12 = mathematica_data_12[:,6]
Re_D4_12 = mathematica_data_12[:,7]
Im_D4_12 = mathematica_data_12[:,8]

frequency_ec = mathematica_data_ec[:,0]
Re_D1_ec = mathematica_data_ec[:,1]
Im_D1_ec = mathematica_data_ec[:,2]
Re_D2_ec = mathematica_data_ec[:,3]
Im_D2_ec = mathematica_data_ec[:,4]
Re_D3_ec = mathematica_data_ec[:,5]
Im_D3_ec = mathematica_data_ec[:,6]
Re_D4_ec = mathematica_data_ec[:,7]
Im_D4_ec = mathematica_data_ec[:,8]

frequency_0 = spec_data_0[:,0]
frequency_08 = spec_data_1_a[:,0]
frequency_12 = spec_data_1_b[:,0]
S_vv_0 = spec_data_0[:,5]
S_vv_08 = spec_data_1_a[:,5]
S_vv_12 = spec_data_1_b[:,5]

omega_0 = 2 * np.pi * frequency_0

df_ec = frequency_ec[1]
omega_ec = 2 * np.pi * frequency_ec

# simulation parameters
df = frequency_0[1]
N_frq = frequency_0.shape[0]
steps = 2 * (N_frq - 1) # number of simulation steps
T = 1/df # simulation time
dt = T / steps
f_nyquist = 0.5 / dt # maximal measurable frequency

# physical parameters
D = 0.1 # noise intensity
v_R = 0 # reset voltage
v_T = 1 # threshold voltage
mu_0 = 0 # mean input
mu_08 = 0.8 # mean input
mu_12 = 1.2 # mean input
N_frq_max = Re_D1_08.shape[0] # maximal stimulus frequency

# calculate analytical results 
S_vv_theory_0 = calc_S_vv_theory(mu_0, Re_D1_0, Im_D1_0, Re_D2_0, Im_D2_0, Re_D3_0, Im_D3_0, Re_D4_0, Im_D4_0)
S_vv_theory_08 = calc_S_vv_theory(mu_08, Re_D1_08, Im_D1_08, Re_D2_08, Im_D2_08, Re_D3_08, Im_D3_08, Re_D4_08, Im_D4_08)
S_vv_theory_12 = calc_S_vv_theory(mu_12, Re_D1_12, Im_D1_12, Re_D2_12, Im_D2_12, Re_D3_12, Im_D3_12, Re_D4_12, Im_D4_12)
S_vv_theory_OUP = 2 * D / (1 + omega_0 ** 2)

# compare voltage FRR for noise and mean driven regime
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(111)

f_min = df
f_max = 60

#(0.961, 0.314, 0.114)(0.965, 0.392, 0.216)(0, 0.514, 1)(0.969, 0.482, 0.329)(0.98, 0.627, 0.522)(0.863, 0.231, 0.039)
ax1.plot(frequency_0, S_vv_0, "o-", markersize = 6, color = (0.863, 0.231, 0.039), mfc = (0.98, 0.627, 0.522), mec = (0.98, 0.627, 0.522))
ax1.plot(frequency_08, S_vv_08, "o-", markersize = 6, color = (0, 0.290, 0.565), mfc = (0.18, 0.604, 1), mec = (0.18, 0.604, 1))
ax1.plot(frequency_12, S_vv_12, "o-", markersize = 6, color = (0.290,0.565,0), mfc = (0.514, 1, 0), mec = (0.514, 1, 0))

ax1.plot(frequency_0, S_vv_0, "o", markersize = 6, color = (0.98, 0.627, 0.522))
ax1.plot(frequency_0[:N_frq_max], S_vv_theory_0, "-", linewidth = 3, color = (0.863, 0.231, 0.039))
# ax1.plot(frequency_0, S_vv_theory_OUP, "k--", linewidth = 3)

ax1.plot(frequency_08, S_vv_08, "o", markersize = 6, color = (0.18, 0.604, 1))
ax1.plot(frequency_0[:N_frq_max], S_vv_theory_08, "-", linewidth = 3, color = (0, 0.290, 0.565))

ax1.plot(frequency_12, S_vv_12, "o", markersize = 6, color = (0.514, 1, 0))
ax1.plot(frequency_0[:N_frq_max], S_vv_theory_12, "-", linewidth = 3,  color = (0.290,0.565,0))
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(f_min, f_max)
ax1.set_ylim(1e-3, 3e-1)
ax1.set_ylabel(r"$S_{vv}$", fontsize=35)
ax1.set_xlabel(r"$f \tau_m$", fontsize=35)
ax1.tick_params(axis = "both", labelsize = 24)
ax1.legend([r"$\mu = 0$", r"$\mu = 0.8$", r" $\mu = 1.2$"], fontsize=24)

plt.tight_layout()
plt.savefig("./pictures/paper/3_S_vv.pdf") 

# check S_vv for edgecase 
mu = 0.95 # mean input
D = 0.00001 # noise intensity

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

R1 = Re_D1_ec - e4 * Re_D3_ec
R2 = Re_D2_ec - e4 * Re_D4_ec
I1 = Im_D1_ec - e4 * Im_D3_ec
I2 = Im_D2_ec - e4 * Im_D4_ec

A = R2 ** 2 + I2 ** 2

# 1) calculate S_xx(f) 
S_xx_theory = r_0 * (Re_D2_ec ** 2 + Im_D2_ec ** 2 - e2 * (Re_D4_ec ** 2 + Im_D4_ec ** 2)) / A
    
# 2) calculate Re(X) and Im(X)
sus_x_real_theory = r_0 * omega_ec * (omega_ec * (R1 * R2 + I1 * I2) + I1 * R2 - R1 * I2) / (np.sqrt(D) * (1 + (omega_ec) ** 2) * A)

S_vv_theory_ec = ((v_T - v_R) ** 2 * S_xx_theory + 2 * D * (1 - 2 * (v_T - v_R) * sus_x_real_theory)) / (1 + (omega_ec) ** 2)

print(S_vv_theory_ec[:10])

# compare voltage FRR for noise and mean driven regime
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(111)

f_min = df_ec
f_max = 60


ax1.plot(frequency_ec, S_vv_theory_ec, "-", linewidth = 3, color = (0.863, 0.231, 0.039))
ax1.plot(frequency_0, S_vv_theory_OUP, "k--", linewidth = 3)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(f_min, f_max)
# ax1.set_ylim(4e-6, 3e1)
ax1.set_ylabel(r"$S_{vv}$", fontsize=35)
ax1.set_xlabel(r"$f \tau_m$", fontsize=35)
ax1.tick_params(axis = "both", labelsize = 24)
ax1.legend([r"$\mu = 0$", r"$\mu = 0.8$", r" $\mu = 1.2$"], fontsize=24)

plt.tight_layout()
plt.savefig("./pictures/S_vv_edge_case.png") 
