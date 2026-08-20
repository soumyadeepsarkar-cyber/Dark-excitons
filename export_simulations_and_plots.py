import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, mesolve, expect
import json
import base64
from io import BytesIO

print("Running 6-level Adiabatically Eliminated simulation...")
# ----------------------------------------------------
# 6-LEVEL SIMULATION
# ----------------------------------------------------
dim6 = 6
Delta_L1 = 1500.0
Delta_L_minus = 1510.0
g = 20.0
E_7 = 130.0
E_8 = -130.0
Omega_max = 150.0
eps_H = 0.10 * np.exp(1j * 0.0)
eps_V = 0.10 * np.exp(1j * np.pi / 4.0)
t0 = 2500.0
sigma = 650.0

tlist = np.linspace(0, 5000, 2000)
pulse_Omega_arr = Omega_max * np.exp(-((tlist - t0)**2) / (2 * sigma**2))
pulse_Omega_sq_arr = pulse_Omega_arr**2

ket_L_plus = (1 + eps_V)*basis(6, 0) + (1 + eps_H)*basis(6, 1)
ket_C_plus = g * (basis(6, 2) + basis(6, 3) + eps_H*basis(6, 4) + eps_V*basis(6, 5))
ket_L_minus = eps_H*basis(6, 0) + eps_V*basis(6, 1)
ket_C_minus = g * (eps_V*basis(6, 2) + eps_H*basis(6, 3))

def proj6(i, j):
    return basis(6, i) * basis(6, j).dag()

H0_6 = E_7 * proj6(4, 4) + E_8 * proj6(5, 5)
H_cav_cav = -(1/Delta_L1) * (ket_C_plus * ket_C_plus.dag()) - (1/Delta_L_minus) * (ket_C_minus * ket_C_minus.dag())
H_static_6 = H0_6 + H_cav_cav

H_Stark = -(1/Delta_L1) * (ket_L_plus * ket_L_plus.dag()) - (1/Delta_L_minus) * (ket_L_minus * ket_L_minus.dag())
H_Raman_plus = -(1/Delta_L1) * (ket_C_plus * ket_L_plus.dag() + ket_L_plus * ket_C_plus.dag())
H_Raman_minus = -(1/Delta_L_minus) * (ket_C_minus * ket_L_minus.dag() + ket_L_minus * ket_C_minus.dag())
H_Raman = H_Raman_plus + H_Raman_minus

H_eff = [H_static_6, [H_Stark, pulse_Omega_sq_arr], [H_Raman, pulse_Omega_arr]]

psi0_6 = (basis(6, 0) + basis(6, 1)).unit()
rho0_6 = psi0_6 * psi0_6.dag()
psi_target_6 = (basis(6, 2) + basis(6, 3)).unit()

result_6 = mesolve(H_eff, rho0_6, tlist, c_ops=[], options={'nsteps': 500000})

t_ns = tlist / 1519.26
pop_6 = {
    r'$|1\rangle$ ($\uparrow_x, 0$)': [expect(proj6(0, 0), rho).real for rho in result_6.states],
    r'$|2\rangle$ ($\downarrow_x, 0$)': [expect(proj6(1, 1), rho).real for rho in result_6.states],
    r'$|5\rangle$ ($\downarrow_x, 1V$) [Target]': [expect(proj6(2, 2), rho).real for rho in result_6.states],
    r'$|6\rangle$ ($\uparrow_x, 1H$) [Target]': [expect(proj6(3, 3), rho).real for rho in result_6.states],
    r'$|7\rangle$ ($\downarrow_x, 1H$) [Leakage]': [expect(proj6(4, 4), rho).real for rho in result_6.states],
    r'$|8\rangle$ ($\uparrow_x, 1V$) [Leakage]': [expect(proj6(5, 5), rho).real for rho in result_6.states],
}
fidelity_6 = [expect(psi_target_6 * psi_target_6.dag(), rho).real for rho in result_6.states]

# Save 6-level Population Plot
plt.figure(figsize=(10, 6), dpi=300)
for label, pop in pop_6.items():
    plt.plot(t_ns, pop, label=label, linewidth=1.8)
plt.xlabel('Time (ns)', fontsize=12)
plt.ylabel('Population', fontsize=12)
plt.title('6-Level Effective Model: Population Dynamics', fontsize=14, fontweight='bold')
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('vstirap_6level_populations.png')
plt.close()

# Save 6-level Fidelity Plot
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(t_ns, fidelity_6, label=r'Bell State Fidelity $F(t)$', color='darkblue', linewidth=2)
plt.xlabel('Time (ns)', fontsize=12)
plt.ylabel('Fidelity', fontsize=12)
plt.title(r'6-Level Effective Model: Target Bell State Fidelity $F(t) = \langle\Psi_{\rm target}|\rho(t)|\Psi_{\rm target}\rangle$', fontsize=13, fontweight='bold')
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('vstirap_6level_fidelity.png')
plt.close()

print("Running 8-level Full simulation...")
# ----------------------------------------------------
# 8-LEVEL SIMULATION
# ----------------------------------------------------
dim8 = 8
states8 = {str(i): basis(dim8, i-1) for i in range(1, 9)}
def proj8(i, j):
    return states8[str(i)] * states8[str(j)].dag()

Delta_h = 10.0
Delta_L_minus_8 = Delta_L1 + Delta_h
Delta_e_2 = 130.0
g_H2 = 20.0
g_V1 = 20.0
Omega_H1_max = 150.0
Omega_V2_max = 150.0

pulse_H1_arr = Omega_H1_max * np.exp(-((tlist - t0)**2) / (2 * sigma**2))
pulse_V2_arr = Omega_V2_max * np.exp(-((tlist - t0)**2) / (2 * sigma**2))

H0_8 = (Delta_L1 * proj8(3, 3) + 
        Delta_L_minus_8 * proj8(4, 4) + 
        0.0 * proj8(5, 5) + 
        0.0 * proj8(6, 6) + 
        E_7 * proj8(7, 7) + 
        E_8 * proj8(8, 8))

H_cav_ideal_plus = g_V1 * proj8(3, 5) + g_H2 * proj8(3, 6)
H_cav_anom_plus = eps_H * g_H2 * proj8(3, 7) + eps_V * g_V1 * proj8(3, 8)
H_cav_anom_minus = eps_H * g_H2 * proj8(4, 6) + eps_V * g_V1 * proj8(4, 5)
H_cav_8 = H_cav_ideal_plus + H_cav_anom_plus + H_cav_anom_minus
H_cav_8 = H_cav_8 + H_cav_8.dag()

H_H1_8 = proj8(3, 1) + eps_H * proj8(3, 2) + eps_H * proj8(4, 1)
H_H1_8 = H_H1_8 + H_H1_8.dag()

H_V2_8 = proj8(3, 2) + eps_V * proj8(3, 1) + eps_V * proj8(4, 2)
H_V2_8 = H_V2_8 + H_V2_8.dag()

H_8 = [H0_8 + H_cav_8, [H_H1_8, pulse_H1_arr], [H_V2_8, pulse_V2_arr]]

psi0_8 = (states8['1'] + states8['2']).unit()
rho0_8 = psi0_8 * psi0_8.dag()
psi_target_8 = (states8['5'] + states8['6']).unit()

result_8 = mesolve(H_8, rho0_8, tlist, c_ops=[], options={'nsteps': 500000})

pop_8 = {
    r'$|1\rangle$ ($\uparrow_x, 0$)': [expect(proj8(1, 1), rho).real for rho in result_8.states],
    r'$|2\rangle$ ($\downarrow_x, 0$)': [expect(proj8(2, 2), rho).real for rho in result_8.states],
    r'$|3\rangle$ ($T_+, 0$) [Trion]': [expect(proj8(3, 3), rho).real for rho in result_8.states],
    r'$|4\rangle$ ($T_-, 0$) [Trion]': [expect(proj8(4, 4), rho).real for rho in result_8.states],
    r'$|5\rangle$ ($\downarrow_x, 1V$) [Target]': [expect(proj8(5, 5), rho).real for rho in result_8.states],
    r'$|6\rangle$ ($\uparrow_x, 1H$) [Target]': [expect(proj8(6, 6), rho).real for rho in result_8.states],
    r'$|7\rangle$ ($\downarrow_x, 1H$) [Leakage]': [expect(proj8(7, 7), rho).real for rho in result_8.states],
    r'$|8\rangle$ ($\uparrow_x, 1V$) [Leakage]': [expect(proj8(8, 8), rho).real for rho in result_8.states],
}
fidelity_8 = [expect(psi_target_8 * psi_target_8.dag(), rho).real for rho in result_8.states]

# Save 8-level Population Plot
plt.figure(figsize=(10, 6), dpi=300)
for label, pop in pop_8.items():
    plt.plot(t_ns, pop, label=label, linewidth=1.8)
plt.xlabel('Time (ns)', fontsize=12)
plt.ylabel('Population', fontsize=12)
plt.title('8-Level Full Model: Population Dynamics', fontsize=14, fontweight='bold')
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('vstirap_8level_populations.png')
plt.close()

# Save 8-level Fidelity Plot
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(t_ns, fidelity_8, label=r'Bell State Fidelity $F(t)$', color='darkred', linewidth=2)
plt.xlabel('Time (ns)', fontsize=12)
plt.ylabel('Fidelity', fontsize=12)
plt.title(r'8-Level Full Model: Target Bell State Fidelity $F(t) = \langle\Psi_{\rm target}|\rho(t)|\Psi_{\rm target}\rangle$', fontsize=13, fontweight='bold')
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('vstirap_8level_fidelity.png')
plt.close()

# Save Comparison Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=300)

axes[0].plot(t_ns, fidelity_8, label='8-Level Full Model', color='crimson', linewidth=2)
axes[0].plot(t_ns, fidelity_6, label='6-Level Effective Model', color='dodgerblue', linestyle='--', linewidth=2)
axes[0].set_xlabel('Time (ns)', fontsize=12)
axes[0].set_ylabel('Fidelity', fontsize=12)
axes[0].set_title('Target Bell State Fidelity Comparison', fontsize=13, fontweight='bold')
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(fontsize=11)

target_pop_8 = np.array(pop_8[r'$|5\rangle$ ($\downarrow_x, 1V$) [Target]']) + np.array(pop_8[r'$|6\rangle$ ($\uparrow_x, 1H$) [Target]'])
leak_pop_8 = np.array(pop_8[r'$|7\rangle$ ($\downarrow_x, 1H$) [Leakage]']) + np.array(pop_8[r'$|8\rangle$ ($\uparrow_x, 1V$) [Leakage]'])

target_pop_6 = np.array(pop_6[r'$|5\rangle$ ($\downarrow_x, 1V$) [Target]']) + np.array(pop_6[r'$|6\rangle$ ($\uparrow_x, 1H$) [Target]'])
leak_pop_6 = np.array(pop_6[r'$|7\rangle$ ($\downarrow_x, 1H$) [Leakage]']) + np.array(pop_6[r'$|8\rangle$ ($\uparrow_x, 1V$) [Leakage]'])

axes[1].plot(t_ns, target_pop_8, label='Target States (|5>+|6>) [8-Level]', color='green', linewidth=2)
axes[1].plot(t_ns, target_pop_6, label='Target States (|5>+|6>) [6-Level]', color='lime', linestyle='--', linewidth=2)
axes[1].plot(t_ns, leak_pop_8, label='Leakage States (|7>+|8>) [8-Level]', color='purple', linewidth=2)
axes[1].plot(t_ns, leak_pop_6, label='Leakage States (|7>+|8>) [6-Level]', color='magenta', linestyle='--', linewidth=2)
axes[1].set_xlabel('Time (ns)', fontsize=12)
axes[1].set_ylabel('Total Subspace Population', fontsize=12)
axes[1].set_title('Target vs Leakage Population Comparison', fontsize=13, fontweight='bold')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('vstirap_model_comparison.png')
plt.close()

print("All plots generated and exported successfully!")
