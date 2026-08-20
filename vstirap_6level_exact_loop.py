import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, mesolve, expect, Qobj

# 1. The 6-State Basis & Energies
# i=0: |up_x, 0> (Ground 1) -> E_0 = 0.0
# i=1: |down_x, 0> (Ground 2) -> E_1 = 0.0
# i=2: |down_x, 1V> (Target 1) -> E_2 = 0.0
# i=3: |up_x, 1H> (Target 2) -> E_3 = 0.0
# i=4: |down_x, 1H> (Leakage 1) -> E_4 = 130.0
# i=5: |up_x, 1V> (Leakage 2) -> E_5 = -130.0
E_base = [0.0, 0.0, 0.0, 0.0, 130.0, -130.0]

# Virtual Trion states
# k=0: |T_+> -> Delta_0 = 1500.0
# k=1: |T_-> -> Delta_1 = 1510.0
Delta = [1500.0, 1510.0]

# 2. Physical Parameters
g = 20.0
eps_H = 0.10 * np.exp(1j * 0.0)
eps_V = 0.10 * np.exp(1j * (np.pi / 4.0))
Omega_max = 150.0

# Pulse Envelopes
t0 = 2500.0
sigma = 650.0

def pulse_Omega(t, args):
    return Omega_max * np.exp(-((t - t0)**2) / (2 * sigma**2))

def pulse_Omega_sq(t, args):
    return pulse_Omega(t, args)**2

# 3. The Coupling Arrays (V)
V_L = np.zeros((6, 2), dtype=complex)
V_C = np.zeros((6, 2), dtype=complex)

# V_L (Laser Couplings - Only connect to ground states 0 and 1)
V_L[0, 0] = 1.0 + eps_V
V_L[1, 0] = 1.0 + eps_H
V_L[0, 1] = eps_H
V_L[1, 1] = eps_V

# V_C (Cavity Couplings - Only connect to photon states 2, 3, 4, and 5)
V_C[2, 0] = g
V_C[3, 0] = g
V_C[4, 0] = eps_H * g
V_C[5, 0] = eps_V * g
V_C[2, 1] = eps_V * g
V_C[3, 1] = eps_H * g

# 4. Programmatic Effective Hamiltonian (The Loop)
H_Stark_mat = np.zeros((6, 6), dtype=complex)
H_cav_mat = np.zeros((6, 6), dtype=complex)
H_Raman_mat = np.zeros((6, 6), dtype=complex)

for i in range(6):
    for j in range(6):
        for k in range(2):
            denom = 0.5 * (1.0 / (Delta[k] - E_base[i]) + 1.0 / (Delta[k] - E_base[j]))
            
            H_Stark_mat[i, j] += - (V_L[i, k] * np.conj(V_L[j, k])) * denom
            H_cav_mat[i, j] += - (V_C[i, k] * np.conj(V_C[j, k])) * denom
            H_Raman_mat[i, j] += - (V_C[i, k] * np.conj(V_L[j, k]) + V_L[i, k] * np.conj(V_C[j, k])) * denom

# Convert to QuTiP Qobj
H_Stark = Qobj(H_Stark_mat)
H_cav = Qobj(H_cav_mat)
H_Raman = Qobj(H_Raman_mat)

# 5. Simulation Setup & Output
H0 = E_base[4] * basis(6, 4) * basis(6, 4).dag() + E_base[5] * basis(6, 5) * basis(6, 5).dag()
H = [H0 + H_cav, [H_Stark, pulse_Omega_sq], [H_Raman, pulse_Omega]]

psi0 = (basis(6, 0) + basis(6, 1)).unit()
psi_target = (basis(6, 2) + basis(6, 3)).unit()
tlist = np.linspace(0, 5000, 2000)

print("Starting mesolve integration...")
# Standard integration (no stiff solver/high nsteps needed per user instruction)
result = mesolve(H, psi0, tlist, c_ops=[], options={'nsteps': 500000})
print("Integration complete!")

# Outputs
t_ns = tlist / 1519.26

# Populations
labels = ['0: |up_x, 0>', '1: |down_x, 0>', '2: |down_x, 1V>', '3: |up_x, 1H>', '4: |down_x, 1H>', '5: |up_x, 1V>']
populations = {}
for idx in range(6):
    op = basis(6, idx) * basis(6, idx).dag()
    populations[idx] = [expect(op, state).real for state in result.states]

# Fidelity
fidelity = [expect(psi_target * psi_target.dag(), state).real for state in result.states]

# Plot Population Dynamics
plt.figure(figsize=(10, 6))
for idx in range(6):
    plt.plot(t_ns, populations[idx], label=labels[idx])
plt.xlabel('Time (ns)')
plt.ylabel('Population')
plt.title('Exact Programmatic 6-Level vSTIRAP Dynamics')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.savefig('exact_loop_populations.png')
print("Saved exact_loop_populations.png")

# Plot Fidelity
plt.figure(figsize=(10, 6))
plt.plot(t_ns, fidelity, label='Target Bell State Fidelity', color='black', linewidth=2)
plt.xlabel('Time (ns)')
plt.ylabel('Fidelity')
plt.title('Instantaneous Fidelity (Exact Denominator Model)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('exact_loop_fidelity.png')
print("Saved exact_loop_fidelity.png")

# Final populations of State 0 and State 1
pop_0_final = populations[0][-1]
pop_1_final = populations[1][-1]
print("\n--- Final Ground State Populations ---")
print(f"State 0 (|up_x, 0>): {pop_0_final:.6f}")
print(f"State 1 (|down_x, 0>): {pop_1_final:.6f}")
print(f"Difference: {abs(pop_0_final - pop_1_final):.6f}")
