import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import diffrax

# 2. State Basis and Physical Parameters
# 0: |↑_x, 0> (Ground 1)
# 1: |↓_x, 0> (Ground 2)
# 2: |T_+, 0> (Virtual Trion 1)
# 3: |T_-, 0> (Virtual Trion 2)
# 4: |↓_x, 1V> (Target 1)
# 5: |↑_x, 1H> (Target 2)
# 6: |↓_x, 1H> (Leakage 1)
# 7: |↑_x, 1V> (Leakage 2)

Delta_L1 = 1500.0
Delta_L_minus = 1510.0
E_7 = 130.0
E_8 = -130.0
g = 20.0
eps_H = 0.10 * jnp.exp(1j * 0.0)
eps_V = 0.10 * jnp.exp(1j * jnp.pi / 4.0)
Omega_max = 150.0
t0 = 2500.0
sigma = 650.0

def pulse_Omega(t):
    return Omega_max * jnp.exp(-((t - t0)**2) / (2 * sigma**2))

# 3. Hamiltonian Construction
H_static = jnp.zeros((8, 8), dtype=jnp.complex128)
H_drive = jnp.zeros((8, 8), dtype=jnp.complex128)

# H_static (Energies & Cavity)
H_static = H_static.at[2, 2].set(Delta_L1)
H_static = H_static.at[3, 3].set(Delta_L_minus)
H_static = H_static.at[6, 6].set(E_7)
H_static = H_static.at[7, 7].set(E_8)

# Ideal Cavity
H_static = H_static.at[2, 4].set(g)
H_static = H_static.at[2, 5].set(g)
# Anomalous Cavity from T+
H_static = H_static.at[2, 6].set(eps_H * g)
H_static = H_static.at[2, 7].set(eps_V * g)
# Anomalous Cavity from T-
H_static = H_static.at[3, 5].set(eps_H * g)
H_static = H_static.at[3, 4].set(eps_V * g)

# Make H_static Hermitian
H_static = H_static + jnp.conj(H_static.T)

# H_drive (Laser Couplings)
# To T+
H_drive = H_drive.at[2, 0].set(1.0 + eps_V)
H_drive = H_drive.at[2, 1].set(1.0 + eps_H)
# To T-
H_drive = H_drive.at[3, 0].set(eps_H)
H_drive = H_drive.at[3, 1].set(eps_V)

# Make H_drive Hermitian
H_drive = H_drive + jnp.conj(H_drive.T)

# 4. Diffrax Integration (State Vector & Implicit Solver)
psi0 = jnp.zeros(8, dtype=jnp.complex128)
psi0 = psi0.at[0].set(1.0 / jnp.sqrt(2.0))
psi0 = psi0.at[1].set(1.0 / jnp.sqrt(2.0))

def vector_field(t, psi, args):
    H_t = H_static + pulse_Omega(t) * H_drive
    return -1j * jnp.matmul(H_t, psi)

import optimistix

term = diffrax.ODETerm(vector_field)
solver = diffrax.Kvaerno5()
stepsize_controller = diffrax.PIDController(rtol=1e-10, atol=1e-12)
saveat = diffrax.SaveAt(ts=jnp.linspace(0, 5000, 2000))

@jax.jit
def solve_system():
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=5000.0,
        dt0=0.1,
        y0=psi0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=5000000
    )

print("Compiling and running Diffrax implicit integration...")
solution = solve_system()
print("Integration complete!")

# 5. Outputs
ys = solution.ys
ts = solution.ts
t_ns = ts / 1519.26

populations = jnp.abs(ys)**2

labels = [
    r'|0> ($\uparrow_x, 0$)',
    r'|1> ($\downarrow_x, 0$)',
    r'|2> ($T_+, 0$)',
    r'|3> ($T_-, 0$)',
    r'|4> ($\downarrow_x, 1V$)',
    r'|5> ($\uparrow_x, 1H$)',
    r'|6> ($\downarrow_x, 1H$)',
    r'|7> ($\uparrow_x, 1V$)'
]

plt.figure(figsize=(10, 6))
for i in range(8):
    plt.plot(t_ns, populations[:, i], label=labels[i])
plt.xlabel('Time (ns)')
plt.ylabel('Population')
plt.title('8-Level State Vector Dynamics (JAX/Diffrax - Implicit Kvaerno5)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.savefig('jax_8level_implicit_populations.png')
print("Saved jax_8level_implicit_populations.png")

pop_0_final = populations[-1, 0]
pop_1_final = populations[-1, 1]

print("\n--- Final Ground State Populations ---")
print(f"State 0: {pop_0_final:.6f}")
print(f"State 1: {pop_1_final:.6f}")
print(f"Difference: {abs(pop_0_final - pop_1_final):.6f}")
