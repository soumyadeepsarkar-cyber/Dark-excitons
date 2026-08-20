import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import diffrax

# 2. Parameter Dictionary (The PyTree)
params = {
    'Delta': jnp.array([1500.0, 1510.0]),
    'gamma': 0.0,
    'E_4': 130.0,
    'E_5': -130.0,
    'g': 20.0,
    'eps_H': 0.10 * jnp.exp(1j * 0.0),
    'eps_V': 0.10 * jnp.exp(1j * jnp.pi / 4.0),
    'Omega_max': 150.0,
    't0': 2500.0,
    'sigma': 650.0
}

# 3. Static Coupling Matrices
def build_couplings(p):
    V_L = jnp.zeros((6, 2), dtype=jnp.complex128)
    V_C = jnp.zeros((6, 2), dtype=jnp.complex128)
    
    # V_L
    V_L = V_L.at[0, 0].set(1.0 + p['eps_V'])
    V_L = V_L.at[1, 0].set(1.0 + p['eps_H'])
    V_L = V_L.at[0, 1].set(p['eps_H'])
    V_L = V_L.at[1, 1].set(p['eps_V'])
    
    # V_C
    V_C = V_C.at[2, 0].set(p['g'])
    V_C = V_C.at[3, 0].set(p['g'])
    V_C = V_C.at[4, 0].set(p['eps_H'] * p['g'])
    V_C = V_C.at[5, 0].set(p['eps_V'] * p['g'])
    V_C = V_C.at[2, 1].set(p['eps_V'] * p['g'])
    V_C = V_C.at[3, 1].set(p['eps_H'] * p['g'])
    
    return V_L, V_C

# 4. The Dynamic Vector Field (Adiabatic Elimination)
def vector_field(t, rho, args):
    p = args
    current_Omega = p['Omega_max'] * jnp.exp(-((t - p['t0'])**2) / (2 * p['sigma']**2))
    V_L, V_C = build_couplings(p)
    V_tot = current_Omega * V_L + V_C
    
    shift_0 = -jnp.real(jnp.abs(V_tot[0, 0])**2 / (p['Delta'][0] - 1j*p['gamma']/2) + jnp.abs(V_tot[0, 1])**2 / (p['Delta'][1] - 1j*p['gamma']/2))
    shift_1 = -jnp.real(jnp.abs(V_tot[1, 0])**2 / (p['Delta'][0] - 1j*p['gamma']/2) + jnp.abs(V_tot[1, 1])**2 / (p['Delta'][1] - 1j*p['gamma']/2))
    E_base = jnp.array([shift_0, shift_1, 0.0, 0.0, p['E_4'], p['E_5']])
    
    H_eff = jnp.zeros((6, 6), dtype=jnp.complex128)
    for k in range(2):
        V_k = V_tot[:, k]
        coupling_matrix = jnp.outer(V_k, jnp.conj(V_k))
        denom = 0.5 * (1.0 / (p['Delta'][k] - E_base[:, None] - 1j*p['gamma']/2) + 1.0 / (p['Delta'][k] - E_base[None, :] - 1j*p['gamma']/2))
        H_eff += -coupling_matrix * denom
        
    E_bare = jnp.array([0.0, 0.0, 0.0, 0.0, p['E_4'], p['E_5']])
    H_eff += jnp.diag(E_bare)
    
    return -1j * (H_eff @ rho - rho @ jnp.conj(H_eff).T)

# 5. Integration and Execution
rho0 = jnp.zeros((6, 6), dtype=jnp.complex128)
rho0 = rho0.at[0, 0].set(0.5)
rho0 = rho0.at[1, 1].set(0.5)
rho0 = rho0.at[0, 1].set(0.5)
rho0 = rho0.at[1, 0].set(0.5)

term = diffrax.ODETerm(vector_field)
solver = diffrax.Dopri5()
stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)
saveat = diffrax.SaveAt(ts=jnp.linspace(0, 5000, 2000))

@jax.jit
def solve_system(p):
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=5000.0,
        dt0=0.1,
        y0=rho0,
        args=p,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=5000000
    )

print("Starting JAX/Diffrax compilation and execution for Adiabatic Elimination...")
solution = solve_system(params)
print("Integration complete!")

# 6. Outputs
ys = solution.ys
ts = solution.ts
t_ns = ts / 1519.26

populations = jnp.real(jnp.diagonal(ys, axis1=1, axis2=2))

labels = [
    r'0: |$\uparrow_x, 0$>',
    r'1: |$\downarrow_x, 0$>',
    r'2: |$\downarrow_x, 1V$>',
    r'3: |$\uparrow_x, 1H$>',
    r'4: |$\downarrow_x, 1H$>',
    r'5: |$\uparrow_x, 1V$>'
]

plt.figure(figsize=(10, 6))
for i in range(6):
    plt.plot(t_ns, populations[:, i], label=labels[i])
plt.xlabel('Time (ns)')
plt.ylabel('Population')
plt.title('6-Level vSTIRAP Dynamics (JAX/Diffrax - Adiabatic Elimination)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.savefig('jax_6level_ae_populations.png')
print("Saved jax_6level_ae_populations.png")

pop_0_final = populations[-1, 0]
pop_1_final = populations[-1, 1]

print("\n--- Final Ground State Populations ---")
print(f"State 0: {pop_0_final:.6f}")
print(f"State 1: {pop_1_final:.6f}")
print(f"Difference: {abs(pop_0_final - pop_1_final):.6f}")
