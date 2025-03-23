import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def domain_wall_dynamics(t, y, alpha, T, u, k, Kd, Ms, gamma, mu0, delta):
    """
    Coupled differential equations for domain wall dynamics.
    """
    X, phi = y
    dXdt = (gamma * (Kd/(mu0*Ms)) * np.sin(2*phi) - T*u + (1-T)*alpha*delta*u*k) / (1 + alpha**2)
    dphidt = (-gamma * alpha * (Kd/(mu0*Ms)) * np.sin(2*phi) + (1-T)*u*k + T*alpha*u/delta) / (1 + alpha**2)
    return [dXdt, dphidt]

def main():
    """
    Generates the model calculation curves for Figure 2.
    """
    # Physical constants
    mu0 = 4 * np.pi * 1e-7
    gamma = 2.21e5
    
    # Material parameters (consistent with paper)
    Ms = 8.6e5
    Aex = 1.3e-11
    K1 = 5.8e5
    alpha = 0.01
    delta = np.pi * np.sqrt(Aex/K1)
    
    # Calculated parameters
    Nx_minus_Ny = 0.05
    Kd = 0.5 * mu0 * Ms**2 * Nx_minus_Ny
    
    # Time span
    t_span = (0, 50e-9)
    t_eval = np.linspace(0, 50e-9, 500)

    # Initial conditions
    y0 = [0, np.pi/2]

    # Parameters for 22 GHz case
    T_22GHz = 0.4  # Transmission coefficient (from paper)
    f_22GHz = 22e9
    vg_22GHz = 1000
    k_22GHz = 2*np.pi*f_22GHz/vg_22GHz
    u_22GHz = 35   # Adjusted based on data

    # Parameters for 70 GHz case
    T_70GHz = 0.98  # Transmission coefficient (from paper)
    f_70GHz = 70e9
    vg_70GHz = 2000
    k_70GHz = 2*np.pi*f_70GHz/vg_22GHz
    u_70GHz = 16   # Adjusted based on data

    # Solve for 22 GHz case with damping (alpha=0.01)
    sol_22GHz_with_damping = solve_ivp(
        lambda t, y: domain_wall_dynamics(t, y, 0.01, T_22GHz, u_22GHz, k_22GHz, 
                                         Kd, Ms, gamma, mu0, delta),
        t_span, y0, method='RK45', t_eval=t_eval, dense_output=True
    )

    # Solve for 70 GHz case with damping (alpha=0.01)
    sol_70GHz_with_damping = solve_ivp(
        lambda t, y: domain_wall_dynamics(t, y, 0.01, T_70GHz, u_70GHz, k_70GHz, 
                                        Kd, Ms, gamma, mu0, delta),
        t_span, y0, method='RK45', t_eval=t_eval, dense_output=True
    )

    # Generate Plot
    plt.figure(figsize=(10, 6))

    # Plot 22 GHz data
    plt.plot(sol_22GHz_with_damping.t * 1e9, sol_22GHz_with_damping.y[0] * 1e9, 'r-', label='22 GHz (Model α=0.01)')

    # Plot 70 GHz data
    plt.plot(sol_70GHz_with_damping.t * 1e9, sol_70GHz_with_damping.y[0] * 1e9, 'b-', label='70 GHz (Model α=0.01)')

    # Customize Plot
    plt.xlabel('Time (ns)')
    plt.ylabel('Wall Displacement (nm)')
    plt.title('Domain Wall Displacement vs. Time (Model Calculation)')
    plt.xlim(0, 50)
    plt.ylim(-60, 800) #Adjust as needed
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save and Display
    plt.savefig('domain_wall_motion_model_only.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

