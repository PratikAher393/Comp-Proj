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
    Main function to reproduce Figure 2 from the paper.
    """
    # Physical constants
    mu0 = 4 * np.pi * 1e-7
    gamma = 2.21e5
    
    # Material parameters
    Ms = 8.6e5
    Aex = 1.3e-11
    K1 = 5.8e5
    alpha = 0.01
    delta = np.pi * np.sqrt(Aex/K1)
    
    # Calculated parameters
    Nx_minus_Ny = 0.05
    Kd = 0.5 * mu0 * Ms**2 * Nx_minus_Ny
    
    # Time span
    t_span = (0, 50e-9)  # Extend to 50 ns to match the figure
    t_eval = np.linspace(0, 50e-9, 500)  # Increase points for smoother curves

    # Initial conditions
    y0 = [0, np.pi/2]

    # Parameters for 22 GHz case
    T_22GHz = 0.4
    f_22GHz = 22e9
    vg_22GHz = 1000
    k_22GHz = 2*np.pi*f_22GHz/vg_22GHz
    u_22GHz = 60  # Calibrated to reduce fluctuations

    # Parameters for 70 GHz case
    T_70GHz = 0.98
    f_70GHz = 70e9
    vg_70GHz = 2000
    k_70GHz = 2*np.pi*f_70GHz/vg_70GHz
    u_70GHz = 30 # Calibrated to reduce fluctuations

    # Solve for 22 GHz case with alpha=0.01
    sol_22GHz_with_damping = solve_ivp(
        lambda t, y: domain_wall_dynamics(t, y, 0.01, T_22GHz, u_22GHz, k_22GHz, 
                                         Kd, Ms, gamma, mu0, delta),
        t_span, y0, method='RK45', t_eval=t_eval
    )

    # Solve for 70 GHz case with alpha=0.01
    sol_70GHz_with_damping = solve_ivp(
        lambda t, y: domain_wall_dynamics(t, y, 0.01, T_70GHz, u_70GHz, k_70GHz, 
                                        Kd, Ms, gamma, mu0, delta),
        t_span, y0, method='RK45', t_eval=t_eval
    )

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Plot for both 22 GHz and 70 GHz on the same plot
    plt.plot(sol_22GHz_with_damping.t * 1e9, sol_22GHz_with_damping.y[0] * 1e9, 'r-', 
             linewidth=2, label='22 GHz (Model α=0.01)')
    plt.plot(sol_70GHz_with_damping.t * 1e9, sol_70GHz_with_damping.y[0] * 1e9, 'b-', 
             linewidth=2, label='70 GHz (Model α=0.01)')

    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('Wall displacement (nm)', fontsize=12)
    plt.title('Domain Wall Displacement vs. Time', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.xlim(0, 50)
    plt.ylim(-60, 1200)  # Set y-axis limits

    plt.tight_layout()
    plt.savefig('3domain_wall_motion_combined.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

