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
    u_22GHz = 50

    # Parameters for 70 GHz case
    T_70GHz = 0.98
    f_70GHz = 70e9
    vg_70GHz = 2000
    k_70GHz = 2*np.pi*f_70GHz/vg_70GHz
    u_70GHz = 20

    # Solve for 22 GHz case with alpha=0.01
    sol_22GHz_with_damping = solve_ivp(
        lambda t, y: domain_wall_dynamics(t, y, 0.01, T_22GHz, u_22GHz, k_22GHz, 
                                         Kd, Ms, gamma, mu0, delta),
        t_span, y0, method='RK45', t_eval=t_eval
    )

    # Solve for 22 GHz case with alpha=0
    sol_22GHz_no_damping = solve_ivp(
        lambda t, y: domain_wall_dynamics(t, y, 0, T_22GHz, u_22GHz, k_22GHz, 
                                        Kd, Ms, gamma, mu0, delta),
        t_span, y0, method='RK45', t_eval=t_eval
    )

    # Solve for 70 GHz case with alpha=0.01
    sol_70GHz_with_damping = solve_ivp(
        lambda t, y: domain_wall_dynamics(t, y, 0.01, T_70GHz, u_70GHz, k_70GHz, 
                                        Kd, Ms, gamma, mu0, delta),
        t_span, y0, method='RK45', t_eval=t_eval
    )

    # Solve for 70 GHz case with alpha=0
    sol_70GHz_no_damping = solve_ivp(
        lambda t, y: domain_wall_dynamics(t, y, 0, T_70GHz, u_70GHz, k_70GHz, 
                                       Kd, Ms, gamma, mu0, delta),
        t_span, y0, method='RK45', t_eval=t_eval
    )

    # Extract micromagnetic simulation data points from paper
    micro_data_22GHz = np.array([
        [0, 0], [2.5, 15], [5, 44], [7.5, 83], [10, 132], 
        [12.5, 188], [15, 246], [17.5, 301], [20, 352], 
        [22.5, 398], [25, 440], [27.5, 478], [30, 512],
        [32.5, 544], [35, 574], [37.5, 602], [40, 628],
        [42.5, 652], [45, 674], [47.5, 695], [50, 714]
    ])

    micro_data_70GHz = np.array([
        [0, 0], [2.5, -2.5], [5, -5.2], [7.5, -8], [10, -11], 
        [12.5, -14], [15, -17], [17.5, -20], [20, -23], 
        [22.5, -26], [25, -29], [27.5, -32], [30, -35],
        [32.5, -38], [35, -41], [37.5, -44], [40, -47],
        [42.5, -50], [45, -53], [47.5, -56], [50, -59]
    ])

    # Extract spin-polarized current simulation data from paper
    current_data_22GHz = np.array([
        [0, 0], [5, 42], [10, 130], [15, 240], 
        [20, 350], [25, 435], [30, 510],
        [35, 570], [40, 630], [45, 680], [50, 720]
    ])

    current_data_70GHz = np.array([
        [0, 0], [5, -5], [10, -11], [15, -17], 
        [20, -23], [25, -29], [30, -35],
        [35, -41], [40, -47], [45, -53], [50, -59]
    ])

    # Plot the results
    plt.figure(figsize=(10, 8))

    # Plot for 22 GHz
    plt.subplot(2, 1, 1)
    plt.plot(sol_22GHz_with_damping.t * 1e9, sol_22GHz_with_damping.y[0] * 1e9, 'r-', 
             linewidth=2, label='Model α=0.01')
    plt.plot(sol_22GHz_no_damping.t * 1e9, sol_22GHz_no_damping.y[0] * 1e9, 'g--', 
             linewidth=2, label='Model α=0')
    plt.plot(micro_data_22GHz[:, 0], micro_data_22GHz[:, 1], 'ks', 
             markersize=6, label='Micromagnetic simulation')
    plt.plot(current_data_22GHz[:, 0], current_data_22GHz[:, 1], 'bo', 
             markersize=6, label='Spin-polarized current')
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('Wall displacement (nm)', fontsize=12)
    plt.title('(a) f = 22 GHz', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.xlim(0, 50)  # Set x-axis limit

    # Plot for 70 GHz
    plt.subplot(2, 1, 2)
    plt.plot(sol_70GHz_with_damping.t * 1e9, sol_70GHz_with_damping.y[0] * 1e9, 'r-', 
             linewidth=2, label='Model α=0.01')
    plt.plot(sol_70GHz_no_damping.t * 1e9, sol_70GHz_no_damping.y[0] * 1e9, 'g--', 
             linewidth=2, label='Model α=0')
    plt.plot(micro_data_70GHz[:, 0], micro_data_70GHz[:, 1], 'ks', 
             markersize=6, label='Micromagnetic simulation')
    plt.plot(current_data_70GHz[:, 0], current_data_70GHz[:, 1], 'bo', 
             markersize=6, label='Spin-polarized current')
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('Wall displacement (nm)', fontsize=12)
    plt.title('(b) f = 70 GHz', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.xlim(0, 50)  # Set x-axis limit

    plt.tight_layout()
    plt.savefig('2domain_wall_motion.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

