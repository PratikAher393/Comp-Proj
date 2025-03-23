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
    Main function to reproduce Figure 2 from the paper, aligning with spin current and simulation.
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
    k_70GHz = 2*np.pi*f_70GHz/vg_70GHz
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

    # Experimental data - Digitized Data (from image)
    # Micromagnetic simulation data for 22 GHz
    mm_22GHz_t = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Time in ns
    mm_22GHz_x = [0, 44, 132, 246, 352, 440, 512, 574, 628, 674, 714]  # Displacement in nm

    # Micromagnetic simulation data for 70 GHz
    mm_70GHz_t = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Time in ns
    mm_70GHz_x = [0, -5.2, -11, -17, -23, -29, -35, -41, -47, -53, -59]  # Displacement in nm

    # Spin-polarized electrical current data for 70 GHz
    sp_70GHz_t = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Time in ns
    sp_70GHz_x = [0, -5, -11, -17, -23, -29, -35, -41, -47, -53, -59]  # Displacement in nm

    # Interpolate model data to match experimental data points
    interp_t_22GHz = np.array(mm_22GHz_t) * 1e-9  # Convert ns to s
    interp_x_22GHz = sol_22GHz_with_damping.sol(interp_t_22GHz)[0] * 1e9  # Interpolate and convert to nm

    interp_t_70GHz = np.array(mm_70GHz_t) * 1e-9  # Convert ns to s
    interp_x_70GHz = sol_70GHz_with_damping.sol(interp_t_70GHz)[0] * 1e9  # Interpolate and convert to nm

    # Plot the results
    plt.figure(figsize=(12, 6))

    # 22 GHz Plot
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    plt.plot(sol_22GHz_with_damping.t * 1e9, sol_22GHz_with_damping.y[0] * 1e9, 'r-', label='Model α=0.01')
    #plt.plot(sol_22GHz_no_damping.t * 1e9, sol_22GHz_no_damping.y[0] * 1e9, 'g--', label='Model α=0') #Removed this line
    plt.plot(mm_22GHz_t, mm_22GHz_x, 'ks', label='Simulation')
    plt.plot(interp_t_22GHz * 1e9, interp_x_22GHz, 'm.', label='Model Interp') #Added this line
    plt.xlabel('Time (ns)')
    plt.ylabel('Wall Displacement (nm)')
    plt.title('(a) 22 GHz')
    plt.xlim(0, 50)
    plt.ylim(-60, 800)
    plt.legend()
    plt.grid(True)

    # 70 GHz Plot
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    plt.plot(sol_70GHz_with_damping.t * 1e9, sol_70GHz_with_damping.y[0] * 1e9, 'r-', label='Model α=0.01')
    #plt.plot(sol_70GHz_no_damping.t * 1e9, sol_70GHz_no_damping.y[0] * 1e9, 'g--', label='Model α=0') #Removed this line
    plt.plot(mm_70GHz_t, mm_70GHz_x, 'ks', label='Simulation')
    plt.plot(sp_70GHz_t, sp_70GHz_x, 'bo', label='Spin-polarized current')  # Plot spin-polarized data
    plt.plot(interp_t_70GHz * 1e9, interp_x_70GHz, 'm.', label='Model Interp') #Added this line
    plt.xlabel('Time (ns)')
    plt.ylabel('Wall Displacement (nm)')
    plt.title('(b) 70 GHz')
    plt.xlim(0, 50)
    plt.ylim(-60, 20)  # Corrected limits
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # Adjust subplot parameters for a tight layout
    plt.savefig('domain_wall_motion_final.png', dpi=300)
    plt.show()
if __name__ == "__main__":
    main()

