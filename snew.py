import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def domain_wall_dynamics(t, y, alpha, T, u, k, Kd, Ms, gamma, mu0, delta):
    """
    Coupled differential equations for domain wall dynamics (Eqs. 2 and 3 from the paper).
    """
    X, phi = y
    dXdt = (gamma * (Kd/(mu0*Ms)) * np.sin(2*phi) - T*u + (1-T)*alpha*delta*u*k) / (1 + alpha**2)
    dphidt = (-gamma * alpha * (Kd/(mu0*Ms)) * np.sin(2*phi) + (1-T)*u*k + T*alpha*u/delta) / (1 + alpha**2)
    return [dXdt, dphidt]

def main(frequencies):
    """
    Solves the model equations and plots domain wall displacement vs. time for different frequencies.
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
    y0 = [0, np.pi/2]  # X=0, phi=pi/2 (Bloch wall)

    # Store results for each frequency
    results = {}

    for frequency in frequencies:
        print(f"Solving for frequency: {frequency/1e9} GHz")

        # Frequency-dependent parameters (T, u, k) - THESE NEED CALIBRATION
        if frequency == 22e9:
            T = 0.4
            vg = 1000
            u = 35  # Needs Calibration - IMPORTANT
        elif frequency == 70e9:
            T = 0.98
            vg = 2000
            u = 16  # Needs Calibration - IMPORTANT
        else:
            #  Interpolate T, vg, u from Figure 4 if possible - REPLACE THIS
            #  This is just a placeholder - YOU MUST DIGITIZE FIGURE 4 AND IMPLEMENT PROPER INTERPOLATION
            T = 0.7  # Example Value - Needs to be based on Figure 4
            vg = 1500 #Example value - Needs to be based on known physics
            u = 25  # Example Value - Needs to be calibrated!
            print(f"  Using placeholder T, vg, u values - Calibration is CRITICAL!")

        k = 2 * np.pi * frequency / vg

        # Solve the equations
        sol = solve_ivp(
            lambda t, y: domain_wall_dynamics(t, y, alpha, T, u, k, Kd, Ms, gamma, mu0, delta),
            t_span, y0, method='RK45', t_eval=t_eval, dense_output=True
        )

        # Store the results
        results[frequency] = sol

    # Plotting
    plt.figure(figsize=(10, 6))
    for frequency, sol in results.items():
        plt.plot(sol.t * 1e9, sol.y[0] * 1e9, label=f"{frequency/1e9} GHz")

    plt.xlabel('Time (ns)')
    plt.ylabel('Wall Displacement (nm)')
    plt.title('Domain Wall Displacement vs. Time (Model Calculation)')
    plt.xlim(0, 50)
    plt.ylim(-60, 800) #Adjust as needed
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('domain_wall_motion_model_frequencies.png', dpi=300)
    plt.show()

# Example usage: Specify the frequencies you want to simulate
frequencies = [22e9, 70e9, 30e9, 40e9, 60e9]  # Example frequencies in Hz
main(frequencies)
