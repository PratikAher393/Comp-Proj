import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def domain_wall_dynamics(t, y, alpha, T, u, k, Kd, Ms, gamma, mu0, delta):
    """
    Coupled differential equations for domain wall dynamics (Eqs. 2 and 3).
    """
    X, phi = y
    dXdt = (gamma * (Kd/(mu0*Ms)) * np.sin(2*phi) - T*u + (1-T)*alpha*delta*u*k) / (1 + alpha**2)
    dphidt = (-gamma * alpha * (Kd/(mu0*Ms)) * np.sin(2*phi) + (1-T)*u*k + T*alpha*u/delta) / (1 + alpha**2)
    return [dXdt, dphidt]

def main(frequencies, T_interp, rho_interp):
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

        # Interpolate T from Figure 4
        T = float(T_interp(frequency/1e9))  # T_interp expects frequency in GHz
        rho = float(rho_interp(frequency/1e9))

        # Estimate vg (Group Velocity) - You might need to calibrate or have a more accurate model
        vg = 1000  #m/s

        #CALIBRATE U --  This is the most important parameter to calibrate.
        u = 0.001 * rho  # Example value -- you MUST calibrate this

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

# ---------------------- Data Interpolation ----------------------
# Load digitized data
data_figure4 = np.array([
    [20, 0.40, 1.00],
    [22, 0.40, 0.98],
    [24, 0.41, 0.96],
    [26, 0.42, 0.93],
    [28, 0.44, 0.90],
    [30, 0.47, 0.86],
    [32, 0.50, 0.82],
    [34, 0.54, 0.78],
    [36, 0.58, 0.73],
    [38, 0.62, 0.68],
    [40, 0.67, 0.63],
    [42, 0.72, 0.58],
    [44, 0.77, 0.53],
    [46, 0.82, 0.48],
    [48, 0.87, 0.43],
    [50, 0.92, 0.38],
    [52, 0.95, 0.33],
    [54, 0.97, 0.28],
    [56, 0.98, 0.23],
    [58, 0.98, 0.18],
    [60, 0.98, 0.13],
    [62, 0.98, 0.08],
    [64, 0.98, 0.03],
    [66, 0.98, 0.02],
    [68, 0.98, 0.01],
    [70, 0.98, 0.00]
])

# Separate frequency, T, and rho data
frequency = data_figure4[:, 0]
T_data = data_figure4[:, 1]
rho_data = data_figure4[:, 2]

# Create interpolation functions
T_interp = interp1d(frequency, T_data, kind='linear', fill_value="extrapolate")
rho_interp = interp1d(frequency, rho_data, kind='linear', fill_value="extrapolate")

# Example usage: Specify the frequencies you want to simulate
frequencies = [22e9, 70e9, 30e9, 40e9, 60e9]  # Example frequencies in Hz

# Run the simulation
main(frequencies, T_interp, rho_interp)

