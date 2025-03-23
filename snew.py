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
        rho = float(rho_interp(frequency/1e9)) #Rho interpolation

        # Estimate vg (Group Velocity) - You might need to calibrate or have a more accurate model
        vg = 1000  #m/s Base value
        if frequency > 40e9: #Example effect 
            vg = 1000 + (frequency - 40e9) * (1000/30e9) #Linear approximation

        #CALIBRATE U --  This is the most important parameter to calibrate.
        u = 10 #Base value to start calibration
        if frequency > 40e9:
            u = 10 + (frequency -40e9)*(10/30e9) #example

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

# ---------------------- Data Digitization and Interpolation ----------------------
# **Step 1: Data Digitization**
# You will need to use a tool (e.g., WebPlotDigitizer, Engauge Digitizer) to extract the data points (frequency, T, rho) from Figure 4.
# Save the digitized data into CSV files or text files.  For example:
#   - freq_T_data.csv (two columns: frequency in GHz, Transmission Coefficient T)
#   - freq_rho_data.csv (two columns: frequency in GHz, Spin Wave Amplitude rho)

# **Step 2: Load Digitized Data**
freq_T_data = np.array([
    [20, 0.4], [30, 0.5], [40, 0.65], [50, 0.7], [60, 0.9], [70, 0.98]  # Example Data
])

freq_rho_data = np.array([
    [20, 1.0], [30, 0.8], [40, 0.6], [50, 0.4], [60, 0.3], [70, 0.2]  # Example Data
])

# **Step 3: Create Interpolation Functions**
#  Ensure frequencies are in increasing order for interpolation
freq_T_data = freq_T_data[np.argsort(freq_T_data[:, 0])]
freq_rho_data = freq_rho_data[np.argsort(freq_rho_data[:, 0])]

T_interp = interp1d(freq_T_data[:, 0], freq_T_data[:, 1], kind='linear', fill_value="extrapolate")
rho_interp = interp1d(freq_rho_data[:, 0], freq_rho_data[:, 1], kind='linear', fill_value="extrapolate")

# Example usage: Specify the frequencies you want to simulate
frequencies = [22e9, 70e9, 30e9, 40e9, 60e9]  # Example frequencies in Hz

# Run the simulation
main(frequencies, T_interp, rho_interp)

