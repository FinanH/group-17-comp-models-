


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------
# Drone + environment parameters
# ----------------------------
g = 9.81                # gravity (m/s^2)
rho = 1.225             # air density (kg/m^3)
C_dx = 0.8              # Drag coefficient horizontal
C_dz = 1                # Drag coefficient verically
A_front = 0.1           # Effective front area of drone (m^2)
A_top = 0.25            # Effective top area of drone (m^2)
A_disk = 0.25           # total rotor disk area (m^2) ~ four 0.14 m rotors
eta = 0.75              # overall efficiency (motor * prop)
Vb = 22.2               # battery voltage (V)
Cb = 5.0                # battery capacity (Ah)
P_av = 5                # Avionics power (W)
usable_frac = 0.8       # usable fraction of battery
E_avail = usable_frac * Vb * Cb * 3600  # convert Wh → J

m_frame = 1.5           # frame mass (kg)
m_payload = 0.5         # payload mass (kg)
m_battery = 0.5         # battery mass (kg)
m_tot = m_frame + m_payload + m_battery  # total mass (kg)

# Desired takeoff climb speed target
vx_target = 15.0            # foward cruise (m/s)
x_target = 2000
vz_target = 0
z_target = 0
transition_time = 3        #  tilt time (s)


# ----------------------------
# Compute drag-braking distance for 15 → ~0.5 m/s
# ----------------------------
v_end = 0.5  # near rest
d_brake = (2*m_tot/(rho*C_dx*A_front)) * np.log(vx_target/v_end)
x_coast_start = x_target - d_brake
print(f"Braking distance ≈ {d_brake:.1f} m | Begin coasting at x ≈ {x_coast_start:.1f} m")


# ----------------------------
# ODEs for takeoff phase
# state vector y = [z, vz, E]
# ----------------------------

def drone_dynamics(t, y):
    z, vz, x, vx, E = y
   
    Drag_target = 0.5 * rho * C_dx * A_front * (vx_target)**2  # Drag at target speed
    Drag_z = 0.5 * rho * C_dz * A_top * vz**2                  # vertical drag (N)
    Drag_x = 0.5 * rho * C_dx * A_front * vx**2                # horizontal drag (N)
    
    
    theta_req =  np.arctan(Drag_target / (m_tot * g))  # radians
    theta = theta_req * min(t / transition_time, 1.0)
    
    # --- Control law / thrust command ---
    # Simple proportional controller on vertical speed:
        
    
  
    k_p_x = 2.0        # if vx not euquak to vx target increase acceleration by 1
    k_p_z = 10.0
    
    theta_brake = np.deg2rad(0)
    
    if x >= x_coast_start:
        # level out, no horizontal thrust or speed control
        theta = theta_brake
        u_x = 0
    else:
        # normal forward acceleration control
        u_x = k_p_x * (vx_target - vx)
    
    
    
    Thrust = (m_tot * (g + (k_p_z * (z_target - z)))) / np.cos(theta)    # Thurst (N)
    
    
    # --- Power model ---
    # Induced velocity (momentum theory)
    vi = np.sqrt(Thrust / (2 * rho * A_disk))
    P_ind = Thrust * vi                     # induced power (W)
    P_par = abs(Drag_x * vx)
    P_elec = ((P_ind + P_par) / eta) + P_av      # electrical power 

    # --- Dynamics ---
    dzdt = vz
    dvzdt = (Thrust*np.cos(theta) - (m_tot * g) - Drag_z )/ m_tot
    dxdt = vx
    dvxdt = ((Thrust * np.sin(theta) + u_x) - Drag_x) / m_tot
    dEdt = -P_elec

    return [dzdt, dvzdt, dxdt, dvxdt, dEdt]

def plotting_drone_moving():
    # ----------------------------
    # Integration setup
    # ----------------------------
    y0 = [0, 0, 0, 0, E_avail]                    # initial altitude, vertical speed, energy
    t_span = (0, 1000)                              # simulate up to 50 s (should reach speed)
    t_eval = np.linspace(t_span[0], t_span[1], 3000)
    
    sol = solve_ivp(drone_dynamics, t_span, y0, t_eval=t_eval, rtol=1e-6, atol=1e-8)
    
    # ----------------------------
    # Extract results
    # ----------------------------
    z = sol.y[0]
    vz = sol.y[1]
    x = sol.y[2]
    vx = sol.y[3]
    E = sol.y[4]
    t = sol.t
    P_loss = (E_avail - E) / 3600  # convert J to Wh
    
    # Stop when target distance reached
    idx_arr = np.where(x >= x_target)[0]
    if len(idx_arr) > 0:
        idx = idx_arr[0]
        z = z[:idx]
        vz = vz[:idx]
        x = x[:idx]
        vx = vx[:idx]
        E = E[:idx]
        t = t[:idx]
        P_loss = P_loss[:idx]
    else:
        print("Warning: x never reached x_target during simulation.")
    
    # ----------------------------
    # Plot results
    # ----------------------------
    plt.figure(figsize=(10,6))
    
    plt.subplot(3,1,1)
    plt.plot(t, x, label='Displacement x(t)')
    plt.ylabel('Displacement (m)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(t, vx, label='Horizontal Speed')
    plt.axhline(vx_target, color='r', linestyle='--', label='Target speed')
    plt.ylabel('v_x (m/s)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3,1,3)
    plt.plot(t, P_loss, label='Energy used')
    plt.ylabel('Energy used (Wh)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Reached velocity: {vx[-1]:.2f} m in {t[-1]:.2f} s")
    print(f"Traveled distance: {x[-1]:.2f} m in {t[-1]:.2f} s")
    print(f"Altitude gained: {z[-1]:.2f} m in {t[-1]:.2f} s")
    print(f"Energy used: {(E_avail - E[-1])/3600:.2f} Wh ({100*(E_avail-E[-1])/E_avail:.2f} % of battery)")

