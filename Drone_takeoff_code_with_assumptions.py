


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------
# Drone + environment parameters
# ----------------------------
g = 9.81                # gravity (m/s^2)
rho = 1.225             # air density (kg/m^3)
C_dz = 0.9                 # Drag coefficient
A_top = 0.175674            # Effective top area of drone (m^2)
A_disk = 1.34           # total rotor disk area (m^2) ~ six 0.223 m2 rotors
eta = 0.75              # overall efficiency (motor * prop)
Vb = 132.2               # battery voltage (V)
Cb = 27                # battery capacity (Ah)
P_av = 12                # Avionics power (W)
usable_frac = 0.9       # usable fraction of battery
E_avail = usable_frac * Vb * Cb * 3600  # convert Wh → J

m_frame = 5.93           # frame mass (kg)
m_payload = 6        # payload mass (kg)
m_battery = 3.57        # x6 battery mass (kg)
m_tot = m_frame + m_payload + m_battery

# Desired takeoff climb speed target
v_t_target = 2.0          # m/s (steady climb)
alt_target = 30.0       # m target altitude (stop integration here)

# ----------------------------
# ODEs for takeoff phase
# state vector y = [z, vz, E]
# ----------------------------

def takeoff_dynamics(t, y):
    z, vz, E = y

    # --- Control law / thrust command ---
    # Simple proportional controller on vertical speed:
    k_p = 6.0         # if vz not euqual to v target increase acceleration by 5
                            
    Thurst = m_tot * (g + k_p * (v_t_target - vz))  # Thurst (N)
    Drag_z = 0.5*rho*C_dz*A_top*vz* abs(vz)        # vertical drag (N)
   
    # --- Power model ---
   
    # Induced velocity (momentum theory)
    vi = np.sqrt(Thurst / (2 * rho * A_disk))
  
    P_ind = Thurst * vi                      # induced power (W)
    P_elec = (P_ind / eta) + P_av            # electrical power 

    # --- Dynamics ---
    dzdt = vz
    dvzdt = (Thurst - (m_tot * g) - Drag_z )/ m_tot
    dEdt = -P_elec

    return [dzdt, dvzdt, dEdt]

# ----------------------------
# Integration setup
# ----------------------------

y0 = [0.0, 0.0, E_avail]                      # initial altitude, vertical speed, energy
t_span = (0, 30)                              # simulate up to 30 s (should reach ~10 m)
t_eval = np.linspace(t_span[0], t_span[1], 300)

sol = solve_ivp(takeoff_dynamics, t_span, y0, t_eval=t_eval, rtol=1e-6, atol=1e-8)

# ----------------------------
# Extract results
# ----------------------------
z = sol.y[0]
vz = sol.y[1]
E = sol.y[2]
t = sol.t
P_loss = (E_avail - E) / 3600  # convert J to Wh

# Stop when target altitude reached
idx = np.where(z >= alt_target)[0][0]
z = z[:idx]
vz = vz[:idx]
E = E[:idx]
t = t[:idx]
P_loss = P_loss[:idx]

# ----------------------------
# Plot results
# ----------------------------

plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.plot(t, z, label='Altitude z(t)')
plt.ylabel('Altitude (m)')
plt.grid(True)
plt.legend()

plt.subplot(3,1,2)
plt.plot(t, vz, label='Vertical Speed')
plt.axhline(v_target, color='r', linestyle='--', label='Target speed')
plt.ylabel('v_z (m/s)')
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

print(f"Reached altitude: {z[-1]:.2f} m in {t[-1]:.2f} s")
print(f"Energy used: {(E_avail - E[-1])/3600:.2f} Wh ({100*(E_avail-E[-1])/E_avail:.2f} % of battery)")