import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------
# Drone + environment parameters
# ----------------------------
g = 9.81                # gravity (m/s^2)
rho = 1.225             # air density (kg/m^3)
C_dx = 0.8              # Drag coefficient horizontal
C_dz = 1                # Drag coefficient vertical
A_front = 0.1           # Effective front area of drone (m^2)
A_top = 0.25            # Effective top area of drone (m^2)
A_disk = 0.25           # total rotor disk area (m^2)
eta = 0.75              # overall efficiency (motor * prop)
Vb = 22.2               # battery voltage (V)
Cb = 5.0                # battery capacity (Ah)
P_av = 5                # avionics power (W)
usable_frac = 0.8
E_avail = usable_frac * Vb * Cb * 3600  # convert Wh → J

m_frame = 1.5
m_payload = 0.5
m_battery = 0.5
m_tot = m_frame + m_payload + m_battery

# ----------------------------
# Flight targets
# ----------------------------
vx_target = 15.0       # forward cruise (m/s)
x_target = 2000        # target horizontal distance (m)
z_target = 0
transition_time = 3    # tilt transition (s)

# Braking tilt
theta_cruise = np.deg2rad(25)  # cruising tilt
theta_brake = np.deg2rad(-20)  # braking tilt (negative)
theta_level = np.deg2rad(0)    # level

# ----------------------------
# Helper function: braking distance
# ----------------------------
def braking_distance(v0, theta_b):
    x = 0
    v = v0
    dt = 0.01
    max_steps = 10000  # prevent infinite loop
    step = 0

    while v > 0.01 and step < max_steps:
        Drag_x = 0.5 * rho * C_dx * A_front * v**2
        a = (m_tot * g * np.sin(theta_b) - Drag_x)/m_tot
        v += a*dt
        if v < 0:  # prevent going negative
            v = 0
        x += v*dt
        step += 1

    if step >= max_steps:
        print("Warning: braking_distance reached max steps, check theta_b")
    return x

# Compute braking distance and cruise start
d_brake = braking_distance(vx_target, theta_brake)
x_cruise_start = x_target - d_brake 
print(f"Active braking distance ≈ {d_brake:.1f} m | Begin braking at x ≈ {x_cruise_start:.1f} m")

# ----------------------------
# ODEs for flight
# state vector y = [z, vz, x, vx, E]
# ----------------------------
def drone_dynamics(t, y):
    z, vz, x, vx, E = y

    # Horizontal drag
    Drag_x = 0.5 * rho * C_dx * A_front * vx**2
    # Vertical drag
    Drag_z = 0.5 * rho * C_dz * A_top * vz**2

    # Determine tilt
    if x < x_cruise_start:
        theta = theta_cruise
        u_x = 0.0
    elif x < x_target:
    # Only apply braking while moving forward
        theta = theta_brake if vx > 0 else 0.0
        u_x = 0.0
    else:
        theta = theta_level
        vx = 0
        u_x = 0.0

    # Vertical thrust to maintain z_target
    k_p_z = 10.0
    Thrust = m_tot * (g + k_p_z * (z_target - z)) / np.cos(theta)

    # Induced power (momentum theory)
    vi = np.sqrt(Thrust / (2 * rho * A_disk))
    P_ind = Thrust * vi
    P_par = abs(Drag_x * vx)
    P_elec = (P_ind + P_par)/eta + P_av

    # Dynamics
    dzdt = vz
    dvzdt = (Thrust*np.cos(theta) - m_tot*g - Drag_z)/m_tot
    dxdt = vx
    dvxdt = (Thrust*np.sin(theta) + u_x - Drag_x)/m_tot
    dEdt = -P_elec

    return [dzdt, dvzdt, dxdt, dvxdt, dEdt]



# ----------------------------
# Integration setup
# ----------------------------
y0 = [0, 0, 0, 0, E_avail]
t_span = (0, 200)
t_eval = np.linspace(t_span[0], t_span[1], 5000)

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
P_loss = (E_avail - E)/3600

# Stop at target distance
idx_arr = np.where(x >= x_target-0.3)[0]
if len(idx_arr) > 0:
    idx = idx_arr[0]
    z = z[:idx+1]
    vz = vz[:idx+1]
    x = x[:idx+1]
    vx = vx[:idx+1]
    E = E[:idx+1]
    t = t[:idx+1]
    P_loss = P_loss[:idx+1]

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

print(f"Reached velocity: {vx[-1]:.2f} m/s in {t[-1]:.2f} s")
print(f"Traveled distance: {x[-1]:.2f} m in {t[-1]:.2f} s")
print(f"Altitude gained: {z[-1]:.2f} m in {t[-1]:.2f} s")
print(f"Energy used: {(E_avail - E[-1])/3600:.2f} Wh ({100*(E_avail-E[-1])/E_avail:.2f} % of battery)")
